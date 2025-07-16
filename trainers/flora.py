import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# from Dassl.dassl.engine import TRAINER_REGISTRY, TrainerX
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from torch.cuda import device
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict

_tokenizer = _Tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'FedPGP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FLORA.N_CTX
        ctx_init = cfg.TRAINER.FLORA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        bottleneck = cfg.TRAINER.FLORA.BOTTLENECK
        self.N = cfg.TRAINER.FLORA.N
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.FLORA.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")

                sigma = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)

                ctx_local = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)

                Q, R = torch.linalg.qr(ctx_local)

                B = Q[:, :, :bottleneck]
                A = R[:, :bottleneck, :]

                #sigma = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                #nn.init.normal_(sigma, std=0.02)

            nn.init.normal_(sigma, std=0.02)
            nn.init.normal_(B, std=0.02)
            nn.init.normal_(A, std=0.02)
            nn.init.normal_(ctx_local, std=0.02)



            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")


        prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.B = nn.Parameter(B)
        self.A = nn.Parameter(A)
        self.sigma = nn.Parameter(sigma)
        self.ctx_local = nn.Parameter(ctx_local)

        print("classnames:", classnames)
        classnames = [name.replace("_", " ") if isinstance(name, str) else name for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) if isinstance(name, str) else 0 for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("embedding", embedding)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FLORA.CLASS_TOKEN_POSITION

    def forward(self):

        B = self.B
        A = self.A
        ctx_local = self.ctx_local

        BA = torch.matmul(B, A)

        sigma = self.sigma

        ctx = BA+ self.sigma
        embedding = self.embedding


        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])


        if BA.dim() == 3:
            BA = BA.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        BA = BA.permute(1, 0, 2, 3)
        BA = BA.contiguous().view(self.N * self.n_cls, self.n_ctx, BA.shape[3])

        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        sigma = sigma.permute(1, 0, 2, 3)
        sigma = sigma.contiguous().view(self.N * self.n_cls, self.n_ctx, sigma.shape[3])

        if ctx_local.dim() == 3:
            ctx_local = ctx_local.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx_local = ctx_local.permute(1, 0, 2, 3)
        ctx_local =ctx_local.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx_local.shape[3])


        prefix = self.token_prefix
        suffix = self.token_suffix


        if self.class_token_position == "end":

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_sigma = torch.cat(
                [
                    prefix,  # (n_cls, -1, dim)
                    sigma,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_BA = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    BA,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            prompts_local_ctx= torch.cat([prefix,ctx_local,suffix],dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return embedding, prompts_sigma, prompts_BA, prompts, prompts_local_ctx

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        feature_dim = clip_model.visual.output_dim
        self.image_adapter = self.adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Softmax(dim=1)
        ).to(device)


    def forward(self, image):

        image_features = self.image_encoder(image.type(self.dtype))


        image_features_att = self.image_adapter(image_features)
        image_features = torch.mul(image_features_att, image_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding, prompts_sigma, prompts_BA, prompts,prompts_local_ctx = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if self.training == True:
            text_features_0 = self.text_encoder(embedding, tokenized_prompts)
            text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)
            text_features_BA = self.text_encoder(prompts_BA, tokenized_prompts)
            text_features_local_ctx = self.text_encoder(prompts_local_ctx, tokenized_prompts)

            text_features_0 = text_features_0 / text_features_0.norm(dim=-1, keepdim=True)
            text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)
            text_features_BA = text_features_BA / text_features_BA.norm(dim=-1, keepdim=True)
            text_features_local_ctx = text_features_local_ctx / text_features_local_ctx.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return text_features_0, text_features_sigma, text_features_BA, text_features, text_features_local_ctx, logits

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


# @TRAINER_REGISTRY.register()
class FLORA(TrainerX):
    """
    It is based on CoOp.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FLORA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):

        cfg = self.cfg
        self.mu = cfg.TRAINER.FLORA.mu
        self.oo = cfg.TRAINER.FLORA.oo
        self.temp = cfg.TRAINER.FLORA.temp
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FLORA.PREC == "fp32" or cfg.TRAINER.FLORA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "image_adapter" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME == "ImageNet":
            self.device = torch.device("cuda:0")
            # device0 = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)


        self.optim1 = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim2 = build_optimizer(self.model.image_adapter,cfg.OPTIM)
        self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim1, self.sched1)
        self.register_model("image_adapter", self.model.image_adapter, self.optim2, self.sched2)
        self.scaler = GradScaler() if cfg.TRAINER.FLORA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        cos = torch.nn.CosineSimilarity(dim=-1)
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.FLORA.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim1.zero_grad()
            self.optim2.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim1)
            self.scaler.step(self.optim2)
            self.scaler.update()
        else:
            text_features_0, text_features_sigma, text_features_BA, text_features, output = self.model(image)



            cls_loss = F.cross_entropy(output, label)

            posi = cos(text_features_0, text_features_sigma)
            nega = cos(text_features_0, text_features)


            logits = torch.cat((posi.reshape(-1, 1), nega.reshape(-1, 1)), dim=1)
            logits /= self.temp

            target = torch.zeros(logits.size(0)).to(self.device).long()
            contrastive_loss = F.cross_entropy(logits, target)

            l2_reg = 0.0
            l2_weight = 0.01
            for name, param in self.model.prompt_learner.named_parameters():
               if 'B' and 'A' in name:
                   l2_reg += torch.norm(param, p=2)

            ctx_local = self.model.prompt_learner.ctx_local
            l2_reg += torch.norm(ctx_local, p=2)

            ortho_reg = 0.0
            ortho_weight = 0.1
            B = self.model.prompt_learner.B

            for i in range(B.size(0)):
                B_i = B[i]

                gram = torch.matmul(B_i, B_i.transpose(0, 1))

                identity = torch.eye(gram.size(0), device=gram.device)

                ortho_reg += torch.norm(gram - identity, p='fro')

            reg_loss = l2_weight * l2_reg + ortho_weight * ortho_reg
            loss = cls_loss + self.mu * contrastive_loss +  reg_loss

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "cls_loss": cls_loss.item() if prec != "amp" else 0,
            "cont_loss": contrastive_loss.item() if prec != "amp" else 0,
            "reg_loss": reg_loss.item() if prec != "amp" else 0,  }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)



