# FLORA
Here is the official implementation of the model FLORA.

## Abstract
Federated Prompt Learning (FPL) leverages the robust representation learning and remarkable generalization ability of large pre-trained Vision-Language Models (VLMS) into federated learning through prompt learning. Existing FPL methods attempt to address data heterogeneity via model personalization. However, excessive personalization leads to the compromise of model’s generalization ability, which remains a significant challenge to achieve a balance between personalization and generalization under high data heterogeneity. To address this challenge, we propose FLORA, a novel framework that combines orthogonal low-rank adaptation withattention-guided client adapter. Specifically, each client personalizes global prompt through orthogonal low-rank adaptation term, thereby achieving efficient local adaptation while maintaining the generalization of the global prompt. In addition, we introduce a lightweight attention-based adapter for the image encoder, which can enhance cross-modal alignment under nonindependent and nonidentically distributed (Non-IID) environment to further achieve the balance. Extensive experiments on multiple datasets demonstrate that our FLORA achieves superiority performance in balancing generalization and personalization over state-of-the-art methods under high data heterogeneity.

## Method
![F1](https://github.com/chengnan1430/DFM-trans/blob/main/image/F2.png)

* First, DFM-Trans model integrates model knowledge from diverse source domains through an adaptive Feature Fusion Network (FFN), combining local and global features to enhance the model's capacity for feature representation.

* Second, DFM-Trans model introduces a dynamic weight adjustment mechanism based on predictive uncertainty, allowing adaptive adjustment of source model weights to optimize performance in the target domain.

* Finally, a comprehensive confidence-driven pseudo-labeling strategy is proposed, prioritizing knowledge extraction from high-confidence samples and transferring it to lower-confidence samples, effectively reducing the generation of erroneous pseudo labels.

## Setup
### Install Package Dependencies

```
* Python Environment: >= 3.6
* torch >= 1.1.0
* torchvision >= 0.3.0
* scipy == 1.3.1
* sklearn == 0.5.0
* numpy == 1.17.4
* argparse, PIL
```

## Datasets:
* **Office Dataset:** Download the datasets [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [Office-Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar) .
* **DomainNet Dataset:** Download [DomainNet](http://ai.bu.edu/DomainNet/) .
* Place these datasets in './data'.
* Using readfile.py to generate '.txt' file for each dataset (change dataset argument in the file accordingly).
  
```
data
│       
└───Office-home
│   │  Art
│   │  Clipart
│   │  Product
|   |  Real_world
|   |  Art_list.txt 
|   |  Clipart_list.txt
|   |  Product_list.txt 
|   |  Real_world_list.txt
└───DomainNet
│   │    Clipart
│   │    Infograph
│   │   ...
└───Office-Caltech
│   │   ...
└───Office-31
│   │   ...
...
```

## Training:

* Train source models (shown here for Office-31 with source A)

```shell
python train_source.py --dset office-31 --s 1 --t 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```

* Adapt to target domain (shown here for Office-31 with target D)
```shell
python train_target.py --dset office-31 --t 1 --max_epoch 15 --gpu_id 0 --cls_par 0.7 --crc_par 0.01 --crc_mse 0.01 --output_src ckps/source/ --output ckps/DFM
```
