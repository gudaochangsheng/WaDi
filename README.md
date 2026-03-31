
<div align="center">
<img src="./wadi-logo.png" alt="demo" style="width: 10%;" />
  <br>
</div>

# 🚀 [CVPR 2026]WaDi: Weight Direction-aware Distillation for One-step Image Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-WaDi-<COLOR>.svg)](https://arxiv.org/abs/2603.08258) [![arXiv](https://img.shields.io/badge/paper-WaDi-b31b1b.svg)](https://arxiv.org/pdf/2603.08258) ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=gudaochangsheng/WaDi ) [![HF-SD2.1](https://img.shields.io/badge/HF-WaDi--sd2.1-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv2-1.safetensors) [![HF-SD1.5](https://img.shields.io/badge/HF-WaDi--sd1.5-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv1-5.safetensors) [![HF-Pixart-Alpha](https://img.shields.io/badge/HF-WaDi--pixart-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_transformer.safetensors)
[![MS-SD2.1](https://img.shields.io/badge/ModelScope-WaDi--sd2.1-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv2-1.safetensors?status=2)
[![MS-SD1.5](https://img.shields.io/badge/ModelScope-WaDi--sd1.5-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv1-5.safetensors?status=2)
[![MS-Pixart-alpha](https://img.shields.io/badge/ModelScope-WaDi--pixart-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_transformer.safetensors?status=2)
[![Code](https://img.shields.io/badge/Code-WaDi-black?style=flat&logo=github)](https://github.com/gudaochangsheng/WaDi)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?style=flat-square)](https://gudaochangsheng.github.io/WaDi-Page/)

<div align="center">
  <a href="https://gudaochangsheng.github.io/">Lei Wang</a><sup>1</sup>,
  <a href="https://gudaochangsheng.github.io/WaDi-Page/">Yang Cheng</a><sup>1</sup>,
  <a href="https://sen-mao.github.io/">Senmao Li</a><sup>1</sup>,
  <a href="https://github.com/Martinser">Ge Wu</a><sup>1</sup>,
  <a href="https://yaxingwang.github.io/">Yaxing Wang</a><sup>1,3&dagger;</sup>
  <a href="https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=en">Jian Yang</a><sup>1,2&dagger;</sup>
</div>

<div align="center">
  <sup>1</sup> PCA Lab, VCIP, College of Computer Science, Nankai University &nbsp;&nbsp;
  <sup>2</sup> PCA Lab, School of Intelligence Science and Technology, Nanjing University &nbsp;&nbsp;
  <sup>3</sup> Shenzhen Futian, NKIARI &nbsp;&nbsp;
</div>

<div align="center">
  &dagger;Corresponding authors
</div>

<div align="center">
<img src="./abstract.png" alt="demo" style="zoom:150%;" />
  <br>
</div>

---

<div align="center">
<img src="./motivation.png" alt="demo" style="zoom:150%;" />
  <br>
  <em>
      Motivational analysis of our method. (a) Differences in weight norm and direction between the one-step student and the teacher
model. See Suppl. E for details and additional examples. (b) SVD analysis of the residual matrix for DMD2. (c) Replacing the one-step
model’s norm with that of the multi-step model has little effect (① ,④ ); replacing the direction severely degrades generation quality (② ,
⑤ ). (d) Qualitative examples corresponding to (c). (e) Illustration of LoRaD.
  </em>
</div>

## 📘 Introduction
Despite the impressive performance of diffusion models such as Stable Diffusion (SD) in image generation, their slow inference limits practical deployment. Recent works accelerate inference by distilling multi-step diffusion into one-step generators. To better understand the distillation mechanism, we analyze U-Net/DiT weight changes between one-step students and their multi-step teacher counterparts. Our analysis reveals that changes in weight direction significantly exceed those in weight norm, highlighting it as the key factor during distillation. Motivated by this insight, we propose the Low-rank Rotation of weight Direction (LoRaD), a parameter-efficient adapter tailored to one-step diffusion distillation. LoRaD is designed to model these structured directional changes using learnable low-rank rotation matrices. We further integrate LoRaD into Variational Score Distillation (VSD), resulting in Weight Direction-aware Distillation (WaDi)-a novel one-step distillation framework. WaDi achieves state-of-the-art FID scores on COCO 2014 and COCO 2017 while using only approximately 10% of the trainable parameters of the U-Net/DiT. Furthermore, the distilled one-step model demonstrates strong versatility and scalability, generalizing well to various downstream tasks such as controllable generation, relation inversion, and high-resolution synthesis.

<img src="./method.png" alt="method" />

<div align="center">
<em>(Left) Detailed architecture of the Low-rank Rotation of weight Direction (LoRaD) module. The LoRaD rotates the pre-trained
weight directions using learnable low-rank rotation angles. (Right) Overview of the Weight Direction-aware Distillation (WaDi) framework.
  </em>
</div>

## ✨ Qualitative results

<div align="center">
    <b>
            Quality results compared to other methods.
    </b>
</div>
<img src="./results-vis.png" alt="sd-ddim50" />

## 📈  Quantitative results
<p align="center">
<img src="./results-tab.png" alt="origin" style="width: 90%;margin-right: 20px;" /> 
</p>

## 🏋️ Training

### 🛠️ Installation
```shell
# git clone this repository
https://github.com/gudaochangsheng/WaDi.git
cd WaDi

# create new anaconda env
conda create -n wadi python=3.8 -y
conda activate wadi

# install python dependencies
pip3 install -r requirements.txt
  ```
### 🎯 train
```shell
# Train WaDi on Stable Diffusion 1.5
./train_dkd_sd1.5.sh

# Train WaDi on Stable Diffusion 2.1
./train_dkd_sd2.1.sh

# Train WaDi on PixArt-alpha
./train_dkd_pixart.sh
  ```

## 📦 Model Weights

| Model | Hugging Face | ModelScope |
|-------|--------------|------------|
| WaDi-SD2.1 | [![Download HF](https://img.shields.io/badge/HuggingFace-Download-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv2-1.safetensors) | [![Download MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv2-1.safetensors?status=2) |
| WaDi-SD1.5 | [![Download HF](https://img.shields.io/badge/HuggingFace-Download-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv1-5.safetensors) | [![Download MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv1-5.safetensors?status=2) |
| WaDi-PixArt | [![Download HF](https://img.shields.io/badge/HuggingFace-Download-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_transformer.safetensors) | [![Download MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_transformer.safetensors?status=2) |

## 🎬 Inference
coming soon

## Training SwiftBrushV2
[An unofficial implementation of SwiftBrushV2](https://github.com/gudaochangsheng/SwiftBrushV2)

## Citation

If you find WaDi useful, please consider giving our repository a star (⭐) and citing our [paper](https://arxiv.org/abs/2603.08258).

```
@article{wang2026wadi,
  title={WaDi: Weight Direction-aware Distillation for One-step Image Synthesis},
  author={Wang, Lei and Cheng, Yang and Li, Senmao and Wu, Ge and Wang, Yaxing and Yang, Jian},
  journal={arXiv preprint arXiv:2603.08258},
  year={2026}
}
@inproceedings{li2025one,
      title={One-Way Ticket: Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Models}, 
      author={Li, Senmao and Wang, Lei and Wang, Kai and Liu, Tao and Xie, Jiehang and van de Weijer, Joost and Khan, Fahad Shahbaz and Yang, Shiqi and Wang, Yaxing and Yang, Jian},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      year={2025},
}
```
## Acknowledgement

This project is based on [Diffusers](https://github.com/huggingface/diffusers). Thanks for their awesome works.
We sincerely acknowledge the excellent and inspiring prior work, [TiUE](https://github.com/sen-mao/Loopfree) and [SwiftBrush](https://github.com/VinAIResearch/SwiftBrush).
## Contact
If you have any questions, please feel free to reach out to me at  `scitop1998@gmail.com`. 
