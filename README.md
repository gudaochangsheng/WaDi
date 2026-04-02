

<div align="center">
  <img src="./wadi-logo.png" alt="WaDi logo" width="120" />
</div>

<h1 align="center">🚀 WaDi: Weight Direction-aware Distillation for One-step Image Synthesis</h1>

<p align="center">
  <b>CVPR 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.08258"><img src="https://img.shields.io/badge/arXiv-2603.08258-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2603.08258"><img src="https://img.shields.io/badge/Paper-PDF-red.svg" alt="Paper PDF"></a>
  <a href="https://gudaochangsheng.github.io/WaDi-Page/"><img src="https://img.shields.io/badge/Project-Page-2ea44f?style=flat-square" alt="Project Page"></a>
  <a href="https://github.com/gudaochangsheng/WaDi"><img src="https://img.shields.io/badge/Code-GitHub-black?style=flat&logo=github" alt="Code"></a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=gudaochangsheng/WaDi" alt="Visitors">
</p>

<p align="center">
  <a href="https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv2-1.safetensors"><img src="https://img.shields.io/badge/HF-WaDi--SD2.1-yellow?logo=huggingface" alt="HF SD2.1"></a>
  <a href="https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv1-5.safetensors"><img src="https://img.shields.io/badge/HF-WaDi--SD1.5-yellow?logo=huggingface" alt="HF SD1.5"></a>
  <a href="https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_transformer.safetensors"><img src="https://img.shields.io/badge/HF-WaDi--PixArt-yellow?logo=huggingface" alt="HF PixArt"></a>
</p>

<p align="center">
  <a href="https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv2-1.safetensors?status=2"><img src="https://img.shields.io/badge/ModelScope-WaDi--SD2.1-blue" alt="MS SD2.1"></a>
  <a href="https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv1-5.safetensors?status=2"><img src="https://img.shields.io/badge/ModelScope-WaDi--SD1.5-blue" alt="MS SD1.5"></a>
  <a href="https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_transformer.safetensors?status=2"><img src="https://img.shields.io/badge/ModelScope-WaDi--PixArt-blue" alt="MS PixArt"></a>
</p>

<div align="center">
  <a href="https://gudaochangsheng.github.io/">Lei Wang</a><sup>1</sup>,
  <a href="https://gudaochangsheng.github.io/WaDi-Page/">Yang Cheng</a><sup>1</sup>,
  <a href="https://sen-mao.github.io/">Senmao Li</a><sup>1</sup>,
  <a href="https://github.com/Martinser">Ge Wu</a><sup>1</sup>,
  <a href="https://yaxingwang.github.io/">Yaxing Wang</a><sup>1,3†</sup>,
  <a href="https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=en">Jian Yang</a><sup>1,2†</sup>
</div>

<div align="center">
  <sup>1</sup> PCA Lab, VCIP, College of Computer Science, Nankai University &nbsp;&nbsp;
  <sup>2</sup> PCA Lab, School of Intelligence Science and Technology, Nanjing University &nbsp;&nbsp;
  <sup>3</sup> Shenzhen Futian, NKIARI
</div>

<div align="center">
  † Corresponding authors
</div>

---

## 🔥 Highlights

- **A simple but effective insight:** in one-step diffusion distillation, **weight direction matters much more than weight norm**.
- We propose **LoRaD** (**Lo**w-rank **Ra**tation of weight **D**irection), a parameter-efficient adapter for modeling structured directional changes.
- We build **WaDi**, a weight direction-aware one-step distillation framework based on **VSD**.
- WaDi achieves **state-of-the-art FID** on **COCO 2014** and **COCO 2017**.
- WaDi uses only **~10% trainable parameters** of the original **U-Net / DiT**.
- The distilled one-step model generalizes well to **controllable generation**, **relation inversion**, and **high-resolution synthesis**.

---

## 🖼️ Overview

<div align="center">
  <img src="./abstract.png" alt="WaDi abstract" style="zoom:150%;" />
</div>

---

## 💡 Motivation

<div align="center">
  <img src="./motivation.png" alt="WaDi motivation" style="zoom:150%;" />
  <br>
  <em>
    Motivational analysis of WaDi.
    (a) Differences in weight norm and direction between the one-step student and the teacher model.
    (b) SVD analysis of the residual matrix for DMD2.
    (c) Replacing the one-step model's norm with that of the multi-step model has little effect, while replacing the direction severely degrades generation quality.
    (d) Qualitative examples corresponding to (c).
    (e) Illustration of LoRaD.
  </em>
</div>

---

## 📘 Introduction

Diffusion models such as Stable Diffusion achieve impressive image generation quality, but their multi-step inference is still expensive for practical deployment. Recent works aim to accelerate inference by distilling multi-step diffusion models into one-step generators.

To better understand the distillation mechanism, we analyze the weight changes between one-step students and their multi-step teacher counterparts in both **U-Net** and **DiT** models. Our analysis shows that **directional changes in weights are significantly larger and more important than norm changes** during one-step distillation.

Motivated by this finding, we propose **LoRaD** (**Lo**w-rank **Ra**tation of weight **D**irection), a lightweight adapter that models structured directional changes using learnable low-rank rotation matrices. We further integrate LoRaD into **Variational Score Distillation (VSD)** and build **WaDi**, a novel one-step distillation framework.

WaDi achieves **state-of-the-art FID** on **COCO 2014** and **COCO 2017** while using only **~10%** of the trainable parameters of the original **U-Net / DiT**. In addition, the distilled one-step model remains versatile and scalable, supporting downstream applications such as **controllable generation**, **relation inversion**, and **high-resolution synthesis**.

---

## 🧠 Method

<div align="center">
  <img src="./method.png" alt="WaDi method" />
  <br>
  <em>
    Left: architecture of the <b>LoRaD</b> module, which rotates pretrained weight directions using learnable low-rank rotation angles.
    Right: overview of the <b>WaDi</b> framework.
  </em>
</div>

---

## ✨ Qualitative Results

<div align="center">
  <b>Qualitative comparison with existing one-step distillation methods.</b>
</div>

<div align="center">
  <img src="./results-vis.png" alt="WaDi qualitative results" />
</div>

---

## 📈 Quantitative Results

<p align="center">
  <img src="./results-tab.png" alt="WaDi quantitative results" style="width: 90%; margin-right: 20px;" />
</p>

---

## 📦 Model Zoo

| Model | Architecture | Hugging Face | ModelScope |
|-------|--------------|--------------|------------|
| WaDi-SD2.1 | U-Net | [![Download HF](https://img.shields.io/badge/HuggingFace-Download-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv2-1.safetensors) | [![Download MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv2-1.safetensors?status=2) |
| WaDi-SD1.5 | U-Net | [![Download HF](https://img.shields.io/badge/HuggingFace-Download-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_unet-sdv1-5.safetensors) | [![Download MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_unet-sdv1-5.safetensors?status=2) |
| WaDi-PixArt | DiT | [![Download HF](https://img.shields.io/badge/HuggingFace-Download-yellow?logo=huggingface)](https://huggingface.co/gudaochangsheng/WaDi/blob/main/rotated_transformer.safetensors) | [![Download MS](https://img.shields.io/badge/ModelScope-Download-blue)](https://modelscope.cn/models/gudaochangsheng98/WaDi/file/view/master/rotated_transformer.safetensors?status=2) |

---

## 🛠️ Installation

```bash
git clone https://github.com/gudaochangsheng/WaDi.git
cd WaDi

conda create -n wadi python=3.8 -y
conda activate wadi

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
# Train WaDi on Stable Diffusion 1.5
bash train_dkd_sd1.5.sh

# Train WaDi on Stable Diffusion 2.1
bash train_dkd_sd2.1.sh

# Train WaDi on PixArt-alpha
bash train_dkd_pixart.sh
```

---

## 🎬 Inference

```bash
# Inference for Stable Diffusion models
python infer_sd_model.py

# Inference for PixArt-alpha
python infer_pixart.py
```

---

## 🔗 Related Projects

- [SwiftBrushV2 (unofficial implementation)](https://github.com/gudaochangsheng/SwiftBrushV2)

---

## 📚 Citation

If you find **WaDi** useful, please consider giving this repository a **star** ⭐ and citing our paper.

```bibtex
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
  year={2025}
}
```

---

## 🙏 Acknowledgements

This project is built upon [Diffusers](https://github.com/huggingface/diffusers).  
We also sincerely acknowledge the inspiring prior work:
[TiUE](https://github.com/sen-mao/Loopfree) and [SwiftBrush](https://github.com/VinAIResearch/SwiftBrush).

---

## 📮 Contact

If you have any questions, please feel free to contact:

`scitop1998@gmail.com`
