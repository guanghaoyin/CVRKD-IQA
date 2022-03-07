# CVRKD-IQA(DistillationIQA)
This repository is for CVRKD-IQA introduced in the following paper
uanghao Yin, Wei Wang, Zehuan Yuan, Chuchu Han, Wei Ji, Shouqian Sun and Changhu Wang "Content-Variant Reference Image Quality Assessment via Knowledge Distillation", AAAI, 2022 [arxiv](https://arxiv.org/abs/2202.13123)

## Introduction
CVRKD-IQA is the first content-variant reference IQA method via knowledge distillation. The practicability of previous FR-IQAs is affected by the requirement for pixel-level aligned reference images. And NR-IQAs still have the potential to achieve better performance since HQ image information is not fully exploited. Hence, we use non-aligned reference (NAR) images to introduce various prior distributions of high-quality images. Moreover, The comparisons of distribution differences between HQ and LQ images can help our model better assess the image qual- ity. Further, the knowledge distillation transfers more HQ-LQ distribution difference information from the FR-teacher to the NAR-student and stabilizing CVRKD-IQA performance. Since the content-variant and non-aligned reference HQ images are easy to obtain, our model can support more IQA applications with its relative robustness to content variations.

<div align=center><img src="https://github.com/guanghaoyin/CVRKD-IQA/blob/main/imgs/distillationIQA.png" alt="Distillation" align="middle" /></div>

