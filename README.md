# CVRKD-IQA(DistillationIQA)
This repository is for CVRKD-IQA introduced in the following paper

Guanghao Yin, Wei Wang, Zehuan Yuan, Chuchu Han, Wei Ji, Shouqian Sun and Changhu Wang, "Content-Variant Reference Image Quality Assessment via Knowledge Distillation", AAAI Oral, 2022 [arxiv](https://arxiv.org/abs/2202.13123)

## Introduction
CVRKD-IQA is the first content-variant reference IQA method via knowledge distillation. The practicability of previous FR-IQAs is affected by the requirement for pixel-level aligned reference images. And NR-IQAs still have the potential to achieve better performance since HQ image information is not fully exploited. Hence, we use non-aligned reference (NAR) images to introduce various prior distributions of high-quality images. Moreover, the comparisons of distribution differences between HQ and LQ images can help our model better assess the image quality. Further, the knowledge distillation transfers more HQ-LQ distribution difference information from the FR-teacher to the NAR-student and stabilizing CVRKD-IQA performance. Since the content-variant and non-aligned reference HQ images are easy to obtain, our model can support more IQA applications with its relative robustness to content variations.

<div align=center><img src="https://github.com/guanghaoyin/CVRKD-IQA/blob/main/imgs/distillationIQA.png" alt="Distillation" align="middle" /></div>

## Prepare data
### Training datasets
Download synthetic [Kaddid-10K](http://database.mmsp-kn.de/kadid-10k-database.html) dataset. And download the training HQ images of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as the reference dataset.

### Testing datasets
Download synthetic [LIVE](http://live.ece.utexas.edu/index.php), [CSIQ](https://qualinet.github.io/databases/image/categorical_image_quality_csiq_database/) [TID2013](http://www.ponomarenko.info/tid2013.htm) and authentic [KonIQ-10K](http://database.mmsp-kn.de/koniq-10k-database.html) datasets. And download the testing HQ images of [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as the reference dataset.

Place those unzipped data in ./dataset file
## Train
### 1.Train the FR-teacher
(1) (optional) Download models for our paper and place it in './model_zoo/'
   The models for FR-teacher can be downloaded from [Google Cloud](https://drive.google.com/file/d/1niFBV-ysJeVoaXUPQp-08ovrjS9tPhGW/view?usp=sharing)
   
(2) Quick start (you can change the options in option_train_DistillationIQA_FR.py)
```
Python train_DistillationIQA_FR.py --self_patch_num 10 --patch_size 224
```
### 2.Fix pretrained FR-teacher and train the NAR-student
(1) (optional) Download models for our paper and place it in './model_zoo/'
   The models for FR-teacher and NAR-student can be downloaded from [Google Cloud](https://drive.google.com/file/d/107TI1pa0TDxs3V8tO2KhmhKJmfc9ZOl4/view?usp=sharing)
   
(2) Quick start (you can change the options in option_train_DistillationIQA.py)
```
Python train_DistillationIQA.py --self_patch_num 10 --patch_size 224
```

## Test
(1) Make sure the trained models are placed in './model_zoo/FR_teacher_cross_dataset.pth' and './model_zoo/NAR_student_cross_dataset.pth' 

(2) Quick start (you can change the options in option_train_DistillationIQA.py)
```
Python test_DistillationIQA.py
```
(3) test single image
```
Python test_DistillationIQA_single.py
```
## More visual results
Synthetic examples of IQA scores predicted by our NAR-student when gradually increasing the distortion levels.
<div align=center><img src="https://github.com/guanghaoyin/CVRKD-IQA/blob/main/imgs/synthetic_example.png" alt="Distillation" align="middle" /></div>

Real-data examples of IQA scores predicted by our NAR-student.
<div align=center><img src="https://github.com/guanghaoyin/CVRKD-IQA/blob/main/imgs/real_example.png" alt="Distillation" align="middle" /></div>

## T-SNE visual visualization of HQ-LQ difference-aware features of NAR-student w/ and w/o KD
After distilled with FR-teacher, the HQ-LQ features in Fig(b) are clusterd. It proves that the HQ-LQ distribution difference prior from the FR-teacher can indeed
help the NAR-student extract quality-sensitive discriminative features for more accurate and consistent performance.

<div align=center><img src="https://github.com/guanghaoyin/CVRKD-IQA/blob/main/imgs/KD.png" alt="Distillation" align="middle" /></div>

## Citation

If you find the code helpful in your resarch or work, please cite the following papers.

```
@article{yin2022content,
  title={Content-Variant Reference Image Quality Assessment via Knowledge Distillation},
  author={Yin, Guanghao and Wang, Wei and Yuan, Zehuan and Han, Chuchu and Ji, Wei and Sun, Shouqian and Wang, Changhu},
  journal={arXiv preprint arXiv:2202.13123},
  year={2022}
}
```
## Acknowledgements
Part of our code is built on [HyperIQA](https://github.com/SSL92/hyperIQA). We thank the authors for sharing their codes. Also thanks for the support of [Bytedance.Inc](https://github.com/bytedance)
