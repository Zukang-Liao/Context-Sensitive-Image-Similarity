<h1 align="center">Image Similarity using An Ensemble of Context-Sensitive Models [Accepted by SIGKDD2024]</h1>


<p align="center">
<a href="https://arxiv.org/abs/2401.07951"><img  src="https://img.shields.io/badge/Arxiv-Paper-blue" ></a>
<a href="https://drive.google.com/drive/folders/1N5IE7FEevMiVTxeS92d_fanmDH9_d2kY"><img  src="https://img.shields.io/badge/google_drive-CoSIS-green" ></a>
</p>


<h5 align="center"><em>Zukang Liao<sup>1</sup> and Min Chen<sup>1</em></h5>
<h6 align="center"><sup>1</sup> University of Oxford, Department of Engineering Science</h6>

## Context-Sensitive-Image-Similarity (CoSIS Dataset)
For a triple (Reference, Candidate A, Candidate B), we provide Two-Alternative Forced-Choice(2AFC) annotations:

<img src="https://drive.google.com/uc?export=view&id=1zTETyLsvdBO_Smwj1XQxOxrTcG4fbQsF" style="width: 500px; max-width: 100%; height: auto" title="Click to enlarge picture" />

Note that: Candidate A and B are not necessarily similar to the Reference.

Our Annotations are based on the [BG20K](https://github.com/JizhiziLi/GFM?tab=readme-ov-file#bg-20k) Dataset.

Our labelled triplets (CoSIS) can be found in our [Metadata](https://github.com/Zukang-Liao/Context-Sensitive-Image-Similarity/blob/main/metadata)

## Fine-tune Your Context-Sensitive Models
Specify reference id(s) and modelarch in dist_trainmodel.py.
```python
refids = (2723,)
modelarch = 'resnet18' # Available Archs: vgg11/13/16(bn), resnet18/34/50, resnet18-place365, cvnet, lora-vit, vit
```

Availabel Reference IDs:

<img src="https://drive.google.com/uc?export=view&id=1-PW_7PxMF2G-7HcU4VyBqWvQuUBHeaaj" style="width: 800px; max-width: 100%; height: auto" title="Click to enlarge picture" />


Example of local improvement (whilst similar global performance from the beginning to the end of fine-tuning):

<img src="https://drive.google.com/uc?export=view&id=1O3nRDwvMI7I2EaPC-23JPx2UKzkaRf1b" style="width: 700px; max-width: 100%; height: auto" title="Click to enlarge picture" />

## Build an Ensemble of Your CS-Models
1. Obtain embeddings of images in the [BG20K](https://github.com/JizhiziLi/GFM?tab=readme-ov-file#bg-20k) Dataset. Note that: in CoSIS(Ours) dataset, the annotations for CS model fine-tuning and Ensemble Strategies are based on images in the [BG20K](https://github.com/JizhiziLi/GFM?tab=readme-ov-file#bg-20k) Dataset Validation Set. And our annotations for testing the performance of the ensemble model are based on images in the [BG20K](https://github.com/JizhiziLi/GFM?tab=readme-ov-file#bg-20k) Dataset Training Set.


```bash
python dist_savefeats.py --cs_refid=2723 --cs_arch=resnet18
```

2. Put the embeddings from the selected CS models in the "./saved_feats/val/selected" and "./saved_feats/train/selected" folder.(Optional) Download and save the [ViT4PCA](https://drive.google.com/drive/folders/1pEgXO3GnivLrfQOfqr-jNQHad_AziQxL) embeddings (for PCA) to "./saved_feats" folder accordingly. Choen CS-models in the paper are:

| | #Indoor | #City | #Ocean | #Field | #Mountain | #Forest | #Flower | #Abstract |
| :----:| :----: | :----: | :----: |  :----: |  :----: |  :----: |  :----: | :---:|
|Model Arch| ResNet18 | ResNet18-Place365 | ViT-Lora | VGG16 | ViT-Lora | ViT-Lora | ViT-Lora | ResNet-Place365 |
|Acc on its CS cluster| 88.0% | 87.5% | 89.5% | 90.1% | 90.7% | 86.2% | 86.8% | 83.8% |

3. Build your own Ensemble model:

```bash
python dist_cluster.py
```

| Model Description | Global Performance |
|-|-|
| Best existing model (not fine-tuned) | 79.9% |
| Best global model (directly and globally fine-tuned) | 72.1% |
| Best CS model (locally fine-tuned) | 79.1% |
| Majority Voting of CS models | 81.9% |
| Ensemble PCA - Ours | 83.3% |
| Ensemble MLP - Ours | **84.7%** |


## Citation
Please consider citing the paper if your work is related to our CoSIS Dataset or Image Similarity:
```
@article{liao2024image,
  title={Image Similarity using An Ensemble of Context-Sensitive Models},
  author={Liao, Zukang and Chen, Min},
  journal={arXiv preprint arXiv:2401.07951},
  year={2024}
}
```
