[![arXiv](https://img.shields.io/badge/arXiv-2208.07227-b31b1b.svg)](https://arxiv.org/abs/2208.07227)
![visitors](https://visitor-badge.glitch.me/badge?page_id=vLAR-group/DM-NeRF)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

# DM-NeRF: 3D Scene Geometry Decomposition and Manipulation from 2D Images

This is the official repository of the **DM-NeRF**. For technical details, please refer to:

**DM-NeRF: 3D Scene Geometry Decomposition and Manipulation from 2D Images** <br />
[Bing Wang](https://www.cs.ox.ac.uk/people/bing.wang/), [Lu Chen](https://chenlu-china.github.io/), [Bo Yang<sup>*</sup>](https://yang7879.github.io/) <br />
[**Paper**](https://arxiv.org/abs/2208.07227) | [**Video**](https://www.youtube.com/watch?v=yQtpPfM5dTA) | [**DM-SR**](https://github.com/vLAR-group/DM-NeRF)


<p align="center"> <img src="/figs/architecture.png" width="100%"> </p>

The architecture of our proposed DM-NeRF. Given a 3D point $\boldsymbol{p}$, we learn an object code through a series of loss functions using both 2D and 3D supervision signals.

## 1. Decomposition and Manipulation:

<div align=center>
     <img float=left src="/figs/decomposition.gif" width=42.5%>&nbsp<img float=right src="/figs/manipulation.gif" width=56.5%> <br/></br>
</div>

## 2. Qualitative Results

### Scene Decomposition
<div align=center>
     <img src="/figs/decomposition_scenes.png" width=85% > <br/></br>
</div>

### Object Manipulation

#### Rigid Transformation

<div align=center>
     <img src="/figs/edit_replica.png" width=80% > <br/>
     <img src="/figs/edit_dmsr.png" width=80% > <br/>
</div>

#### Deformable Manipulation

<div align=center>
     <img src="/figs/edit_deformable.png" width=80% > <br/>
</div>

## 3. Installation

DM-NeRF uses a Conda environment that makes it easy to install all dependencies.

1. Create the `DM-NeRF` Conda environment (Python 3.7) with [miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Install all dependencies by running:

```bash
pip install pytorch==1.8.1 torchVision==0.9.1 torchaudio===0.8.1
pip install -r environment.txt
```

3. Activate the environment: `conda activate DM-NeRF`.

### Datasets

In this paper, we consider the following three different datasets:

#### [DM-SR](https://github.com/vLAR-group/DM-NeRF)

To the best of our knowledge, there is no existing 3D scene dataset suitable for quantitative evaluation of geometry manipulation. Therefore, we create a synthetic dataset with 8 types of different and complex indoor rooms, called DM-SR. The room types and designs follow [Hypersim Dataset](https://github.com/apple/ml-hypersim). Overall, we firstly render the static scenes, and then manipulate each scene followed by second round rendering. Each scene has a physical size of about 12x12x3 meters with around 8 objects.  We will keep updating it for future research in the community. 

#### [Replica](https://github.com/facebookresearch/Replica-Dataset)

In this paper, we use 7 scenes `office0, office2, office3, office4, room0, room1, room2` from the [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset). We request the authors of [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf) to generate color images and 2D object masks with camera poses at 640x480 pixels for each of 7 scenes. Each scene has 59~93 objects with very diverse sizes. Details of camera settings and trajectories can be found in the Semantic-NeRF repository.

#### [ScanNet](http://www.scan-net.org/)

In this paper, we use 8 scenes `scene0010_00, scene0012_00, scene0024_00, scene0033_00, scene0038_00, scene0088_00, scene0113_00, scene0192_00` from the ScanNet Dataset.

### Training

For the training of our standard DM-NeRF , you can simply run the following command with a chosen config file specifying data directory and hyper-params.

```bash

CUDA_VISIBLE_DEVICES=0 python -u train_dmsr.py --config configs/train/dmsr/study.txt

```
Other working modes and set-ups can be also made via the above command by choosing different config files.


### Evaluation

In this paper, we use PSNR, SSIM, LPIPS for rendering evaluation, and mAPs for both decomposition and manipulation evluations.

#### Decomposition

For decomposition evaluation, you need set `render=True` and `log_time="your log folder name"` in a config file. And then run:

```bash 

CUDA_VISIBLE_DEVICES=0 python -u test_dmsr.py --config configs/test/dmsr/study.txt

```

#### Manipulation

##### Quantitative Evaluation

We provide the DM-SR dataset for the quantitative evaluation of geometry manipulation.

Set the target object and desired manipulated settings in a sepcific config file. And then run:

```bash

CUDA_VISIBLE_DEVICES=0 python -u test_dmsr.py --config configs/manipulation/dmsr/manipulation_translation/study.txt

```
##### Qualitative Evaluation

For other qualitative evaluations, you can simply change the config file and then run:

```bash

CUDA_VISIBLE_DEVICES=0 python -u test_dmsr.py --config configs/test/dmsr/study.txt

```

## 4. Video (Youtube)
<p align="center"> <a href="https://www.youtube.com/watch?v=iE0RwmdLIzk"><img src="/figs/mesh.gif" width="80%"></a> </p>

### Citation
If you find our work useful in your research, please consider citing:

      @article{wang2022dmnerf,
      title={DM-NeRF: 3D Scene Geometry Decomposition and Manipulation from 2D Images},
      author={Bing, Wang and Chen, Lu and Yang, Bo},
      journal={arXiv preprint arXiv:2208.07227},
      year={2022}
    }

### License
Licensed under the CC BY-NC-SA 4.0 license, see [LICENSE](./LICENSE).

### Updates
* 15/8/2022: Initial release！

## Related Repos
1. [RangeUDF: Semantic Surface Reconstruction from 3D Point Clouds](https://github.com/vLAR-group/RangeUDF) ![GitHub stars](https://img.shields.io/github/stars/vLAR-group/RangeUDF.svg?style=flat&label=Star)
2. [GRF: Learning a General Radiance Field for 3D Representation and Rendering](https://github.com/alextrevithick/GRF) ![GitHub stars](https://img.shields.io/github/stars/alextrevithick/GRF.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)

