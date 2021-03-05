# ProgressiveGAN

## Result
Image size is 512 x 512
<p align="center"><img width="100%" src="png/generated_image.png" /></p>

## Prerequisites
* [Python 3.6+](https://www.continuum.io/downloads)
* [PyTorch 1.7.0](http://pytorch.org/)

## How to use
### Dataset
```
├── dataset
   └── FFHQ
       ├── 000001.jpg 
       ├── 000002.png
       └── ...
```

### Train
```python main.py --exp=PGGAN_512 --data_root=$PATH```

The model parameters and images will be stored in the PGGAN_512 folder.


### reference
- https://github.com/rosinality/progressive-gan-pytorch
