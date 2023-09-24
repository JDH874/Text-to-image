# Text-to-image
SAW-GAN: Multi-Granularity Text Fusion Generative Adversarial Networks for Text-to-Image Generation

# Dependencies
python 3.9
pytorch 1.9
Install [CLIP](https://github.com/openai/CLIP)
In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`

**Data**
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`

**Training**
  ```
  cd SAW-GAN/code/
  ```
- Train DAE-GAN models:
  - For bird dataset: `bash scripts/train.sh ./cfg/bird.yml`
  - For coco dataset: `bash scripts/train.sh ./cfg/coco.yml`

**Pretrained Model**
- [SAW-GAN for bird]    Download and save it to `./code/saved_models/pretrained/`
- [SAW-GAN for coco]    Download and save it to `./code/saved_models/pretrained/`

**Validation**
 ```
  cd SAW-GAN/code/
  ```
set **pretrained_model** in test.sh
- For bird dataset: `bash scripts/test.sh ./cfg/bird.yml`
- For COCO dataset: `bash scripts/test.sh ./cfg/coco.yml`
