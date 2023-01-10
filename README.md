# Lama-with-MaskDINO
It was inspired by [Auto-LaMa](https://github.com/andy971022/auto-lama#readme).

Unlike Auto-Lama, it differs in:
1. Use the object instance segmentation model [MaskDINO](https://github.com/IDEA-Research/MaskDINO) instead of the object detection model [DETR](https://github.com/facebookresearch/detr).
1. Use [LaMa with refiner](https://github.com/geomagical/lama-with-refiner) for better results.

## Environment setup
1. Download pre-trained weights [MaskDINO](https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth) and [LaMa](https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) 
1. Put the directory like this
```
  .root
  ├─demo.py
  ├─ckpt
  │  ├──maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth
  │  └─models
  │      ├──config.yaml
  │      └─models
  │          └─best.ckpt
  └─images
       ├──buildings.png
       ├──cat.png
       └──park.png     
```
3. conda environment setup
```
conda create --name maskdino python=3.8 -y
conda activate maskdino
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

mkdir repo
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git

cd ..
git clone https://github.com/MeAmarP/MaskDINO.git
cd MaskDINO
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
python setup.py build install
cd ../../../../..

git clone https://github.com/geomagical/lama-with-refiner.git
cd lama-with-refiner
pip install -r requirements.txt 
pip install --upgrade numpy==1.23.0
cd ../..
pip install gradio
```
4. Run
```
python demo.py
```
## Acknowledgments
Many thanks to these excellent opensource projects
* [LaMA](https://github.com/saic-mdal/lama)
* [LaMa with refiner](https://github.com/geomagical/lama-with-refiner)
* [MaskDINO](https://github.com/IDEA-Research/MaskDINO)
* [MaskDINO inference code](https://github.com/MeAmarP/MaskDINO/tree/quickfix/infer_demo)
* [Detectron2](https://github.com/facebookresearch/detectron2)
