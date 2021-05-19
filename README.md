# vision-transformer-pytorch
A collection of SOTA ViT on Pytorch

Build on [__Swin-Transformer__](https://github.com/microsoft/Swin-Transformer) for Single GPU Training or DataParallel Training, remove DDP Training 

## Contents
- [__basic config file__](https://github.com/rentainhe/vision-transformer/blob/master/config.py)
- [__model zoo__]()

## Usage
### Install
- Clone this repo:
```bash
git clone https://github.com/rentainhe/vision-transformer.git
cd vision-transformer
```

- Create a conda virtual environment and activate it
```bash
conda create -n vit python=3.7 -y
conda activate vit
```

- Install `Pytorch=1.8.0` and `torchvision=0.9.0` with `CUDA=11.0`
```bash
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

- Install `timm`
```bash
pip install timm
```

- Install `Apex`:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install `tensorboard`:
```bash
pip install tensorboard
```

- Install other requirements:
```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

### Training
- Training `vit-b_16` model for `cifar100`
```bash
python main.py --cfg configs/ViT/vit-b_16.yaml
```
Addition args:
- `--batch-size`: batch size for Single GPU, e.g, `--batch-size 128`
- `--dataset`: dataset for training, e.g, `--dataset cifar100`
- `--accumulation-steps`: gradient accumulation steps, e.g, `--accumulation-steps 2`

- tensorboard
```bash
tensorboard --logdir tensorboard_output/ --port 6006 --host localhost
```


## reference
- [__Swin-Transformer__](https://github.com/microsoft/Swin-Transformer)
