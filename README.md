# MICCAI 2023: FSDiffReg: Feature-wise and Score-wise Diffusion-guided Unsupervised Deformable Image Registration for Cardiac Images

This repository is the official implementation of "FSDiffReg: Feature-wise and Score-wise Diffusion-guided Unsupervised Deformable Image Registration for Cardiac Images".
<img src="./img/mainfigure.png">

## Requirements
Please use command
```
pip install -r requirements.txt
```
to install the environment. We used PyTorch 1.12.0, Python 3.8.10 for training.

## Data
* We used 3D cardiac MR images for training: [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

## Training

To run training process:

```
python train.py -c config/train_3D.json
```

## Test

To run testing process:

```
python3 test.py -c config/test_3D.json -w [YOUR TRAINED WEIGHTS]
```
Trained model can be found [here](https://drive.google.com/drive/folders/1x4NC9hHor2JexrclDmUMfKYTHhOQvVYT?usp=sharing)

## Acknowledgement

We would like to thank the great work of the following open-source project: [DiffuseMorph](https://github.com/DiffuseMorph/DiffuseMorph).

## Citation

