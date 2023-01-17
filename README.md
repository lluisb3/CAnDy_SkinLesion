# Computed Aided Diagnostics Final Project
# CAnDy_SkinLesion: Skin Lesion Classification from ISIC dataset based on Deep Learning

## MAIA Master 2023


---------------------------------------
## Team Members

- ### Borras Ferris Lluis

- ### Leon Contreras Nohemi Sofia

This repository contains all the code for our final project on skin lesion classification
on dermatoscopic images that we developed for our CAD course. We finetuned different architectures from pretrained
Torchvision models. The dataset used can be found on


##Instructions
To reproduce the results or use the trained networks available in the `models` directory, you should do:

- Environmental set up
- Download the checkpoints for the models
- Download the example image (or provide one of your own)
- Run the examples

### Setting up the environment
- Create a conda environment
```
conda create -n cadskin python==3.9.13 anaconda -y && conda activate cadskin
```
- Install the requirements
```
pip install -r requirements.txt
```
### Download checkpoints of pretrained models

The checkpoints for the binary classification are found in `models/BinaryClassification/binary_efficient_unfreeze_dropout`.
directory with the images you would like to test.
The checkpoints for the multiclass classification are found in `models/MulticlassClassification/multiclass_efficient_nofreeze_multisteplr01_dropout`.

To run the test.py you will need to have a `data/MulticlassClassification/test` directory with the images you would like to test.

### Download trained U-net

The trained UNet in the ISIC 2017 to perform the segmentation of the data by using the function segmentation on
`segmentation.py` can be downloaded in the following link:
`https://drive.google.com/file/d/1Ae0M2pNVBgbKa7b13_kmY3uYmaTv5ayK/view?usp=share_link`

Once downloaded the file named `Unet_trained.tar` must be located in the project sub-folder `models`.

## Run the segmentation

To compute the segmentations from the trained UNet run the following lin eof code. 

````commandline
python -m segmentation.segmentation 
````

Then will prompt to specify for which challenge you want to perform the segmentation either: `BinaryClassification` or 
`MulticlassClassification`. Then in which set either `train`, `val` or `test`.

````commandline
python -m classification_binary.train
````

````commandline
python -m classification_multi.train
````
