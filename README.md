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


##Instructions for testing
To use the trained networks available in the `models` directory, you should do:

- Environmental set up
- Download the checkpoints for the models

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
The checkpoints for the multiclass classification are found in `models/MulticlassClassification/multiclass_efficient_nofreeze_multisteplr01_dropout`.

To run the `test.py` you will need to have a `data/MulticlassClassification/test` or a `data/BinaryClassification/test`  directory with the images you would like to test.
Run the `database/metadata.py` to have the metadata.csv for the SkinDataset.

##Instructions for training 
To finetune the models or reproduce our results, you should do:

- Environmental set up (same as described above)
- Clone this repository 
- If training the binary challenge, create a `data\BinaryClassification\train` and `data\BinaryClassification\val` directory with the images you are going to use for training and validation. 
If training for the multiclass challenge create a `data\MulticlassClassification\train` and `data\MulticlassClassification\val` directory with the images you are going to use for training and validation. 
- Run the `database/metadata.py` to have the metadata.csv. Use the challenge option of your like
````commandline
python -m database.metadata --challenge_option BinaryClassification
````

- If using segmentation follow the instructions of Download trained U-Net section.
- Modify the config.yml.example with your desired settings for training and save it as config.yml
- Run `classification_binary/train.py` or `classification_multi/train.py` accordingly.

````commandline
python -m classification_binary.train
````

````commandline
python -m classification_multi.train
````

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

