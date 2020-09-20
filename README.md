This is the code for the MSc_project_2467121d.

run.ipynb has a general structure of the recommended practice in working with the python scripts in this repository.

A brief introduction of each file is as follows:
data.py preprocesses data
read.py reads and displays insights from the preprocessed data
train.py is the main training script for classification of the processed data
ae_py has an autoencoder to work with the processed data
graph.py holds the ECG classification model from "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks"
graph_ae holds the structure of the encoder and decoder for use in ae.py
vae_v1.py and vae_v2.py correspond to VAE-1 AND VAE-2 designations in the text
ae_multi.py holds the subject-pattern disentanglement multi-ecnoder structure
ae_mulit_tri.py improves upon ae_multi.py and implements the triplet loss
config.py holds parameter argument used across the aforementioned scripts
utils.py holds unitiliy functions used across the aforementioned scripts
gpu.py checks for an existing gpu configured to work with tensorflow

/dataset is the directory for the downloaded datasets and preprocessed data
/logs is the directory for training logs
/models stores trained models ready for use with predict and classification report
/results stores output images from other scripts

requirements.txt lists the require modules for these scripts



The base code that this project is base off of is from https://github.com/physhik/ecg-mit-bih
The organisation of the scripts and data is continued.