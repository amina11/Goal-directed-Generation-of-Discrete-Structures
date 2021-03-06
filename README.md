# Goal-directed-Generation-of-Discrete-Structures-with-Conditional-Generative-Models

This repository is the official implementation of [Goal-directed-Generation-of-Discrete-Structures-with-Conditional-Generative-Models](https://arxiv.org/pdf/2010.02311.pdf).


## Requirements

To code requires: 
```
Python 3.6.9
torch = 1.2.0a0+e6a7071
numpy=1.16.4
```

Easy way is to pull the image given in the link below,  it sets up all the environment you need.
https://hub.docker.com/repository/docker/2012913/icml_2020


## Dataset
All the raw data and processed data that are used in the experiments can be downloaded from this dropbox link:

https://www.dropbox.com/sh/cnt3j5hlf3z29f4/AABQpmFNkQPiygfx9OE2HVxAa?dl

You can either generate the data following the instruction or directly download the folder named `data` in the dropbox and place it in the current directory. Once the data download/generated, you should have the following structure:

```sh
Goal-directed-Generation-of-Discrete-Structures (root)
 |__  README.md
 |__  data
 |__  gencond
 |__  |__ lstm
 |__  |__ our_model
 |__  |__ our_model_entropy
 |__ Guacamol_preprocess_properties.p
 |__ QM9_preprocess_properties.py
```

##### QM9: 
train/test/validation set size: 113885/10k/10k

`data/QM9/QM9_clean_smi_%s_smile.npy`, include train/test smile strings

`data/QM9/QM9_clean_smi_%s_smile.npz`, include train/test smile strings plus the property vector for each smiles (9 properties). This file is generated by `QM9_preprocess_properties.py` taking input as the smile string and calculating the properties.


##### Guacamol:
train/test/validation set size: 1273104/238706/79568

`data/Guacamol/%s_props.npz`, train/test/validation, include both smile strings and corresponding properties.


The raw data which include train, test and validation smile strings can be downloaded from 
 https://figshare.com/projects/GuacaMol/56639

You can either download the raw data and run `Guacamol_preprocess_properties.py` to generate the files `%s_props.npz` or directly download it from this link https://www.dropbox.com/sh/cnt3j5hlf3z29f4/AABQpmFNkQPiygfx9OE2HVxAa?dl=0

## Training

##### LSTM
Vanilla conditional LSTM


To train this model, go to folder `/gencond/lstm`, run 

`python train_QM9.py` or `python train_Guacamol.py`, 

the trained model will be saved in folder named `/gencond/lstm/output/QM9(Guacamol)`




##### Our_model 

To train our model, we first need to generate approximated normalized reward distribution to be able to sample. One could calculate it on the fly for each mini-batch. In our experiments, to speed up the training, we pre-generate the probability table and saved the sample index and the corresponding sampling probability that have non-zero rewards.


1. To generate the probability table and corresponding sample indexs, go to the folder `/gencond/our_model/`, run

 `python generate_reward_distribution_QM9(Guacamol).py`
 
 The output file will be saved in `/data/QM9(Guacamol)/sampling_prob_table.npz` This file can also be downloaded from the dropbox.
 
 2. To train the model go to the folder `/gencond/our_model/` and run
 
  `python train_qm9(Guacamol).py`
 
 

##### Our_model with entropy term
To train this model, go to folder `/gencond/our_model_entroy/` and run

 `bash cross_validation.sh`

This will train the model for different values of lambda (scaling factor for the entropy term in the loss) and save the resulting models in the folder `/gencond/our_model_entropy/output/batch_size_%sentropy_lambda_%s`


To find the best lambda, run the python file `Entropy_models_validation_performance.py` by specifying the trained model path, this gives you the performance analysis of the trained model on the validation. For example: 


`python  Entropy_models_validation_performance.py  --model_path  './output/QM9/batch_size_60entropy_lambda_0.0002/model_11_0.214.pt'`



##### Other baselines: RAML-like data augmentation and classic data augmentation

The pre-generated augmented data (data/QM9/smiles_data_augmentation_RAML_Like.npz, data/QM9/smiles_data_augmentation.npz)  are included in the dropbox.

If you want to generate the augmented data yourself, follow the instruction below:

1. Go to the folder `/data/QM9` and run 
`python RAML_sample_generation_with_edit_distance.py` 
It will take the QM9 training data and apply m-edit distnce augmentation where m is sampled from a distribution that is defined on RAML paper. It will produce a file `data/QM9/Edit_smiles.npz` which includes sampled edit distance, the probability of the sampled edit distance, augmented smiles, original smiles properties. 

2. Run `python prepare_augmented_data.py`
This will take the previesly generated file `Edit_smiles.npz` generate `data/QM9/smiles_data_augmentation_RAML_Like.npz`,`data/QM9/smiles_data_augmentation.npz` by removing the douplicated ones and also generating the correct label from the RDkit for the augmented smiles (for classic data augmentation). 

Now given the augmented data are generated and stored in data file, you can run standard lstm model by to train those baslines.

3. Go to the folder `/gencond/lstm/` and `run train_QM9_data_augmentation.py` by specifying the augmented data versions, for example:
```sh
python train_QM9_data_augmentation.py  --augmented_data '../../data/QM9/smiles_data_augmentation.npz'

python train_QM9_data_augmentation.py  --augmented_data '../../data/QM9/smiles_data_augmentation_RAML_Like.npz'
```


## Pre-trained Models
You can also find trained models saved in the dropbox link (https://www.dropbox.com/sh/cnt3j5hlf3z29f4/AABQpmFNkQPiygfx9OE2HVxAa?dl=0). To play with it, go to the folder  `/gencond/lstm/`  and run `Test-Conditional-LSTM-Simulation.ipynb` by specifying the model name and dataset path.
