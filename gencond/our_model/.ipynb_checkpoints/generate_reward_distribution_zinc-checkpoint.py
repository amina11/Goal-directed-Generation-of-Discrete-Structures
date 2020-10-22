# +
'''
This file will take training data and generate the probaility(approximated normalized reward distribution)  table that can be used to sample training instances during traing
 input: data_path: where to save the generated probability table and corresponding  indexs
        train_data: training data to load the data
        tao: the thresh holding parameter that describe if the distance of the sample property to the target property bigger ##                than tao such sample has zero prob to be sampled.

output: the generated probability table and corresponding index of the samples are saved under the path called train_data with ##          the name sampling_prob_table.npz, load it later with prob_table, index_table= np.load(args.data_path + 
        '/sampling_prob_table.npz', allow_pickle=True).values()
'''
import pickle
import argparse
import os
import sys
import numpy as np
parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES RNN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', default = '../../data/ZINC', help='training data path')
parser.add_argument('--train_data', default='../../data/ZINC/ZINC_clean_smi_train_smile.npz',
                        help='Full path to SMILES file containing training data')
#parser.add_argument('--valid_data', default='../../data/Guacamol/valid_props_smaller.npz',
                        #help='Full path to SMILES file containing validation data')
                  
parser.add_argument('--max_len', default=100, type=int, help='Max length of a SMILES string')
parser.add_argument('--tao', default=0.4, type=float, 
                    help='The threshold value used to zero out the prob of samples, the allowed maximum error on the property')
args,_ = parser.parse_known_args()
# -

# 1. prepare the training and validation data
from rnn_utils import get_tensor_dataset, load_smiles_and_properties, set_random_seed

# +
#train_seqs, train_prop = load_smiles_and_properties(args.train_data, False)

train_seqs, train_prop = load_smiles_and_properties(args.train_data, False)
train_x, train_y = train_seqs[10000:,:], train_prop[10000:,:]
valid_x, valid_y = train_seqs[:10000,:], train_prop[:10000,:]
'''
train_x, train_y  = load_smiles_and_properties(training_set, args.max_len)
valid_x, valid_y = load_smiles_and_properties(validation_set, args.max_len) 
'''
all_y = np.concatenate((train_y, valid_y), axis=0)
mean = np.mean(all_y, axis = 0)
std = np.std(all_y, axis = 0)
#np.save(args.data_path + '/normalizer.py', [mean, std])
train_y = (train_y - mean ) / std
valid_y = (valid_y - mean) / std


# +
mean = np.mean(train_prop, axis = 0)
std = np.std(train_prop, axis = 0)
#np.save(args.data_path + '/normalizer.py', [mean, std])
train_y = (train_prop - mean ) / std

args.tao = 0.7
## 2. calculat the probability table
prob_list = []
nested_list = []
len_ = []
from scipy.spatial.distance import cdist
batch_size = 200
for i in range(int(np.ceil(train_y.shape[0] / batch_size))):
    properties = train_y[i * batch_size: (i+1)* batch_size,:]
    K = cdist(properties, train_y,'minkowski', p=1)
    index_ = np.where(K< args.tao)
    print(i)
    for j in range(properties.shape[0]):
        ind = np.where(index_[0] == j)
        nested_list.append(index_[1][ind])
        d =(np.absolute(properties[j, :] - train_y[index_[1][ind],:])).sum(axis = 1)
        P = np.exp(-d) / np.exp(-d).sum()
        len_.append(P.shape)
        prob_list.append(P)
np.savez(args.data_path + '/sampling_prob_table', prob_list, nested_list)     
# -

prop, index = np.load(args.data_path + '/sampling_prob_table.npz', allow_pickle=True).values()

prop

train_x, train_prop = np.load(args.train_data, allow_pickle=True).values()

train_x

train_prop[10]

index[10]

train_prop[353]


