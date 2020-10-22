# +
import argparse
import numpy as np
import random
parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES RNN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_name', default='QM9')
parser.add_argument('--data_path', default='../../data/QM9/',
                    help='path to save the knn sampler matrix')
parser.add_argument('--train_data', default='../../data/QM9/QM9_clean_smi_train_smile.npz',
                    help='Full path to SMILES file containing training data')
parser.add_argument('--valid_data', default='',
                    help='Full path to SMILES file containing validation data')
'''
parser.add_argument('--data_name', default='Guacamol')
parser.add_argument('--data_path', default='../../data/Guacamol/',
                    help='path to save the knn sampler matrix')
parser.add_argument('--train_data', default='../../data/Guacamol/train_props.npz',
                        help='Full path to SMILES file containing training data')
parser.add_argument('--valid_data', default='../../data/Guacamol/valid_props.npz',
                        help='Full path to SMILES file containing validation data')
parser.add_argument('--prop_model', default="../../data/Guacamol/prior.pkl.gz", help='Saved model for properties distribution')    
parser.add_argument('--output_dir', default='./output/Guacamol/', help='Output directory')
'''                        

parser.add_argument('--max_len', default=100, type=int, help='Max length of a SMILES string')
args,_ = parser.parse_known_args()


# +
### helper functions
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
sys.path.append('./../../gencond/')
from properties import PROPERTIES

def get_properties(valid_mol_list):
    prop = []
    i = 0
    if type(valid_mol_list)!=list:
        valid_mol_list = [valid_mol_list]
    
    for row in range(len(valid_mol_list)):
        sample = valid_mol_list[row]
        mol = Chem.MolFromSmiles(sample)
        try:
            prop.append([p(mol) for p in PROPERTIES.values()])
        except:
            pass # molecule invalid
    return prop  


# -

from rnn_utils import get_tensor_dataset, load_smiles_and_properties, set_random_seed
training_set=args.train_data
validation_set=args.valid_data


data_name = args.data_name
data_name

if data_name == 'QM9': 
    train_seqs, train_prop = load_smiles_and_properties(training_set, args.max_len)
    train_x, train_y = train_seqs[10000:,:], train_prop[10000:,:]
    valid_x, valid_y = train_seqs[:10000,:], train_prop[:10000,:]
else:
    train_x, train_y  = load_smiles_and_properties(training_set, args.max_len)
    valid_x, valid_y = load_smiles_and_properties(validation_set, args.max_len)

## normalize the properties 
all_y = np.concatenate((train_y, valid_y), axis=0)
#Normalizer = np.abs(all_y).max(axis = 0)
mean = np.mean(all_y, axis = 0)
std = np.std(all_y, axis = 0)
train_y = (train_y - mean) / std
valid_y = (valid_y -mean) / std

# find the k nearest neigbour for every instances in the training data and save the index in K
## calculate the pairwise distance of the trianing instances and output knearest instances index for
from scipy.spatial.distance import cdist
batch_size = 500
num_nearest = 10
k_sampled_train = np.zeros([train_y.shape[0], num_nearest])
for i in range (int(train_y.shape[0] / batch_size) + 1):
    properties = train_y[i * batch_size: (i+1)* batch_size,:]
    K = cdist(properties, train_y)
    #selected = np.argsort(K, axis=1)[:,:num_nearest]
    selected = np.argsort(K)[:,:num_nearest].astype(int)
    print(i)
    k_sampled_train[i * batch_size: (i+1)* batch_size,:] = selected

print('training data done')
np.save(args.data_path + '/Knn_sampled_train.npy', k_sampled_train)


k_sampled_valid = np.zeros([valid_y.shape[0], num_nearest])
for i in range (int(valid_y.shape[0] / batch_size) + 1):
    properties = valid_y[i * batch_size: (i+1)* batch_size,:]
    K = cdist(properties, valid_y)
    selected = np.argsort(K, axis=1)[:,:num_nearest]
    #selected = np.argpartition(K,num_nearest, axis = 1)[:,:num_nearest].astype(int)
    print(i)
    k_sampled_valid[i * batch_size: (i+1)* batch_size,:] = selected

print('valid data done')
np.save(args.data_path + '/Knn_sampled_valid.npy', k_sampled_valid)

# ### make a new data set by above 


# # Generate augmented data

#  ## 1. in our way

### make the augmented data set following our strategy and save it 
#smiles, prop = np.load('../../data/QM9/QM9_clean_smi_train_smile.npz', allow_pickle=True).values()
smiles, prop = np.load('../../data/Guacamol/train_props.npz', allow_pickle=True).values()
import time


K = np.load('../../data/Guacamol/Knn_sampled_train.npy')

from scipy.spatial.distance import cdist
new_data_smile= []
new_data_y= []
batch_size = prop.shape[0]
num_nearest = 10
#k_sampled_train = np.zeros([prop.shape[0], num_nearest])
#int(train_y.shape[0] / batch_size) + 1
for i in range (int(prop.shape[0] / batch_size) + 1):
    t = time.time()
    properties = prop[i * batch_size: (i+1)* batch_size,:]
    #K = cdist(properties, prop)
    #selected = np.argsort(K, axis=1)[:,:num_nearest]
    #selected = np.argsort(K)[:,:num_nearest].astype(int)
    selected = K[i * batch_size: (i+1)* batch_size,:].astype(int)
    new_data_smile.append(smiles[selected.flatten()])
    new_data_y.append(np.repeat(properties, repeats = [num_nearest], axis=0))
    #k_sampled_train[i * batch_size: (i+1)* batch_size,:] = selected
    print(i)

s = np.array(new_data_smile[0])

y = np.array(new_data_y[0])

s.shape

s[10:20]

np.savez(args.data_path + '/Knn_augmented_ours_10sample', s, y)


args.data_path

# +

ss, yy = np.load(args.data_path  + 'train_props.npz')
# -

# ## 2. RAML way 

aug_s, aug_y = np.load('../../data/QM9/smiles_data_augmentation.npz').values()
train_s, train_y = np.load('../../data/QM9/QM9_clean_smi_train_smile.npz', allow_pickle=True).values()
full_s = np.concatenate((aug_s, train_s), axis = 0)
full_y = np.concatenate((aug_y,train_y), axis = 0)
np.savez(args.data_path + '/RAM_augmented_and_train_data', full_s, full_y)
