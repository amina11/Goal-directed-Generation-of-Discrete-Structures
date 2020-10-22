import sys
#sys.path.insert(0, '..')
import matplotlib.pyplot as plt
from rdkit import Chem
from pathlib import Path
import pickle, gzip, torch
import numpy as np
import warnings
from rnn_utils import load_rnn_model
import action_sampler, rnn_sampler
from gencond.properties import PROPERTIES, TYPES
import argparse
# %matplotlib inline
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES RNN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_path', default='./output/QM9/batch_size_60entropy_lambda_0.0001/model_7_0.213.pt')
parser.add_argument('--output_dir', default='./output/QM9/', help='Output directory')
parser.add_argument('--data_path', default='../../data/QM9')
parser.add_argument('--train_data', default='../../data/QM9/QM9_clean_smi_train_smile.npz',
                    help='Full path to SMILES file containing training data')
args,_ = parser.parse_known_args()


#test_smiles, test_prop = np.load("../../data/QM9/QM9_clean_smi_test_smile.npz", allow_pickle=True).values()
train_smiles, train_prop = np.load("../../data/QM9/QM9_clean_smi_train_smile.npz", allow_pickle=True).values()

val_smiles, val_prop = train_smiles[:10000],train_prop[:10000,:]
train_smiles, train_prop = train_smiles[10000:],train_prop[10000,:]
train_smiles = train_smiles.tolist()
print('training smiles num:', len(train_smiles))
print('validation smiles num:', val_smiles.shape)
#print('test smiles num:', test_smiles.shape)

def normalize(prop):
    train_smiles, train_prop = np.load("../../data/QM9/QM9_clean_smi_train_smile.npz",allow_pickle=True).values()
    mean = np.mean(train_prop, axis = 0)
    std = np.std(train_prop, axis = 0)
    return (prop - mean) / std


model_def = Path(args.model_path).with_suffix('.json')
model = load_rnn_model(model_def, args.model_path, device='cpu', copy_to_cpu=True)
sampler = rnn_sampler.ConditionalSmilesRnnSampler(device='cpu')


#1. generate
sample_size = 1
generated = sampler.sample(model, normalize(val_prop), sample_size)

#2. check generation performance
from itertools import chain
valid_mol = []
num_valid = 0
for row in range(generated.shape[0]):
    valid = []
    for samples in generated[row]:
        mol = Chem.MolFromSmiles(samples)
        try:
            valid.append(Chem.MolToSmiles(mol))
            num_valid = num_valid + 1

        except:
            pass # molecule invalid
    valid_mol.append(valid)
validity_ = num_valid / (len(generated) * generated[0].shape[0])
print('validity')
print('unicity')
print('noelty')

print(np.around(validity_, decimals=6))
str_list = [x for x in valid_mol if x != []]
valid_mol_list = list(chain.from_iterable(str_list))
num_unique = len(np.unique(np.array(valid_mol_list), return_index=True)[1])
print(np.around(num_unique / num_valid, decimals=6))

#common = list(set(train_smiles.intersection(valid_mol_list)))
common = list(set(train_smiles).intersection((valid_mol_list)))  # remove the validation set
novelty_= (num_valid - len(common)) / num_valid
print(np.around(novelty_, decimals = 6))


# 3. check condtional performance
#3.1 generate properties
simulated = []
#smiles_matches = []
index = []
for row in range(generated.shape[0]):
    prop = []
    canonical = []
    for sample in generated[row]:
        mol = Chem.MolFromSmiles(sample)
        try:
            prop.append([p(mol) for p in PROPERTIES.values()])
            canonical.append(Chem.MolToSmiles(mol))
        except:
            pass # molecule invalid

    if prop !=[]:
        index.append(row)

    simulated.append(np.array(prop))
    #smiles_matches.append(np.sum([test_smiles == s for s in canonical]))

if simulated[0].shape[0]>1:
    mean_prop= [item.mean(axis = 0 ) for item in simulated]

else:
    mean_prop = simulated

valid_smiles_prop = np.array(mean_prop)[index]
generated_valid_smiles_prop = np.vstack(valid_smiles_prop)
target_valid_prop = val_prop[index]


#3.2 MSE, MAE report
MSE = np.mean(np.power(generated_valid_smiles_prop - target_valid_prop, 2))
print('MSE')
print(np.around(MSE, decimals = 6))

MAE = np.mean(np.absolute(generated_valid_smiles_prop - target_valid_prop))
print('MAE')
print(np.around(MAE, decimals = 6))


# -logp
from torch.utils.data import DataLoader
from atalaya import Logger


sys.path.insert(0, '../lstm')
from rnn_utils import load_smiles_from_list
from smiles_char_dict import SmilesCharDictionary

from torch.utils.data import DataLoader
from atalaya import Logger

def get_tensor_dataset(smiles_array, properties_array):
    """
    Gets a numpy array of indices, convert it into a Torch tensor,
    divided it into inputs and targets and wrap it
    into a TensorDataset

    Args:
        numpy_array: to be converted

    Returns: a TensorDataset
    """

    tensor = torch.from_numpy(smiles_array).long()
    props = torch.from_numpy(properties_array).float()

    inp = tensor[:, :-1]
    target = tensor[:, 1:]

    return TensorDataset(inp, target, props)

graph_logger = ''

_, train_prop = np.load(args.train_data, allow_pickle=True).values()
mean = np.mean(train_prop, axis = 0)
std = np.std(train_prop,axis =0)


sd = SmilesCharDictionary()
criterion = torch.nn.CrossEntropyLoss(ignore_index=sd.pad_idx)

device = 'cpu'

test_s,_ = load_smiles_from_list(val_smiles, val_prop)
test_set = get_tensor_dataset(test_s, normalize(val_prop))
data_loader = DataLoader(test_set,batch_size=500,shuffle=False,num_workers=1,pin_memory=True)


sys.path.insert(0, '../lstm')
from rnn_trainer import SmilesRnnTrainer



optimizer = torch.optim.Adam(model.parameters(), 0.01)
criterion = torch.nn.CrossEntropyLoss(ignore_index=sd.pad_idx)
trainer = SmilesRnnTrainer(graph_logger, model=model,
                                   normalizer_mean = mean,
                                   normalizer_std = std,
                                   criteria=[criterion],
                                   optimizer=optimizer,
                                   device=device,
                                   log_dir=None)

_logp = trainer.validate(data_loader, 1)

print('negitive log likeligood on validation set:',np.round(_logp, decimals = 6))
