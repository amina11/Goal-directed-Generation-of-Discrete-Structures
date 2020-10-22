import logging
from typing import List
import random
import torch
import numpy as np


from rnn_model import ConditionalSmilesRnn
from rnn_trainer import SmilesRnnTrainer
from rnn_utils import get_tensor_dataset, load_smiles_and_properties, set_random_seed
from smiles_char_dict import SmilesCharDictionary
from torch.utils.data import TensorDataset


from guacamol.distribution_matching_generator import DistributionMatchingGenerator


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# TODO: hard-coded to 9 properties?
PROPERTY_SIZE = 9


class SmilesRnnDistributionLearner:
    def __init__(self, data_set,graph_logger, entropy_lambda, output_dir: str, n_epochs=10, hidden_size=512, n_layers=3,max_len=100, batch_size=64, rnn_dropout=0.2, lr=1e-3, valid_every=100) -> None:
        self.data_set = data_set
        self.n_epochs = n_epochs
        self.output_dir = output_dir
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_len = max_len
        self.batch_size = batch_size
        self.rnn_dropout = rnn_dropout
        self.lr = lr
        self.valid_every = valid_every
        self.print_every = 10
        self.seed = 42 
        self.entropy_lambda = entropy_lambda
        self.graph_logger = graph_logger

    def train(self, data_path, training_set: List[str], validation_set: List[str]) -> DistributionMatchingGenerator:
        # GPU if available
        cuda_available = torch.cuda.is_available()
        device_str = 'cuda' if cuda_available else 'cpu'
        device = torch.device(device_str)
        logger.info(f'CUDA enabled:\t{cuda_available}')

        set_random_seed(self.seed, device)
        
        if self.data_set == "Guacamol" and validation_set is not None:
            print("loading guacamol dataset")

            # load data
            #train_seqs, _ = load_smiles_from_list(training_set, self.max_len)
            #valid_seqs, _ = load_smiles_from_list(validation_set, self.max_len)
    
            ## return one hot encoding of the training smiles, size  train_seqs: (123885, 102), train_prop: (123885, 9)
            train_x, train_y  = load_smiles_and_properties(training_set, self.max_len)
            #valid_x, valid_y = load_smiles_and_properties(validation_set, self.max_len)
            valid_x, valid_y = np.load(validation_set).values()
        else:
            train_seqs, train_prop = load_smiles_and_properties(training_set, False)
            #original validation set train_seqs[:10000,:] is left as validation set to report the result for cross validation
            train_x, train_y = train_seqs[10000:,:], train_prop[10000:,:]
            valid_x, valid_y = train_seqs[:10000,:], train_prop[:10000,:]
            
        
        #scale the property to fall between -1 and 1
        all_y = np.concatenate((train_y, valid_y), axis=0)
        #Normalizer = np.abs(all_y).max(axis = 0)
        mean = np.mean(all_y, axis = 0)
        std = np.std(all_y, axis = 0)
        train_y = (train_y - mean) / std
        valid_y = (valid_y -mean) / std
        
        
        prob_table, index_table = np.load(data_path + '/sampling_prob_table.npz', allow_pickle=True).values()

      
        # convert to torch tensor, input, output smiles and properties    
        train_set = get_tensor_dataset(train_x, train_y)
        valid_set = get_tensor_dataset(valid_x, valid_y)
        valid_set = TensorDataset(valid_set[0], valid_set[1])
       
        
        
        sd = SmilesCharDictionary()
        n_characters = sd.get_char_num()

        # build network
        smiles_model = ConditionalSmilesRnn(input_size=n_characters,
                                            property_size=PROPERTY_SIZE,
                                            hidden_size=self.hidden_size,
                                            output_size=n_characters,
                                            n_layers=self.n_layers,
                                            rnn_dropout=self.rnn_dropout)
        
        # wire network for training
        optimizer = torch.optim.Adam(smiles_model.parameters(), lr=self.lr)
        criteria = torch.nn.CrossEntropyLoss(ignore_index=sd.pad_idx)
        #criteria = CrossEntropyLoss(aggregate='mean')

        #pretrained_model = './output/Guacamol/model_48_0.141.pt'
        #if pretrained_model is not None:
            #smiles_model.load_state_dict(torch.load(pretrained_model))
            #print('the pretrained model is loaded')

        
        trainer = SmilesRnnTrainer(self.entropy_lambda,
                                   self.graph_logger,
                                   normalizer_mean = mean,
                                   normalizer_std = std,
                                   model=smiles_model,
                                   criteria=criteria,
                                   optimizer=optimizer,
                                   device=device,
                                   train_x = train_x,
                                   log_dir=self.output_dir)

        trainer.fit(train_set, valid_set, prob_table, index_table,
                     self.n_epochs, 
                     batch_size=self.batch_size,
                     print_every=self.print_every,
                     valid_every=self.valid_every,
                     num_workers=0)
