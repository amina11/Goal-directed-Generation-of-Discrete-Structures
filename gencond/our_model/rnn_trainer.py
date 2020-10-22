import logging
import os
from glob import glob
from time import time
from typing import List
from scipy.spatial.distance import cdist
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from rnn_utils import save_model, time_since, get_mini_batch
import random
from torch.distributions.categorical import Categorical
from operator import itemgetter 

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from gencond.properties import PROPERTIES, TYPES

from rdkit import Chem
from smiles_char_dict import SmilesCharDictionary
sd = SmilesCharDictionary()

## calculate the entropy term 
from rnn_utils import load_rnn_model, load_smiles_from_list, rnn_start_token_vector

def greedy_entropy_est(model, y, max_len=100):
    """ Quick re-implementation of greedy entropy estimation """
    H_est = 0.0
    device = y.device

    # initialization for LSTM sampling
    hidden = model.init_hidden(y.shape[0], device)
    inp = rnn_start_token_vector(y.shape[0], device)
    
    # run character-by-character...
    for char in range(max_len):
        # one step forward
        output, hidden = model(inp, y.expand(y.shape[0], -1), hidden)
        action = output.argmax(dim=2)

        log_prob = output - torch.logsumexp(output, dim=-1, keepdim=True)
        H_est += -(log_prob.exp() * log_prob).sum()
        inp = action

    return H_est


class SmilesRnnTrainer:

    def __init__(self, entropy_lambda, graph_logger,normalizer_mean,normalizer_std, model, criteria, optimizer, device, train_x,  sample_size = 10, log_dir=None, clip_gradients=True) -> None:
        self.model = model.to(device)
        self.criteria = criteria.to(device)
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.clip_gradients = clip_gradients
        self.train_x = train_x
        self.sample_size = sample_size
        self.entropy_lambda = torch.as_tensor(entropy_lambda)
        self.graph_logger = graph_logger
        self.mean = normalizer_mean
        self.std = normalizer_std
        
    def process_batch(self,properties, P_in, index):
        
        
        batch_size = properties.size(0)
        properties = properties.to(self.device)
        selected_ind = []
        for j in range(batch_size):
        
            select = np.random.choice(P_in[j].shape[0], self.sample_size, p = P_in[j])
            selected_ind.append(np.array(index[j])[select])
      
            
        selected = np.array(selected_ind)
        target_prop = (properties.repeat(1,self.sample_size)).reshape(selected.shape[0]* selected.shape[1],9)
        index = selected.reshape(batch_size * self.sample_size,1)
        selected_x = np.squeeze(self.train_x[index,:])
        selected_inp= torch.from_numpy(selected_x[:, :-1]).long().to(self.device)
        selected_tgt = torch.from_numpy(selected_x[:, 1:]).long().to(self.device)
        hidden_selected = self.model.init_hidden(selected_inp.size(0), self.device)
        selected_output, selected_hidden = self.model(selected_inp, target_prop, hidden_selected) 
        selected_logits = selected_output.view(selected_output.size(0) * selected_output.size(1), -1)
        ML_loss = self.criteria(selected_logits,  selected_tgt.view(-1))
        entropy = torch.as_tensor(0.0).to(self.device)
        ## entropy term 
        final_loss = ML_loss  - entropy * self.entropy_lambda.to(self.device)
      
        return final_loss,ML_loss , entropy, batch_size

    def train_on_batch(self, properties, P_in, index):

        # setup model for training
        self.model.train()
        self.model.zero_grad()

        # forward / backward
        final_loss,MLE, entropy, batch_size = self.process_batch(properties, P_in, index)
        final_loss.backward()

        # optimize
        if self.clip_gradients:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return final_loss.item(), MLE.item(),entropy.item(), batch_size
    
    def test_on_batch(self, batch, properties):

        # setup model for evaluation
        self.model.eval()

        # forward
        inp, tgt = batch[:, :-1], batch[:, 1:]
      
        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        batch_size = inp.size(0)
        properties = properties.to(self.device)
        hidden = self.model.init_hidden(inp.size(0), self.device)
        output, hidden = self.model(inp, properties, hidden)
        output = output.view(output.size(0) * output.size(1), -1)
        loss = self.criteria(output, tgt.view(-1))
        #final_loss, entropy, MLE, batch_size = self.process_batch(batch, properties, k_sampled)
        return loss.item(), batch_size
    
    
    def test_on_batch_MSE(self, batch, properties):

        # setup model for evaluation
        self.model.eval()

        # forward
        inp, tgt = batch[:, :-1], batch[:, 1:]

        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        batch_size = inp.size(0)
        properties = properties.to(self.device)
        hidden = self.model.init_hidden(inp.size(0), self.device)
        output, hidden = self.model(inp, properties, hidden)
        indices = output.argmax(dim = 2)
        smiles = np.array(sd.matrix_to_smiles(indices))
        valid = []
        valid_index = []
        prop = []        
        for i in range(smiles.shape[0]):
            
            mol = Chem.MolFromSmiles(smiles[i])
            try:
                valid.append(Chem.MolToSmiles(mol))
                prop.append([p(mol) for p in PROPERTIES.values()])
                valid_index.append(i)
            except:
                pass   
            
        
        if np.array(valid_index).shape[0] == 0:
            loss = 0
        else:  
            prop = (np.array(prop) - self.mean ) / self.std
            loss = np.absolute(np.array(prop) - properties[np.array(valid_index),:].cpu().detach().numpy()).mean()   
        
        #output = output.view(output.size(0) * output.size(1), -1)
        #loss = self.criteria(output, tgt.view(-1))
        #final_loss, entropy, MLE, batch_size = self.process_batch(batch, properties, k_sampled)
        return loss, batch_size
   
    def validate_MSE(self, data_loader, n_molecule):
        """Runs validation and reports the average loss"""
        valid_losses = []
        with torch.no_grad():
            for batch_all in data_loader:
                batch = batch_all[0]
                properties = batch_all[1]
                
                loss, size = self.test_on_batch_MSE(batch, properties)
                if loss > 0:
                    valid_losses += [loss]
                else: 
                    valid_losses = valid_losses
        return np.array(valid_losses).mean()
    

    def validate(self, data_loader, n_molecule):
        """Runs validation and reports the average loss"""
        valid_losses = []
        with torch.no_grad():
            for batch_all in data_loader:
                batch = batch_all[0]
                properties = batch_all[1]
                
                loss, size = self.test_on_batch(batch, properties)
                valid_losses += [loss]
        return np.array(valid_losses).mean()

    def train_extra_log(self, n_molecules):
        pass

    def valid_extra_log(self, n_molecules):
        pass

    def fit(self, training_data, test_data, prob_table,index_table, n_epochs, batch_size, print_every, valid_every, num_workers=0):
        
        training_round = _ModelTrainingRound(self, training_data, test_data, prob_table,index_table, n_epochs, batch_size, print_every, valid_every, num_workers)
        return training_round.run()


class _ModelTrainingRound:
    """
    Performs one round of model training.

    Is a separate class from ModelTrainer to allow for more modular functions without too many parameters.
    This class is not to be used outside of ModelTrainer.
    """
    class EarlyStopNecessary(Exception):
        pass

    def __init__(self, model_trainer: SmilesRnnTrainer, training_data, test_data, prob_table, index_table, n_epochs, batch_size, print_every, valid_every, num_workers=0) -> None:
        self.model_trainer = model_trainer
        self.training_data = training_data
        self.test_data = test_data
        self.prob_table = prob_table
        self.index_table = index_table
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.print_every = print_every
        self.valid_every = valid_every
        self.num_workers = num_workers

        self.start_time = time()
        
        self.unprocessed_entropy: List[float] = []
        self.unprocessed_final_losses: List[float] = []
        self.unprocessed_MLE: List[float] = [] 
            
            
        self.all_train_losses: List[float] = []
        self.all_valid_losses: List[float] = []
        self.min_valid_loss = np.inf
        self.min_avg_train_loss = np.inf
        
        self.n_molecules_so_far = 0
        self.has_run = False
        self.iter = 0
        
        

    def run(self):
        if self.has_run:
            raise Exception('_ModelTrainingRound.train() can be called only once.')

        try:
            for epoch_index in range(1, self.n_epochs + 1):
                self._train_one_epoch(epoch_index)
                avg_valid_loss = self._validate_current_model()
                #self._save_current_model(self.model_trainer.log_dir, epoch_index, avg_valid_loss)

            self._validation_on_final_model()
            
        except _ModelTrainingRound.EarlyStopNecessary:
            logger.error('Probable explosion during training. Stopping now.')

        self.has_run = True
        return self.all_train_losses, self.all_valid_losses

    def _train_one_epoch(self, epoch_index: int):
        logger.info(f'EPOCH {epoch_index}')

        
        
        tensor, props = self.training_data[0], self.training_data[1]
        sample_idxes = list(range(props.shape[0]))
        random.shuffle(sample_idxes)
        
        epoch_t0 = time()
        self.unprocessed_final_losses.clear()
        self.unprocessed_entropy.clear()
        self.unprocessed_MLE.clear()
        
        for batch_index in range(int(np.ceil(len(sample_idxes)/ self.batch_size))):
            selected_idx = sample_idxes[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
            properties,P_in, index = get_mini_batch(tensor, props, self.prob_table,self.index_table, selected_idx)
            self._train_one_batch(batch_index, properties, P_in, index, epoch_index, epoch_t0)
        
        
    def _train_one_batch(self, batch_index, properties, P_in, index, epoch_index, train_t0):
        final_loss,MLE,entropy, size = self.model_trainer.train_on_batch(properties, P_in, index)
        
        self.unprocessed_final_losses += [final_loss]
        self.unprocessed_entropy += [entropy]
        self.unprocessed_MLE +=[MLE]
        self.n_molecules_so_far += size

        # report training progress?
        if batch_index > 0 and batch_index % self.print_every == 0:
            self._report_training_progress(batch_index, epoch_index, epoch_start=train_t0)

        # report validation progress?
        if batch_index >= 0 and batch_index % self.valid_every == 0:
            self._report_validation_progress(epoch_index)

    def _report_training_progress(self, batch_index, epoch_index, epoch_start):
        mols_sec = self._calculate_mols_per_second(batch_index, epoch_start)

        # Update train losses by processing all losses since last time this function was executed
        avg_final_loss = np.array(self.unprocessed_final_losses).mean()
        avg_entropy = np.array(self.unprocessed_entropy).mean()
        avg_MLE = np.array(self.unprocessed_MLE).mean()
        
        self.all_train_losses += [avg_final_loss]
        
        self.unprocessed_entropy.clear()
        self.unprocessed_MLE.clear()
        self.unprocessed_final_losses.clear()
        logger.info(
            'TRAIN | '
            f'elapsed: {time_since(self.start_time)} | '
            f'epoch|batch : {epoch_index}|{batch_index} ({self._get_overall_progress():.1f}%) | '
            f'molecules: {self.n_molecules_so_far} | '
            f'mols/sec: {mols_sec:.2f} | '
            f'final_loss: {avg_final_loss:.4f}|'
            f'selected_loss:{avg_MLE:.4f}|' 
            f'MLE:{avg_entropy:.4f}')
        
        self.iter = self.iter + 1   
        self.model_trainer.train_extra_log(self.n_molecules_so_far)
        self._check_early_stopping_train_loss(avg_MLE)
        ## tensor board
        self.model_trainer.graph_logger.add_scalar('average_trainingloss', avg_final_loss, global_step = self.iter, save_csv= True)
        #self.model_trainer.writer.add_scalar('average_MLE', avg_MLE, self.iter)
        #self.model_trainer.writer.add_scalar('average_entropy', avg_entropy, self.iter)
        
        
    def _calculate_mols_per_second(self, batch_index, epoch_start):
        """
        Calculates the speed so far in the current epoch.
        """
        train_time_in_current_epoch = time() - epoch_start
        processed_batches = batch_index + 1
        molecules_in_current_epoch = self.batch_size * processed_batches
        return molecules_in_current_epoch / train_time_in_current_epoch

    def _report_validation_progress(self, epoch_index):
        avg_valid_loss = self._validate_current_model()
        self.model_trainer.graph_logger.add_scalar('average_validloss', avg_valid_loss, global_step = self.iter, save_csv= True)
        self._log_validation_step(epoch_index, avg_valid_loss)
        self._check_early_stopping_validation(avg_valid_loss)

        # save model?
        if self.model_trainer.log_dir:
            if avg_valid_loss <= min(self.all_valid_losses):
                self._save_current_model(self.model_trainer.log_dir, epoch_index, avg_valid_loss)

    def _validate_current_model(self):
        """
        Validate the current model.

        Returns: Validation loss.
        """
        test_loader = DataLoader(self.test_data,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)
        
           
        avg_valid_loss = self.model_trainer.validate_MSE(test_loader, self.n_molecules_so_far)
        self.all_valid_losses += [avg_valid_loss]
        return avg_valid_loss

    def _log_validation_step(self, epoch_index, avg_valid_loss):
        """
        Log the information about the validation step.
        """
        logger.info(
            'VALID | '
            f'elapsed: {time_since(self.start_time)} | '
            f'epoch: {epoch_index}/{self.n_epochs} ({self._get_overall_progress():.1f}%) | '
            f'molecules: {self.n_molecules_so_far} | '
            f'valid_loss: {avg_valid_loss:.4f}')
        self.model_trainer.valid_extra_log(self.n_molecules_so_far)
        logger.info('')

    def _get_overall_progress(self):
        total_mols = self.n_epochs * len(self.training_data)
        return 100. * self.n_molecules_so_far / total_mols

    def _validation_on_final_model(self):
        """
        Run validation for the final model and save it.
        """
        valid_loss = self._validate_current_model()
        logger.info(
            'VALID | FINAL_MODEL | '
            f'elapsed: {time_since(self.start_time)} | '
            f'molecules: {self.n_molecules_so_far} | '
            f'valid_loss: {valid_loss:.4f}')

        if self.model_trainer.log_dir:
            self._save_model(self.model_trainer.log_dir, 'final', valid_loss)

    def _save_current_model(self, base_dir, epoch, valid_loss):
        """
        Delete previous versions of the model and save the current one.
        """
        for f in glob(os.path.join(base_dir, 'model_*')):
            os.remove(f)

        self._save_model(base_dir, epoch, valid_loss)

    def _save_model(self, base_dir, info, valid_loss):
        """
        Save a copy of the model with format:
                model_{info}_{valid_loss}
        """
        base_name = f'model_{info}_{valid_loss:.3f}'
        logger.info(base_name)
        save_model(self.model_trainer.model, base_dir, base_name)

    def _check_early_stopping_train_loss(self, avg_train_loss):
        """
        This function checks whether the training has exploded by verifying if the avg training loss
        is more than 10 times the minimal loss so far.

        If this is the case, a EarlyStopNecessary exception is raised.
        """
        threshold = 10 * self.min_avg_train_loss
        if avg_train_loss > threshold:
            print('average_train_loss', avg_train_loss)
            print('threshold', threshold)
            raise _ModelTrainingRound.EarlyStopNecessary()

        # update the min train loss if necessary
        if avg_train_loss < self.min_avg_train_loss:
            self.min_avg_train_loss = avg_train_loss

    def _check_early_stopping_validation(self, avg_valid_loss):
        """
        This function checks whether the training has exploded by verifying if the validation loss
        has more than doubled compared to the minimum validation loss so far.

        If this is the case, a EarlyStopNecessary exception is raised.
        """
        threshold = 2 * self.min_valid_loss
        if avg_valid_loss > threshold:
            print('average valid loss:', avg_valid_loss)
            print('threshold', threshold)
            raise _ModelTrainingRound.EarlyStopNecessary()

        if avg_valid_loss < self.min_valid_loss:
            self.min_valid_loss = avg_valid_loss

