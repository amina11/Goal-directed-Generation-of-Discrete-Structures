# + {"endofcell": "--"}
import pickle
import gzip
import argparse
import os
import sys
import numpy as np
from smiles_rnn_distribution_learner import SmilesRnnDistributionLearner

from guacamol.utils.helpers import setup_default_logger
setup_default_logger()
from atalaya import Logger


parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES RNN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--train_data', default='../../data/Guacamol/train_props.npz',
                        #help='Full path to SMILES file containing training data')
#parser.add_argument('--valid_data', default='../../data/Guacamol/valid_props.npz',
                        #help='Full path to SMILES file containing validation data')

# # +
#parser.add_argument('--prop_model', default="../../data/Guacamol/prior.pkl.gz", help='Saved model for properties distribution')    
#parser.add_argument('--output_dir', default='./output/Guacamol/', help='Output directory')
# -

parser.add_argument('--train_data', default='../../data/QM9/QM9_clean_smi_train_smile.npz',
                    help='Full path to SMILES file containing training data')
parser.add_argument('--valid_data', default='', help='Full path to SMILES file containing validation data')
parser.add_argument('--data_path', default = '../../data/QM9', help='training data path')
parser.add_argument('--output_dir', default='./output/QM9', help='Output directory')


parser.add_argument('--entropy_lambda', default=0.0, type=float, help='weighting of the entropy term in the loss')
parser.add_argument('--batch_size', default=20, type=int, help='Size of a mini-batch for gradient descent')
parser.add_argument('--valid_every', default=1000, type=int, help='Validate every so many batches')
parser.add_argument('--print_every', default=10, type=int, help='Report every so many batches')
parser.add_argument('--n_epochs', default=100, type=int, help='Number of training epochs')
parser.add_argument('--max_len', default=100, type=int, help='Max length of a SMILES string')
parser.add_argument('--hidden_size', default=512, type=int, help='Size of hidden layer')
parser.add_argument('--n_layers', default=3, type=int, help='Number of layers for training')
parser.add_argument('--rnn_dropout', default=0.2, type=float, help='Dropout value for RNN')
parser.add_argument('--lr', default=1e-3, type=float, help='RNN learning rate')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
args,_ = parser.parse_known_args()

args.output_dir = args.output_dir + '/batch_size_' + str(args.batch_size) + '_2'

if args.output_dir is None:
    args.output_dir = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

graph_logger = Logger(
    name="QM9_batchszie_" + str(args.batch_size) + "_ourmodel_baobab",         # name of the logger
    path=args.output_dir + "logs",        # path to logs
    verbose=True,       # logger in verbose mode
    grapher="visdom",
    server="http://send2.visdom.xyz",
    port=8999
)

graph_logger.add_parameters(args)

trainer = SmilesRnnDistributionLearner(data_set = "QM9",
                                           graph_logger= graph_logger,
                                           entropy_lambda = args.entropy_lambda,
                                           output_dir=args.output_dir,
                                           n_epochs=args.n_epochs,
                                           hidden_size=args.hidden_size,
                                           n_layers=args.n_layers,
                                           max_len=args.max_len,
                                           batch_size=args.batch_size,
                                           rnn_dropout=args.rnn_dropout,
                                           lr=args.lr,
                                           valid_every=args.valid_every)

trainer.train(data_path = args.data_path,training_set=args.train_data, validation_set = args.valid_data)

print(f"All done, your trained model is in {args.output_dir}")

       
# --
