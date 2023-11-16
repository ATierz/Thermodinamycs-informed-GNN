import os
import json
import argparse

import torch
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl

from src.dataLoader.dataset import GraphDataset
from src.gnn import PlasticityGNN
from src.utils import str2bool
from src.evaluate import generate_results

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

# Study Case
parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
parser.add_argument('--pretrain_weights', default=r'weights\epoch=331-step=866852_BUENO.ckpt', type=str, help='name')

# Dataset Parametersa
parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
# parser.add_argument('--dset_name', default='d6_waterk10_noTensiones_radius_.pt', type=str, help='dataset directory')
parser.add_argument('--dset_name', default=r'jsonFiles\dataset_1.json', type=str, help='dataset directory')

# Net Parameters
parser.add_argument('--n_hidden', default=2, type=int, help='number of hidden layers per MLP')
parser.add_argument('--dim_hidden', default=124, type=int, help='dimension of hidden units')
parser.add_argument('--passes', default=6, type=int, help='number of message passing')
# Training Parameters
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--fixLr', default=True, type=str2bool, help='flag fig learning rate')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--lambda_d', default=10, type=float, help='data loss weight')
parser.add_argument('--noise_var', default=1e-7, type=float, help='training noise variance')  # 0.0003 1e-5
parser.add_argument('--batch_size', default=1, type=int, help='training batch size')
parser.add_argument('--max_epoch', default=600, type=int, help='maximum training iterations')
parser.add_argument('--miles', default=[100, 200, 300, 500], nargs='+', type=int,
                    help='learning rate scheduler milestones')
parser.add_argument('--gamma', default=3e-1, type=float, help='learning rate milestone decay')

# Save and plot options
parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
parser.add_argument('--output_dir_exp', default=r'C:\Users\AMB\Documents\PhD\code\Experiments\Foam2/', type=str,
                    help='output directory')
parser.add_argument('--plot_sim', default=True, type=str2bool, help='plot test simulation')
parser.add_argument('--experiment_name', default='exp2', type=str, help='experiment output name tensorboard')
args = parser.parse_args()  # Parse command-line arguments

device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

f = open(os.path.join(args.dset_dir, args.dset_name))
datasetInfo = json.load(f)

train_set = GraphDataset(datasetInfo,
                         os.path.join(args.dset_dir, datasetInfo['DatasetPaths']['train']))
test_set = GraphDataset(datasetInfo,
                        os.path.join(args.dset_dir, datasetInfo['DatasetPaths']['test']))
train_dataloader = DataLoader(train_set, batch_size=args.batch_size)
test_dataloader = DataLoader(test_set, batch_size=args.batch_size)

scaler = train_set.get_stats()

# Instantiate model
plasticity_gnn = PlasticityGNN(args, train_set.dims, scaler, datasetInfo['dt'])
plasticity_gnn.to(device)
# use model after training or load weights and drop into the production system

load_name = args.pretrain_weights
load_path = os.path.join(args.dset_dir, load_name)
checkpoint = torch.load(load_path, map_location='cuda')
plasticity_gnn.load_state_dict(checkpoint['state_dict'])
# plasticity_gnn.eval()

# Set Trainer
trainer = pl.Trainer(accelerator="gpu",
                     profiler="simple")

generate_results(plasticity_gnn, test_dataloader, datasetInfo, device,
                 os.path.join(args.output_dir_exp, args.experiment_name))

print()
