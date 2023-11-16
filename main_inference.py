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
parser.add_argument('--pretrain_weights', default=r'weights\epoch=82-step=216713.ckpt', type=str, help='name')

# Dataset Parametersa
parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
# parser.add_argument('--dset_name', default='d6_waterk10_noTensiones_radius_.pt', type=str, help='dataset directory')
parser.add_argument('--dset_name', default=r'jsonFiles\dataset_1.json', type=str, help='dataset directory')

# Save and plot options
parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
parser.add_argument('--output_dir_exp', default=r'C:\Users\AMB\Documents\PhD\code\Experiments\Foam2/', type=str,
                    help='output directory')
parser.add_argument('--plot_sim', default=True, type=str2bool, help='plot test simulation')
parser.add_argument('--experiment_name', default='exp3', type=str, help='experiment output name tensorboard')
args = parser.parse_args()  # Parse command-line arguments

device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

f = open(os.path.join(args.dset_dir, args.dset_name))
dInfo = json.load(f)

train_set = GraphDataset(dInfo,
                         os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['train']))
test_set = GraphDataset(dInfo,
                        os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['test']))
train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'])
test_dataloader = DataLoader(test_set, batch_size=dInfo['model']['batch_size'])

scaler = train_set.get_stats()

# Instantiate model
plasticity_gnn = PlasticityGNN(train_set.dims, scaler, dInfo)
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

generate_results(plasticity_gnn, test_dataloader, dInfo, device,
                 os.path.join(args.output_dir_exp, args.experiment_name), args.dset_name, args.pretrain_weights)

