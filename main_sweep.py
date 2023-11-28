# Import the W&B Python Library and log into W&B
"""main.py"""

import wandb

import os
import json
import argparse
import datetime
import torch

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.dataLoader.dataset import GraphDataset
from src.gnn import PlasticityGNN
from src.callbacks import RolloutCallback, FineTuneLearningRateFinder
from src.utils import str2bool


def main():

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--pretrain_weights', default=r'weights/epoch=82-step=216713.ckpt', type=str, help='name')

    # Dataset Parametersa
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    # parser.add_argument('--dset_name', default='d6_waterk10_noTensiones_radius_.pt', type=str, help='dataset directory')
    parser.add_argument('--dset_name', default=r'dataset_Beam3D.json', type=str, help='dataset directory')

    # Save and plot options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_dir_exp', default=r'/home/atierz/Documentos/code/Experiments/fase_2D/Foam/Foam2/', type=str,
                        help='output directory')
    parser.add_argument('--plot_sim', default=True, type=str2bool, help='plot test simulation')
    parser.add_argument('--experiment_name', default='exp3_resdiual', type=str, help='experiment output name tensorboard')
    args = parser.parse_args()  # Parse command-line arguments

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    f = open(os.path.join(args.dset_dir, 'jsonFiles', args.dset_name))
    dInfo = json.load(f)
    name = f"train_batchNorm_hiddenDim{dInfo['model']['dim_hidden']}_NumLayers{dInfo['model']['n_hidden']}_Passes{dInfo['model']['passes']}_lr{dInfo['model']['lr']}_noise{dInfo['model']['noise_var']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # name = f"prueba_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_folder = f'outputs/runs/{name}'
    wandb.init(project=dInfo['project_name'], tags=['first_sweep'], name=name)
    dInfo['model']['passes'] = wandb.config.passes
    dInfo['model']['dim_hidden'] = wandb.config.dim_hidden
    dInfo['model']['lambda_d'] = wandb.config.lambda_d
    dInfo['model']['dim_hidden'] = wandb.config.dim_hidden

    train_set = GraphDataset(dInfo,
                             os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['train']))
    train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'])

    scaler = train_set.get_stats()

    # Logger

    val_set = GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['val']))
    val_dataloader = DataLoader(val_set, batch_size=dInfo['model']['batch_size'])
    test_set = GraphDataset(dInfo, os.path.join(args.dset_dir, dInfo['dataset']['datasetPaths']['test']))
    test_dataloader = DataLoader(test_set, batch_size=1)

    wandb_logger = WandbLogger(name=name, project=dInfo['project_name'])

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15, verbose=True, mode="min")
    checkpoint = ModelCheckpoint(dirpath=save_folder,  filename='{epoch}-{val_loss:.2f}', monitor='val_loss', save_top_k=3)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rollout = RolloutCallback(test_dataloader)


    # Instantiate model
    plasticity_gnn = PlasticityGNN(train_set.dims, scaler, dInfo, save_folder)
    print(plasticity_gnn)

    # Set Trainer
    trainer = pl.Trainer(accelerator="gpu",
                         logger=wandb_logger,
                         callbacks=[checkpoint, lr_monitor, rollout],
                         # callbacks=[checkpoint, lr_monitor, FineTuneLearningRateFinder(milestones=(5, 10)), rollout],
                         profiler="simple",
                         num_sanity_val_steps=0,
                         max_epochs=dInfo['model']['max_epoch'])
    # Train model
    trainer.fit(model=plasticity_gnn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)




if __name__ == "__main__":

    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "loss_val"},
        "parameters": {
            "passes": {"values": [4, 6, 9, 12]},
            "dim_hidden": {"values": [50]},
            # "n_hidden": {"values": [2]},
            "lambda_d": {"values": [10]},
            "noise_var": {"values": [4e-5]},
        },
    }


    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Beam_3D')

    wandb.agent(sweep_id, function=main, count=4)

wandb.login()

