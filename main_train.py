import os
import json
import argparse
import datetime

from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.dataLoader.dataset import GraphDataset
from src.gnn import PlasticityGNN, RolloutCallback
from src.utils import str2bool

STATE_VARIABLES = ['Node Label', 'COORD.COOR1', 'COORD.COOR2', 'S.Mises', 'S.S11', 'S.S22', 'S.S33', 'S.S12']

if __name__ == '__main__':

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--sys_name', default='cylinder', type=str, help='physic system name')
    parser.add_argument('--train', default=True, type=str2bool, help='train or test')
    parser.add_argument('--pretrain', default=False, type=str2bool, help='starts the training from checkpoint')
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--start_epoch', default=0, type=int, help='start_epoch')
    parser.add_argument('--pretrain_weights', default='last.pt', type=str, help='name')

    # Dataset Parametersa
    parser.add_argument('--dset_dir', default='data', type=str, help='dataLoader directory')
    parser.add_argument('--dset_name', default=r'jsonFiles\dataset_1.json', type=str, help='dataLoader directory')

    # Net Parameters
    parser.add_argument('--n_hidden', default=2, type=int, help='number of hidden layers per MLP')
    parser.add_argument('--dim_hidden', default=124, type=int, help='dimension of hidden units')
    parser.add_argument('--passes', default=6, type=int, help='number of message passing')
    # Training Parameters
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--fixLr', default=True, type=str2bool, help='flag fig learning rate')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--lambda_d', default=10, type=float, help='data loss weight')
    parser.add_argument('--noise_var', default=0, type=float, help='training noise variance')  # 0.0003 1e-5
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

    f = open(os.path.join(args.dset_dir, args.dset_name))
    datasetInfo = json.load(f)

    train_set = GraphDataset(datasetInfo,
                             os.path.join(args.dset_dir, datasetInfo['DatasetPaths']['train']))
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size)

    scaler = train_set.get_stats()

    # Instantiate model
    plasticity_gnn = PlasticityGNN(args, train_set.dims, scaler, datasetInfo['dt'])
    print(plasticity_gnn)

    # Logger
    if args.train:
        val_set = GraphDataset(datasetInfo,
                               os.path.join(args.dset_dir, datasetInfo['DatasetPaths']['val']))
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size)

        name = f'train_enc2_hiddenDim{args.dim_hidden}_NumLayers{args.n_hidden}_Passes{args.passes}_lr{args.lr}_noise{args.noise_var}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        wandb_logger = WandbLogger(name=name, project='BeamGNNs')
        # Callbacks
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15, verbose=True, mode="min")
        checkpoint = ModelCheckpoint(dirpath=f'outputs/runs/{name}', monitor='val_loss', save_top_k=3)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # rollout = RolloutCallback()

        # Set Trainer
        trainer = pl.Trainer(accelerator="gpu",
                             logger=wandb_logger,
                             callbacks=[checkpoint, lr_monitor],  # , rollout],
                             profiler="simple",
                             num_sanity_val_steps=0,
                             max_epochs=args.max_epoch)
        # Train model
        trainer.fit(model=plasticity_gnn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    test_set = GraphDataset(datasetInfo,
                            os.path.join(args.dset_dir, datasetInfo['DatasetPaths']['test']))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size)

    generate_results(trainer, plasticity_gnn, test_dataloader, os.path.join(args.output_dir_exp, args.experiment_name))
