import os
import torch
import numpy as np
import lightning.pytorch as pl
from pathlib import Path

# from src.plots.plots import make_video_predicted_and_ground_truth
import wandb
from copy import deepcopy
import shutil
from src.utils import compute_connectivity
from src.plots import plot_2D

class HistogramPassesCallback(pl.Callback):

    # def on_train_epoch_end(self, trainer, pl_module):
    #
    #     if pl_module.current_epoch % 5 == 0:
    #
    #         columns = [f"pass_{i+1}" for i in range(pl_module.passes)]
    #         columns.append('epoch')
    #         table = wandb.Table(data=pl_module.error_message_passing, columns=columns)
    #         trainer.logger.experiment.log(
    #             {f'error_message_pass': table})

    def on_validation_end(self, trainer, pl_module):
        if pl_module.current_epoch % 5 == 0:
            table = wandb.Table(data=pl_module.error_message_pass, columns=["pass", "epoch", "error", "sim"])
            trainer.logger.experiment.log(
                {f'error_message_pass': table})


class RolloutCallback(pl.Callback):

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins."""
        pl_module.rollouts_z_t1_pred = []
        pl_module.rollouts_z_t1_gt = []
        pl_module.rollouts_idx = []

        folder_path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        if folder_path.exists():
            # Remove the folder and its contents
            shutil.rmtree(folder_path)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch > 0:
            if trainer.current_epoch%20 ==0 or trainer.current_epoch ==1:
                data = [sample for sample in trainer.val_dataloaders]

                dim_z = data[0].x.shape[1]
                N_nodes = data[0].x.shape[0]
                z_net = torch.zeros(len(data) + 1, N_nodes, dim_z)
                z_gt = torch.zeros(len(data) + 1, N_nodes, dim_z)

                # Initial conditions
                z_net[0] = data[0].x
                z_gt[0] = data[0].x

                z_denorm = data[0].x
                edge_index = data[0].edge_index

                # for sample in data:
                for t, snap in enumerate(data):
                    snap.x = z_denorm
                    snap.edge_index = edge_index
                    snap = snap.to(pl_module.device)
                    with torch.no_grad():
                        z_denorm, z_t1 = pl_module.predict_step(snap, 1)

                    pos = z_denorm[:, :3].clone()
                    pos[:, 2] = pos[:, 2] * 0
                    edge_index = compute_connectivity(np.asarray(pos.cpu()), pl_module.radius_connectivity,
                                                      add_self_edges=False).to(pl_module.device)
                    # edge_index = snap.edge_index

                    z_net[t + 1] = z_denorm
                    z_gt[t + 1] = z_t1
                save_dir = os.path.join(pl_module.save_folder, f'epoch_{trainer.current_epoch}.gif')
                plot_2D(z_net, z_gt, save_dir=save_dir, var=5)
                trainer.logger.experiment.log({"rollout": wandb.Video(save_dir, format='gif')})

    def on_train_end(self, trainer, pl_module):
        # Start Rollout Hard
        print('\nStart Hard Rollout!')
        rollouts_z_t1_pred, rollouts_z_t1_gt, rollout_gt = [], [], []
        # Build rollout ground truth
        for sample in trainer.val_dataloaders:
            if sample.idx == pl_module.rollout_simulation:
                rollout_gt = [sample for sample in trainer.val_dataloaders if
                              sample.idx == pl_module.rollout_simulation]
            if (len(rollout_gt) > 0) and sample.idx != pl_module.rollout_simulation:
                break
        # Start rollout
        z_t0_pred, n = rollout_gt[0].x, rollout_gt[0].n
        contour_nodes = torch.where(n.squeeze() != 0)[0].tolist()
        for sample_idx, sample_gt in enumerate(rollout_gt):
            sample_t0 = deepcopy(sample_gt)
            # Contour conditions on state variables
            z_t0_pred[contour_nodes] = torch.clone(sample_gt.x[contour_nodes])
            sample_t0.x = torch.clone(z_t0_pred)
            # Perform step rollout
            z_t1_pred = pl_module.predict_step(sample_t0, sample_idx)
            z_t0_pred = torch.clone(z_t1_pred)

            rollouts_z_t1_pred.append(z_t1_pred)
            rollouts_z_t1_gt.append(sample_gt.x)

        # Make video out of data pred and gt
        path_to_video_rollout_hard = make_video_predicted_and_ground_truth(rollouts_z_t1_pred, rollouts_z_t1_gt,
                                                                           trainer.current_epoch,
                                                                           Path(
                                                                               trainer.checkpoint_callback.dirpath) / 'videos',
                                                                           'RolloutHard',
                                                                           plot_variable=pl_module.rollout_variable,
                                                                           state_variables=pl_module.state_variables)

        trainer.logger.experiment.log({"rollout hard last": wandb.Video(path_to_video_rollout_hard, format='mp4')})