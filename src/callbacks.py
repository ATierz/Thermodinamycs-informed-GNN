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
from src.plots import plot_2D, plot_3D, plot_2D_image, plot_image3D, plt_LM
from src.evaluate import roll_out,compute_error, print_error, plotError

from lightning.pytorch.callbacks import LearningRateFinder
class HistogramPassesCallback(pl.Callback):

    def on_validation_end(self, trainer, pl_module):
        if (pl_module.current_epoch % 25 == 0 and len(pl_module.error_message_pass) > 0):
            table = wandb.Table(data=pl_module.error_message_pass, columns=["pass", "epoch", "error"])
            trainer.logger.experiment.log(
                {f'error_message_pass': table})


class RolloutCallback(pl.Callback):
    def __init__(self, dataloader, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = dataloader
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
            if trainer.current_epoch%pl_module.rollout_freq == 0 or trainer.current_epoch ==2:
                try:
                    z_net, z_gt, L, M = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.data_dim)
                    save_dir = os.path.join(pl_module.save_folder, f'epoch_{trainer.current_epoch}.gif')
                    # save_dir_L = os.path.join(pl_module.save_folder, f'epoch_L_{trainer.current_epoch}.png')
                    # save_dir_M = os.path.join(pl_module.save_folder, f'epoch_M_{trainer.current_epoch}.png')
                    # plt_LM(L, save_dir_L)
                    # plt_LM(M, save_dir_M)
                    # trainer.logger.experiment.log({"L": wandb.Image(save_dir_L)})
                    # trainer.logger.experiment.log({"M": wandb.Image(save_dir_M)})

                    if pl_module.data_dim == 2:
                        plot_2D(z_net, z_gt, save_dir=save_dir, var=5)
                    else:
                        plot_3D(z_net, z_gt, save_dir=save_dir, var=5)
                    trainer.logger.experiment.log({"rollout": wandb.Video(save_dir, format='gif')})
                except:
                    print()


    def on_train_end(self, trainer, pl_module):
        z_net, z_gt, L, M  = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.data_dim)
        filePath = os.path.join(pl_module.save_folder, 'metrics.txt')
        save_dir = os.path.join(pl_module.save_folder, f'final_{trainer.current_epoch}.gif')
        with open(filePath, 'w') as f:
            error, L2_list = compute_error(z_net, z_gt,pl_module.state_variables)
            lines = print_error(error)
            f.write('\n'.join(lines))
            print("[Test Evaluation Finished]\n")
            f.close()
        plotError(z_gt, z_net, L2_list, pl_module.state_variables, pl_module.data_dim, pl_module.save_folder)
        if pl_module.data_dim == 2:
            plot_2D(z_net, z_gt, save_dir=save_dir, var=5)
            plot_2D_image(z_net, z_gt, -1, 5)
        else:
            plot_3D(z_net, z_gt, save_dir=save_dir, var=5)
            data = [sample for sample in self.dataloader]
            plot_image3D(z_net, z_gt, pl_module.save_folder, var=5, step=-1, n=data[0].n)


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)