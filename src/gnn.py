"""model.py"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch_scatter import scatter_add, scatter_mean


# Multi Layer Perceptron (MLP) class
class MLP(torch.nn.Module):
    def __init__(self, layer_vec):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(layer_vec) - 1):
            layer = nn.Linear(layer_vec[k], layer_vec[k + 1])
            self.layers.append(layer)
            if k != len(layer_vec) - 2: self.layers.append(nn.SiLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Edge model
class EdgeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden, dims):
        super(EdgeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.edge_mlp = MLP([3 * self.dim_hidden + dims['g']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        if u is not None:
            out = torch.cat([edge_attr, src, dest, u[batch]], dim=1)
        else:
            out = torch.cat([edge_attr, src, dest], dim=1)
        out = self.edge_mlp(out)
        return out


# Node model
class NodeModel(torch.nn.Module):
    def __init__(self, n_hidden, dim_hidden, dims):
        super(NodeModel, self).__init__()
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.node_mlp = MLP(
            [2 * self.dim_hidden + dims['f'] + dims['g']] + self.n_hidden * [self.dim_hidden] + [self.dim_hidden])

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):
        src, dest = edge_index
        out = scatter_mean(edge_attr, dest, dim=0, dim_size=x.size(0))
        # out = torch.cat([out, scatter_add(edge_attr, dest, dim=0, dim_size=x.size(0))], dim=1)
        if f is not None:
            out = torch.cat([x, out, f], dim=1)
        elif u is not None:
            out = torch.cat([x, out, u[batch]], dim=1)
        else:
            out = torch.cat([x, out, ], dim=1)
        out = self.node_mlp(out)
        return out


# Modification of the original MetaLayer class
class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr, f=None, u=None, batch=None):
        src = edge_index[0]
        dest = edge_index[1]

        edge_attr = self.edge_model(x[src], x[dest], edge_attr, u,
                                    batch if batch is None else batch[src])
        x = self.node_model(x, edge_index, edge_attr, f, u, batch)

        return x, edge_attr


class PlasticityGNN(pl.LightningModule):
    def __init__(self, dims, scaler, dInfo, rollout_simulation=0, rollout_variable=None, rollout_freq=10):
        super().__init__()
        passes = dInfo['model']['passes']
        n_hidden = dInfo['model']['n_hidden']
        dim_hidden = dInfo['model']['dim_hidden']
        self.dims = dims
        self.dim_z = self.dims['z']
        self.dim_q = self.dims['q']
        dim_node = self.dims['z'] + self.dims['n'] - self.dims['q']
        dim_edge = self.dims['q'] + self.dims['q_0'] + 1
        self.state_variables = dInfo['dataset']['state_variables']

        # Encoder MLPs
        # self.encoder_node = MLP([dim_node] + [dim_hidden])
        self.encoder_node = MLP([dim_node] + n_hidden * [dim_hidden] + [dim_hidden])
        # self.encoder_edge = MLP([dim_edge] + [dim_hidden])
        self.encoder_edge = MLP([dim_edge] + n_hidden * [dim_hidden] + [dim_hidden])

        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(passes):
            node_model = NodeModel(n_hidden, dim_hidden, self.dims)
            edge_model = EdgeModel(n_hidden, dim_hidden, self.dims)
            GraphNet = \
                MetaLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)

        # self.processorNrm = nn.ModuleList()
        # for _ in range(passes):
        #     layer_norm = nn.LayerNorm(dim_hidden)
        #     self.processorNrm.append(layer_norm)
        # Decoder MLPs
        # self.decoder_E = MLP([dim_hidden] + n_hidden * [dim_hidden] + [1])
        self.decoder_E = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])
        # self.decoder_S = MLP([dim_hidden] + n_hidden * [dim_hidden] + [1])
        self.decoder_S = MLP([dim_hidden] + n_hidden * [dim_hidden] + [self.dim_z])

        self.decoder_L = MLP([dim_hidden] + n_hidden * [dim_hidden] + [
            int(self.dim_z * (self.dim_z + 1) / 2 - self.dim_z)])
        self.decoder_M = MLP(
            [dim_hidden] + n_hidden * [dim_hidden] + [int(self.dim_z * (self.dim_z + 1) / 2)])

        diag = torch.eye(self.dim_z, self.dim_z)
        self.diag = diag[None]
        self.ones = torch.ones(self.dim_z, self.dim_z)
        self.scaler = scaler
        self.dt = dInfo['dataset']['dt']
        self.noise_var = dInfo['model']['noise_var']
        self.lambda_d = dInfo['model']['lambda_d']
        self.lr = dInfo['model']['lr']
        self.miles = dInfo['model']['miles']
        self.gamma = dInfo['model']['gamma']

        # Rollout simulation
        self.rollout_simulation = rollout_simulation
        self.rollout_variable = rollout_variable
        self.rollout_freq = rollout_freq

    def integrator(self, L, M, dEdz, dSdz):
        # GENERIC time integration and degeneration
        dzdt = torch.bmm(L, dEdz) + torch.bmm(M, dSdz)
        deg_E = torch.bmm(M, dEdz)
        deg_S = torch.bmm(L, dSdz)

        return dzdt[:, :, 0], deg_E[:, :, 0], deg_S[:, :, 0]

    def training_step(self, batch, batch_idx, g=None):
        # Extract data from DataGeometric
        z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f

        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        z1_norm = torch.from_numpy(self.scaler.transform(z_t1.cpu())).float().to(self.device)

        noise = (self.noise_var) ** 0.5 * torch.randn_like(z_norm[n == 1])
        z_norm[n == 1] = z_norm[n == 1] + noise

        q = z_norm[:, :self.dim_q]
        v = z_norm[:, self.dim_q:]

        x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        # for GraphNet, nrmLayer in zip(self.processor, self.processorNrm):
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=f, u=g)  # TODO  arreglar el batch, batch=batch)
            x += x_res
            # x = nrmLayer(x)
            edge_attr += edge_attr_res

        '''Decode'''
        # Gradients
        dEdz = self.decoder_E(x)
        dSdz = self.decoder_S(x)
        # GENERIC flattened matrices
        l = self.decoder_L(x)
        m = self.decoder_M(x)
        #
        '''Reparametrization'''
        L = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=l.device)
        M = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=m.device)
        L[:, torch.tril(self.ones, -1) == 1] = l
        M[:, torch.tril(self.ones) == 1] = m
        # L skew-symmetric
        L = L - torch.transpose(L, 1, 2)
        # M symmetric and positive semi-definite
        M = torch.bmm(M, torch.transpose(M, 1, 2))

        dzdt_net, deg_E, deg_S = self.integrator(L, M, dEdz.unsqueeze(2), dSdz.unsqueeze(2))

        dzdt = (z1_norm - z_norm) / self.dt

        # Compute losses
        loss_z = torch.nn.functional.mse_loss(dzdt_net, dzdt)
        loss_deg_E = (deg_E ** 2).mean()
        loss_deg_S = (deg_S ** 2).mean()
        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)

        # loss = nn.functional.mse_loss(dzdt_net, dzdt)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)

        if self.state_variables is not None:
            for i, variable in enumerate(self.state_variables):
                loss_variable = nn.functional.mse_loss(dzdt_net[:, i], dzdt[:, i])
                self.log(f"train_loss_{variable}", loss_variable)

        return loss

    def validation_step(self, batch, batch_idx, g=None):
        # Extract data from DataGeometric
        z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f

        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        z1_norm = torch.from_numpy(self.scaler.transform(z_t1.cpu())).float().to(self.device)

        q = z_norm[:, :self.dim_q]
        v = z_norm[:, self.dim_q:]

        x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        # for GraphNet, nrmLayer in zip(self.processor, self.processorNrm):
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=f, u=g)  # TODO  arreglar el batch, batch=batch)
            x += x_res
            # x = nrmLayer(x)
            edge_attr += edge_attr_res

        '''Decode'''
        # Gradients
        dEdz = self.decoder_E(x)
        dSdz = self.decoder_S(x)
        # GENERIC flattened matrices
        l = self.decoder_L(x)
        m = self.decoder_M(x)
        #
        # '''Reparametrization'''
        L = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=l.device)
        M = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=m.device)
        L[:, torch.tril(self.ones, -1) == 1] = l
        M[:, torch.tril(self.ones) == 1] = m
        # L skew-symmetric
        L = L - torch.transpose(L, 1, 2)
        # M symmetric and positive semi-definite
        M = torch.bmm(M, torch.transpose(M, 1, 2))

        dzdt_net, deg_E, deg_S = self.integrator(L, M, dEdz.unsqueeze(2), dSdz.unsqueeze(2))

        dzdt = (z1_norm - z_norm) / self.dt

        # Compute losses
        loss_z = torch.nn.functional.mse_loss(dzdt_net, dzdt)
        loss_deg_E = (deg_E ** 2).mean()
        loss_deg_S = (deg_S ** 2).mean()
        loss = self.lambda_d * loss_z + (loss_deg_E + loss_deg_S)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=False)

        if self.state_variables is not None:
            for i, variable in enumerate(self.state_variables):
                loss_variable = nn.functional.mse_loss(dzdt_net[:, i], dzdt[:, i])
                self.log(f"val_loss_{variable}", loss_variable)

        # if (self.current_epoch % self.rollout_freq == 0) and (self.current_epoch > 0):
        #     if self.rollout_simulation in batch.idx:
        #         if len(self.rollouts_z_t1_pred) == 0:
        #             # Initial state
        #             self.rollouts_z_t1_pred.append(z_t0)
        #             self.rollouts_z_t1_gt.append(z_t0)
        #             self.rollouts_idx.append(batch.idx)
        #         # set only the predicted state variables
        #         z_t1_pred = torch.clone(dzdt)
        #         z_t1_pred[:, self.trainable_idx] = dzdt_net
        #         # append variables
        #         self.rollouts_z_t1_pred.append(z_t1_pred)
        #         self.rollouts_z_t1_gt.append(z_t1)
        #         self.rollouts_idx.append(batch.idx)

    def predict_step(self, batch, batch_idx, g=None):
        # Extract data from DataGeometric
        z_t0, z_t1, edge_index, n, f = batch.x, batch.y, batch.edge_index, batch.n, batch.f

        z_norm = torch.from_numpy(self.scaler.transform(z_t0.cpu())).float().to(self.device)
        z1_norm = torch.from_numpy(self.scaler.transform(z_t1.cpu())).float().to(self.device)

        q = z_norm[:, :self.dim_q]
        v = z_norm[:, self.dim_q:]

        x = torch.cat((v, torch.reshape(n.type(torch.float32), (len(n), 1))), dim=1)
        # Edge attributes
        src, dest = edge_index
        u = q[src] - q[dest]
        u_norm = torch.norm(u, dim=1).reshape(-1, 1)
        edge_attr = torch.cat((u, u_norm), dim=1)

        '''Encode'''
        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        '''Process'''
        # for GraphNet, nrmLayer in zip(self.processor, self.processorNrm):
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr, f=f, u=g)  # TODO  arreglar el batch, batch=batch)
            x += x_res
            # x = nrmLayer(x)
            edge_attr += edge_attr_res

        '''Decode'''
        # Gradients
        dEdz = self.decoder_E(x)
        dSdz = self.decoder_S(x)
        # GENERIC flattened matrices
        l = self.decoder_L(x)
        m = self.decoder_M(x)
        #
        # '''Reparametrization'''
        L = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=l.device)
        M = torch.zeros(x.size(0), self.dim_z, self.dim_z, device=m.device)
        L[:, torch.tril(self.ones, -1) == 1] = l
        M[:, torch.tril(self.ones) == 1] = m
        # L skew-symmetric
        L = L - torch.transpose(L, 1, 2)
        # M symmetric and positive semi-definite
        M = torch.bmm(M, torch.transpose(M, 1, 2))

        dzdt_net, deg_E, deg_S = self.integrator(L, M, dEdz.unsqueeze(2), dSdz.unsqueeze(2))

        z1_net = z_norm + self.dt * dzdt_net

        z1_net_denorm = torch.from_numpy(self.scaler.inverse_transform(z1_net.detach().to('cpu'))).float().to(
            self.device)

        return z1_net_denorm, z_t1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6, verbose=True),
        #                 'monitor': 'val_loss'}
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.miles, gamma=self.gamma),
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class RolloutCallback(pl.Callback):

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins."""
        pl_module.rollouts_z_t1_pred = []
        pl_module.rollouts_z_t1_gt = []
        pl_module.rollouts_idx = []

    def on_validation_epoch_end(self, trainer, pl_module):

        if len(pl_module.rollouts_z_t1_pred) > 0:
            # Concatenate al recorded data from batches
            rollouts_z_t1_pred = torch.cat(pl_module.rollouts_z_t1_pred)
            rollouts_z_t1_gt = torch.cat(pl_module.rollouts_z_t1_gt)
            rollouts_idx = torch.cat(pl_module.rollouts_idx)
            # Reshape data to have a snap per row
            rollouts_z_t1_pred = rollouts_z_t1_pred.reshape(len(rollouts_idx),
                                                            len(rollouts_z_t1_pred) // len(rollouts_idx),
                                                            rollouts_z_t1_pred.shape[-1])
            rollouts_z_t1_gt = rollouts_z_t1_gt.reshape(len(rollouts_idx), len(rollouts_z_t1_gt) // len(rollouts_idx),
                                                        rollouts_z_t1_gt.shape[-1])
            # Remove those snaps that don't belong to desired simulation
            indices_rollout_simulation = torch.where(rollouts_idx == pl_module.rollout_simulation)
            rollouts_z_t1_pred = rollouts_z_t1_pred[indices_rollout_simulation]
            rollouts_z_t1_gt = rollouts_z_t1_gt[indices_rollout_simulation]
            # Make video out of data pred and gt
            path_to_video = make_video_predicted_and_ground_truth(rollouts_z_t1_pred, rollouts_z_t1_gt,
                                                                  trainer.current_epoch,
                                                                  Path(trainer.checkpoint_callback.dirpath) / 'videos',
                                                                  'Rollout easy',
                                                                  plot_variable=pl_module.rollout_variable,
                                                                  state_variables=pl_module.state_variables)

            trainer.logger.experiment.log({"rollout easy": wandb.Video(path_to_video)})

        if (trainer.current_epoch % pl_module.rollout_freq == 0) and trainer.current_epoch != 0:
            from copy import deepcopy
            rollouts_z_t1_pred, rollouts_z_t1_gt = [], []
            # Build rollout ground truth
            for sample in trainer.val_dataloaders:
                if sample.idx == pl_module.rollout_simulation:
                    rollout_gt = [sample for sample in trainer.val_dataloaders if
                                  sample.idx == pl_module.rollout_simulation]
                if (len(rollout_gt) > 0) and sample.idx != pl_module.rollout_simulation:
                    break
            # Start rollout
            z_t0_pred, n = rollout_gt[0].x, rollout_gt[0].n
            contour_nodes = torch.where(n.squeeze() != 0)
            for sample_gt in rollout_gt:
                sample_t0 = deepcopy(sample_gt)
                # Contour conditions on state variables
                z_t0_pred[contour_nodes] = sample_gt.x[contour_nodes]
                sample_t0.x = z_t0_pred
                # Perform step rollout
                z_t1_pred = pl_module.predict_step(sample_t0, 0)
                z_t0_pred = torch.clone(z_t1_pred)

                rollouts_z_t1_pred.append(z_t1_pred)
                rollouts_z_t1_gt.append(sample_gt.x)

            # Make video out of data pred and gt
            path_to_video = make_video_predicted_and_ground_truth(rollouts_z_t1_pred, rollouts_z_t1_gt,
                                                                  trainer.current_epoch,
                                                                  Path(trainer.checkpoint_callback.dirpath) / 'videos',
                                                                  'Rollout Hard',
                                                                  plot_variable=pl_module.rollout_variable,
                                                                  state_variables=pl_module.state_variables)

            trainer.logger.experiment.log({"rollout hard": wandb.Video(path_to_video)})
