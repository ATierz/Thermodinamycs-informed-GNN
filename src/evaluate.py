import os
import torch
import numpy as np
from src.utils import print_error, compute_connectivity,generate_folder
from src.plots import plot_2D_image, plot_2D


def compute_error(z_net, z_gt, state_variables):
    # Compute error
    e = z_net.numpy() - z_gt.numpy()
    gt = z_gt.numpy()

    error = {clave: [] for clave in state_variables}

    for i, sv in enumerate(state_variables):
        L2 = ((e[1:, :, i] ** 2).sum(1) / (gt[1:, :, i] ** 2).sum(1)) ** 0.5
        error[sv].extend(list(L2))
    # plotError_2D(gt, z_net, L2_q, L2_v, L2_e, dEdt, dSdt, self.output_dir_exp)
    return error

def roll_out(plasticity_gnn, dataloader, device, radius_connectivity):
    data = [sample for sample in dataloader]

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
        snap = snap.to(device)
        with torch.no_grad():
            z_denorm, z_t1 = plasticity_gnn.predict_step(snap, 1)

        pos = z_denorm[:, :3].clone()
        pos[:, 2] = pos[:, 2] * 0
        edge_index = compute_connectivity(np.asarray(pos.cpu()), radius_connectivity, add_self_edges=False).to(device)
        # edge_index = snap.edge_index

        z_net[t + 1] = z_denorm
        z_gt[t + 1] = z_t1

    return z_net, z_gt
def generate_results(plasticity_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):

    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')

    # Make roll out
    z_net, z_gt = roll_out(plasticity_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'])

    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        error = compute_error(z_net, z_gt, dInfo['dataset']['state_variables'])
        lines = print_error(error)
        f.write('\n'.join(lines))
        print("[Test Evaluation Finished]\n")
        f.close()

    plot_2D_image(z_net, z_gt, -1, 5)

    plot_2D(z_net, z_gt, save_dir_gif, var=7)

    # output = trainer.predict(model=plasticity_gnn, dataloaders=test_dataloader)
    #
    # z_net = torch.zeros(len(output), output[0][0].shape[0], output[0][0].shape[1])
    # z_gt = torch.zeros(len(output), output[0][0].shape[0], output[0][0].shape[1])
    #
    # for i, out in enumerate(output):
    #     z_net[i] = out[0]
    #     z_gt[i] = out[1]
