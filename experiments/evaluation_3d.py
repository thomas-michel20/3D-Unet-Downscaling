import argparse
from collections import defaultdict
import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import shutil
import os
import pyvista as pv
# import our modules
import sys
sys.path.append("../src")
from unet3d import UNet3d
from implicit_net import ImNet
from pde import PDELayer
from nonlinearities import NONLINEARITIES
from local_implicit_grid import query_local_implicit_grid
import dataloader_spacetime as loader
from physics import get_rb2_pde_layer, get_3d_pde_layer
import torch.nn.functional as F
from torch_flow_stats import *
import pickle, os

def evaluate_feat_grid(pde_layer, latent_grid, z_seq, y_seq, x_seq, mins, maxs, pseudo_batch_size):
    """Evaluate latent feature grid at fixed intervals.

    Args:
        pde_layer: PDELayer instance where fwd_fn has been defined.
        latent_grid: latent feature grid of shape [batch, T, Z, X, C]
        t_seq: flat torch array of t-coordinates to evaluate
        z_seq: flat torch array of z-coordinates to evaluate
        x_seq: flat torch array of x-coordinates to evaluate
        mins: flat torch array of len 3 for min coords of t, z, x
        maxs: flat torch array of len 3 for max coords of t, z, x
        pseudo_batch_size, int, size of pseudo batch during eval
    Returns:
        res_dict: result dict.
    """
    device = latent_grid.device
    nb = latent_grid.shape[0]
    phys_channels = ["p", "u", "v", "w"]
    phys2id = dict(zip(phys_channels, range(len(phys_channels))))

    query_coord = torch.stack(torch.meshgrid(z_seq, y_seq, x_seq), axis=-1)  # [nz, ny, nx, 3]

    nz, ny, nx, _ = query_coord.shape
    query_coord = query_coord.reshape([-1, 3]).to(device)
    n_query  = query_coord.shape[0]

    res_dict = defaultdict(list)

    n_iters = int(np.ceil(n_query/pseudo_batch_size))

    for idx in range(n_iters):
        sid = idx * pseudo_batch_size
        eid = min(sid+pseudo_batch_size, n_query)
        query_coord_batch = query_coord[sid:eid]
        query_coord_batch = query_coord_batch[None].expand(*(nb, eid-sid, 3))  # [nb, eid-sid, 3]

        pred_value, residue_dict = pde_layer(query_coord_batch, return_residue=True)
        pred_value = pred_value.detach().cpu().numpy()
        for key in residue_dict.keys():
            residue_dict[key] = residue_dict[key].detach().cpu().numpy()
        for name, chan_id in zip(phys_channels, range(4)):
            res_dict[name].append(pred_value[..., chan_id]) 
        for name, val in residue_dict.items():
            res_dict[name].append(val[..., 0])   

    for key in res_dict.keys():
        res_dict[key] = (np.concatenate(res_dict[key], axis=1)
                         .reshape([nb, len(z_seq), len(y_seq), len(x_seq)]))[0]
    return res_dict


def frames_to_video(frames_pattern, save_video_to, frame_rate=10, keep_frames=False):
    """Create video from frames.

    frames_pattern: str, glob pattern of frames.
    save_video_to: str, path to save video to.
    keep_frames: bool, whether to keep frames after generating video.
    """
    cmd = (
    "ffmpeg -framerate {frame_rate} -pattern_type glob -i '{frames_pattern}' "
    "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
    "-c:v libx264 -r 30 -pix_fmt yuv420p {save_video_to}"
    .format(frame_rate=frame_rate, frames_pattern=frames_pattern,
            save_video_to=save_video_to)
)
    os.system(cmd)
    # print
    print("Saving videos to {}".format(save_video_to))
    # delete frames if keep_frames is not needed
    if not keep_frames:
        frames_dir = os.path.dirname(frames_pattern)
        shutil.rmtree(frames_dir)


def calculate_flow_stats(pred, hres, args, visc=0.0001):
    data = pred
    uw = np.transpose(data[2:4,:,:,1:1+args.eval_zres], (1, 0, 2, 3))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uw = torch.tensor(uw, device=device).float()
    stats = compute_all_stats(uw[2:,:,:,:], viscosity=visc, description=False)
    s = [stats[..., i].item() for i in range(stats.shape[0])]

    file = open("REPORT___FlowStats_Pred_vs_GroundTruth.txt", "w")

    file.write("***** Pred Data Flow Statistics ******\n")
    file.write("Total Kinetic Energy     : {}\n".format(s[0]))
    file.write("Dissipation              : {}\n".format(s[1]))
    file.write("Rms velocity             : {}\n".format(s[2]))
    file.write("Taylor Micro. Scale      : {}\n".format(s[3]))
    file.write("Taylor-scale Reynolds    : {}\n".format(s[4]))
    file.write("Kolmogorov time sclae    : {}\n".format(s[5]))
    file.write("Kolmogorov length sclae  : {}\n".format(s[6]))
    file.write("Integral scale           : {}\n".format(s[7]))
    file.write("Large eddy turnover time : {}\n\n\n\n\n".format(s[8]))

    data = hres
    uw = np.transpose(data[2:4,:,:,1:1+args.eval_zres], (1, 0, 2, 3))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    uw = torch.tensor(uw, device=device).float()
    stats = compute_all_stats(uw[2:,:,:,:], viscosity=visc, description=False)
    s = [stats[..., i].item() for i in range(stats.shape[0])]

    file.write("***** Ground Truth Data Flow Statistics ******\n")
    file.write("Total Kinetic Energy     : {}\n".format(s[0]))
    file.write("Dissipation              : {}\n".format(s[1]))
    file.write("Rms velocity             : {}\n".format(s[2]))
    file.write("Taylor Micro. Scale      : {}\n".format(s[3]))
    file.write("Taylor-scale Reynolds    : {}\n".format(s[4]))
    file.write("Kolmogorov time sclae    : {}\n".format(s[5]))
    file.write("Kolmogorov length sclae  : {}\n".format(s[6]))
    file.write("Integral scale           : {}\n".format(s[7]))
    file.write("Large eddy turnover time : {}\n".format(s[8]))

# ---------- Metric helpers ----------
def compute_error_metrics(pred, true):
    """Return MAE, NMAE, NMSE"""
    mae = torch.mean(torch.abs(pred - true))
    nmae = torch.sum(torch.abs(pred - true)) / (torch.sum(torch.abs(true)) + 1e-12)
    nmse = torch.sum((pred - true)**2) / (torch.sum(true**2) + 1e-12)
    return mae.item(), nmae.item(), nmse.item()

def mae_pressure_velocity(pred, true):
    """Compute per-channel MAE for pressure and mean velocity"""
    mae_p = torch.mean(torch.abs(pred[0] - true[0]))
    mae_v = torch.mean(torch.abs(pred[1:] - true[1:]))
    return mae_p.item(), mae_v.item()

# ---------- Physics stats ----------
def compute_3d_stats(vel, viscosity=0.000185):
    assert vel.ndim == 4 and vel.shape[0] == 3
    if not torch.is_tensor(vel): vel = torch.from_numpy(vel)
    vel = vel.float()

    ke = 0.5 * torch.mean(torch.sum(vel**2, dim=0))
    rms_v = torch.sqrt(torch.mean(torch.sum(vel**2, dim=0)))

    def grad3d(f): return [(torch.roll(f, -1, d) - torch.roll(f, 1, d)) / 2 for d in range(3)]
    grads = [grad3d(vel[i]) for i in range(3)]
    SijSij = 0.0
    for i in range(3):
        for j in range(3):
            Sij = 0.5 * (grads[i][j] + grads[j][i])
            SijSij += Sij**2
    diss = 2 * viscosity * torch.mean(SijSij)
    taylor = torch.sqrt(15 * viscosity * rms_v**2 / (diss + 1e-12))
    Re_lambda = rms_v * taylor / viscosity

    def grad(f, dim):
        return (torch.roll(f, -1, dims=dim) - torch.roll(f, 1, dims=dim)) / 2.0

    div = grad(vel[0], 0) + grad(vel[1], 1) + grad(vel[2], 2)

    return ke.item(), rms_v.item(), torch.mean(torch.abs(div)).item()

# ---------- Evaluation per sample ----------
def evaluate_sample(hres, lres, res_dict, dataset, args):
    """Evaluate one sample (pred vs interp vs ground truth)"""
    phys_channels = ["p", "u", "v", "w"]
    if dataset:
        lres = dataset.denormalize_grid(lres.copy())
        pred = np.stack([res_dict[k] for k in phys_channels], axis=0)
        pred = dataset.denormalize_grid(pred)
        hres = dataset.denormalize_grid(hres)
    else:
        pred = np.stack([res_dict[k] for k in phys_channels], axis=0)

    hres_t = torch.tensor(hres)
    pred_t = torch.tensor(pred)

    # --- model errors
    metrics = {}
    for i, name in enumerate(["p", "u", "v", "w"]):
        mae, nmae, nmse = compute_error_metrics(pred_t[i], hres_t[i])
        metrics[f"MAE_{name}"] = mae
        metrics[f"NMAE_{name}"] = nmae
        metrics[f"NMSE_{name}"] = nmse

    ke_pred, rms_pred, div_pred = compute_3d_stats(pred_t[1:])
    ke_true, rms_true, div_true = compute_3d_stats(hres_t[1:])
    metrics["Err_KineticEnergy"] = 100 * abs(ke_pred - ke_true) / (ke_true + 1e-12)
    metrics["Err_RMS_Velocity"] = 100 * abs(rms_pred - rms_true) / (rms_true + 1e-12)
    metrics["Div_pred"] = div_pred
    metrics["Div_true"] = div_true

    # --- interpolation baseline
    hr_interp = interpolate_3d(lres, scale_factor=args.downsamp_xyz, mode="trilinear")
    interp_t = torch.tensor(hr_interp)
    for i, name in enumerate(["p", "u", "v", "w"]):
        mae, nmae, nmse = compute_error_metrics(interp_t[i], hres_t[i])
        metrics[f"MAE_{name}_interp"] = mae
        metrics[f"NMAE_{name}_interp"] = nmae
        metrics[f"NMSE_{name}_interp"] = nmse

    ke_interp, rms_interp, div_interp = compute_3d_stats(interp_t[1:])
    metrics["Err_KineticEnergy_interp"] = 100 * abs(ke_interp - ke_true) / (ke_true + 1e-12)
    metrics["Err_RMS_Velocity_interp"] = 100 * abs(rms_interp - rms_true) / (rms_true + 1e-12)
    metrics["Div_interp"] = div_interp

    return metrics

# ---------- Main validation loop ----------
def run_evaluation(dataset, nbr_val_samples, args):
    len_dataset = len(dataset)
    val_idx = np.random.choice(len_dataset, size=min(nbr_val_samples, len_dataset), replace=False)

    all_metrics = []

    for i, idx in enumerate(val_idx):
        hres, lres, _, _ = dataset[idx]
        if args.normalize_channels:
            mean, std = dataset.channel_mean, dataset.channel_std
        else:
            mean = std = None
        pde_layer = get_3d_pde_layer(mean=mean, std=std)
        res_dict = model_inference(args, lres, pde_layer)
        all_metrics.append(evaluate_sample(hres, lres, res_dict, dataset, args))

    # average results
    keys = all_metrics[0].keys()
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in keys}

    # save report
    os.makedirs(args.save_path, exist_ok=True)
    report_path = os.path.join(args.save_path, "Evaluation_Metrics.txt")
    with open(report_path, "w") as f:
        f.write(f"***** Averages over {len(val_idx)} validation samples *****\n\n")

        f.write("== Model vs Ground Truth ==\n")
        for name in ["p", "u", "v", "w"]:
            f.write(f"{name.upper()} - MAE: {avg[f'MAE_{name}']:.6f}, NMAE: {avg[f'NMAE_{name}']:.6f}, NMSE: {avg[f'NMSE_{name}']:.6f}\n")
        f.write(f"Error Kinetic Energy: {avg['Err_KineticEnergy']:.2f}%\n")
        f.write(f"Error RMS Velocity:   {avg['Err_RMS_Velocity']:.2f}%\n\n")

        f.write("== Interpolation Baseline ==\n")
        for name in ["p", "u", "v", "w"]:
            f.write(f"{name.upper()} - MAE: {avg[f'MAE_{name}_interp']:.6f}, NMAE: {avg[f'NMAE_{name}_interp']:.6f}, NMSE: {avg[f'NMSE_{name}_interp']:.6f}\n")
        f.write(f"Error Kinetic Energy: {avg['Err_KineticEnergy_interp']:.2f}%\n")
        f.write(f"Error RMS Velocity:   {avg['Err_RMS_Velocity_interp']:.2f}%\n\n")
        f.write("== Mean Absolute Divergence ==\n")
        f.write(f"Prediction div(u):        {avg['Div_pred']:.4e}\n")
        f.write(f"Truth div(u):        {avg['Div_true']:.4e}\n")
        f.write(f"Interpolation div(u):        {avg['Div_interp']:.4e}\n")

    print(f"✅ Saved evaluation report to {report_path}")


def interpolate_3d(lr_tensor, scale_factor=None, target_shape=None, mode="trilinear", align_corners=False):
    """
    Interpolates a 3D tensor [C, Z, Y, X] to higher resolution.

    Args:
        lr_tensor (torch.Tensor): Input tensor of shape [C, Z, Y, X].
        scale_factor (float or tuple): Upscaling factor, e.g. 4 or (4,4,4).
        target_shape (tuple): Explicit target shape (Z_high, Y_high, X_high).
        mode (str): Interpolation type ("trilinear", "nearest").
        align_corners (bool): Whether to align corners for 'trilinear'.

    Returns:
        torch.Tensor: Interpolated high-resolution tensor [C, Z_high, Y_high, X_high].
    """
    if not torch.is_tensor(lr_tensor):
        lr_tensor = torch.from_numpy(lr_tensor)
    lr_tensor = lr_tensor.float()
    assert lr_tensor.ndim == 4, "Input must be [C, Z, Y, X]"
    lr_tensor = lr_tensor.unsqueeze(0)  # add batch dim [1, C, Z, Y, X]

    if target_shape is not None:
        hr = F.interpolate(lr_tensor, size=target_shape, mode=mode, align_corners=align_corners)
    else:
        hr = F.interpolate(lr_tensor, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    return hr.squeeze(0)  # back to [C, Z, Y, X]

def plot_prediction(hres, pred, lres, phys_channel, save_path, args):

    # --- High-Res ---
    nx1, ny1, nz1 = hres.shape
    grid1 = pv.ImageData()
    grid1.dimensions = hres.shape
    grid1.spacing = (1/nx1, 1/ny1, 1/nz1)
    grid1.point_data["u"] = hres.flatten(order="F")

    # --- Prediction ---
    grid2 = pv.ImageData()
    grid2.dimensions = pred.shape
    grid2.spacing = (1/nx1, 1/ny1, 1/nz1)
    grid2.point_data["u"] = pred.flatten(order="F")

    # --- Low-Res ---
    grid3 = pv.ImageData()

    grid3.dimensions = lres.shape
    grid3.spacing = ((args.downsamp_xyz)/nx1, (args.downsamp_xyz)/ny1, (args.downsamp_xyz)/nz1)
    grid3.point_data["u"] = lres.flatten(order="F")

    # --- Shared color limits ---
    vmin = min(hres.min(), lres.min(), pred.min())
    vmax = max(hres.max(), lres.max(), pred.max())

    # --- Plot side-by-side ---
    p = pv.Plotter(shape=(1, 3), border=False)

    # High-Res
    p.subplot(0, 1)
    p.add_volume(grid1, cmap="viridis", clim=[vmin, vmax], opacity="linear", show_scalar_bar=False)
    p.add_text("High-resolution", font_size=14)
    p.add_scalar_bar(title=phys_channel, position_x=0.2, position_y=0.05)

    # Pred
    p.subplot(0, 2)
    p.add_volume(grid2, cmap="viridis", clim=[vmin, vmax], opacity="linear", show_scalar_bar=False)
    p.add_text("Prediction", font_size=14)
    

    # Low-Res
    p.subplot(0, 0)
    p.add_volume(grid3, cmap="viridis", clim=[vmin, vmax], opacity="linear", show_scalar_bar=False)
    p.add_text("Low-Resolution", font_size=14)

    #Save
    p.link_views()
    p.show()
    p.screenshot(f"{save_path}/prediction_3d_{phys_channel}.png")

def visualize_3d(args, res_dict, hres, lres, dataset):
    phys_channels = ["p", "u", "v", "w"]
    if dataset:
        hres = dataset.denormalize_grid(hres.copy())
        lres = dataset.denormalize_grid(lres.copy())
        pred = np.stack([res_dict[key] for key in phys_channels], axis=0)
        pred = dataset.denormalize_grid(pred)
    print(f'Hres shape {hres.shape}')
    print(f'Lres shape {lres.shape}')
    print(f'Pred shape {pred.shape}')
    for idx, name in enumerate(phys_channels):
        hres_frames = hres[idx]
        lres_frames = lres[idx]
        pred_frames = pred[idx]
        plot_prediction(hres_frames, pred_frames, lres_frames, name, args.save_path, args)

def export_video(args, res_dict, hres, lres, dataset):
    """Export inference result as a video.
    """
    phys_channels = ["p", "u", "v", "w"]
    if dataset:
        hres = dataset.denormalize_grid(hres.copy())
        lres = dataset.denormalize_grid(lres.copy())
        pred = np.stack([res_dict[key] for key in phys_channels], axis=0)
        pred = dataset.denormalize_grid(pred)
        #calculate_flow_stats(pred, hres, args)       # Warning: only works with pytorch > v1.3 and CUDA >= v10.1
        #np.savez_compressed(args.save_path+'highres_lowres_pred', hres=lres, lres=lres, pred=pred)

    os.makedirs(args.save_path, exist_ok=True)
    # enumerate through physical channels first

    for idx, name in enumerate(phys_channels):
        frames_dir = os.path.join(args.save_path, f'frames_{name}')
        os.makedirs(frames_dir, exist_ok=True)
        hres_frames = hres[idx]
        lres_frames = lres[idx]
        pred_frames = pred[idx]

        # loop over each timestep in pred_frames
        max_val = np.max(hres_frames)
        min_val = np.min(hres_frames)

        for pid in range(pred_frames.shape[0]):
            hid = int(np.round(pid / (pred_frames.shape[0] - 1) * (hres_frames.shape[0] - 1)))
            lid = int(np.round(pid / (pred_frames.shape[0] - 1) * (lres_frames.shape[0] - 1)))

            fig, axes = plt.subplots(3, figsize=(10, 10))#, 1, sharex=True)
            # high res ground truth
            im0 = axes[0].imshow(hres_frames[hid], cmap='RdBu',interpolation='none')
            axes[0].set_title(f'{name} channel, high res ground truth.')
            im0.set_clim(min_val, max_val)
            # low res input
            im1 = axes[1].imshow(lres_frames[lid], cmap='RdBu',interpolation='none')
            axes[1].set_title(f'{name} channel, low  res ground truth.')
            im1.set_clim(min_val, max_val)
            # prediction
            im2 = axes[2].imshow(pred_frames[pid], cmap='RdBu',interpolation='none')
            axes[2].set_title(f'{name} channel, predicted values.')
            im2.set_clim(min_val, max_val)
            # add shared colorbar
            cbaxes = fig.add_axes([0.1, 0, .82, 0.05])
            cb = fig.colorbar(im2, orientation="horizontal", pad=0, cax=cbaxes)
            ticks = np.linspace(min_val, max_val, 5)
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{t:.2f}" for t in ticks])
            frame_name = 'frame_{:03d}.png'.format(pid)
            fig.savefig(os.path.join(frames_dir, frame_name), bbox_inches='tight')

        # stitch frames into video (using ffmpeg)
        frames_to_video(
            frames_pattern=os.path.join(frames_dir, "*.png"),
            save_video_to=os.path.join(args.save_path, f"video_{name}.mp4"),
            frame_rate=args.frame_rate, keep_frames=args.keep_frames)

def model_inference(args, lres, pde_layer):
    # select inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # construct model
    print(f"Loading model parameters from {args.ckpt}...")
    igres = (int(args.nz/args.downsamp_xyz),
             int(args.ny/args.downsamp_xyz),
             int(args.nx/args.downsamp_xyz),)
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=igres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf,
                  activation=NONLINEARITIES[args.nonlin])

    # load model params

    resume_dict = torch.load(args.ckpt, weights_only=False)
    unet.load_state_dict(resume_dict["unet_state_dict"])
    imnet.load_state_dict(resume_dict["imnet_state_dict"])

    unet.to(device)
    imnet.to(device)
    unet.eval()
    imnet.eval()
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    # evaluate
    latent_grid = unet(torch.tensor(lres, dtype=torch.float32)[None].to(device))
    latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, Z, Y, X, C]

    # create evaluation grid
    z_max = 1
    y_max = 1
    x_max = 1

    # layout query points for the desired slices
    eps = 1e-6
    z_seq = torch.linspace(eps, z_max-eps, args.eval_zres)  # z sequences
    y_seq = torch.linspace(eps, y_max-eps, args.eval_yres)  # y sequences
    x_seq = torch.linspace(eps, x_max-eps, args.eval_xres)  # x sequences

    mins = torch.zeros(3, dtype=torch.float32, device=device)
    maxs = torch.tensor([z_max, y_max, x_max], dtype=torch.float32, device=device)

    # define lambda function for pde_layer
    fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, mins, maxs)

    # update pde layer and compute predicted values + pde residues
    pde_layer.update_forward_method(fwd_fn)

    res_dict = evaluate_feat_grid(pde_layer, latent_grid, z_seq, y_seq, x_seq, mins, maxs,
                                  args.eval_pseudo_batch_size)

    return res_dict



def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--eval_xres", type=int, default=512, metavar="X",
                        help="x resolution during evaluation (default: 512)")
    parser.add_argument("--eval_zres", type=int, default=128, metavar="Z",
                        help="z resolution during evaluation (default: 128)")
    parser.add_argument("--eval_yres", type=int, default=192, metavar="T",
                        help="y resolution during evaluation (default: 192)")
    parser.add_argument("--eval_downsamp_xyz", default=4, type=int, required=True,
                        help="down sampling factor in x, y and z for low resolution crop.")
    parser.add_argument('--ckpt', type=str, required=True, help="path to checkpoint")
    parser.add_argument("--save_path", type=str, default='eval')
    parser.add_argument("--data_folder_training", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--data_folder_evaluation", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--train_data", type=str, default="rb2d_ra1e6_s42.npz",
                        help="name of training data (default: rb2d_ra1e6_s42.npz)")
    parser.add_argument("--eval_data", type=str, default="rb2d_ra1e6_s42.npz",
                        help="name of training data (default: rb2d_ra1e6_s42.npz)")
    parser.add_argument("--lres_interp", type=str, default='linear',
                        help="str, interpolation scheme for generating low res. choices of 'linear', 'nearest'")
    parser.add_argument("--lres_filter", type=str, default='none',
                        help=" str, filter to apply on original high-res image before \
                        interpolation. choices of 'none', 'gaussian', 'uniform', 'median', 'maximum'")
    parser.add_argument("--frame_rate", type=int, default=10, metavar="N",
                        help="frame rate for output video (default: 10)")
    parser.add_argument("--keep_frames", dest='keep_frames', action='store_true')
    parser.add_argument("--save_video", dest='save_video', action='store_true')
    parser.add_argument("--nbr_val_samples", type=int, default=30, metavar="T",
                        help="Number of samples used for caluclating the validation metrics")
    parser.add_argument("--no_keep_frames", dest='keep_frames', action='store_false')
    parser.add_argument("--eval_pseudo_batch_size", type=int, default=10000,
                        help="psudo batch size for querying the grid. set to a smaller"
                             " value if OOM error occurs")
    parser.set_defaults(keep_frames=False)
    parser.set_defaults(save_video=False)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    param_file = os.path.join(os.path.dirname(args.ckpt), "params.json")
    with open(param_file, 'r') as fh:
        args.__dict__.update(json.load(fh))

    print(args)
    # prepare dataset
    trainset = loader.Spatial3D_DataLoader(
        data_dir=args.data_folder_training, data_filename=args.train_data,
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xyz=args.downsamp_xyz,
        normalize_output=args.normalize_channels, normalize_hres=True, return_hres=args.normalize_channels,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp
    )

    
    dataset = loader.Spatial3D_DataLoader(
        data_dir=args.data_folder_evaluation, data_filename=args.eval_data,
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xyz=args.downsamp_xyz,
        normalize_output=args.normalize_channels, normalize_hres=True, return_hres=args.normalize_channels,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp, trainset_mean=trainset._mean, trainset_std=trainset._std
    )
    len_dataset = len(dataset)
    print(f'Length of dataset = {len_dataset}')
    nbr_val_samples = args.nbr_val_samples
    if nbr_val_samples > len_dataset:
        print('!! More samples required than the size of the validation dataset !!')
    run_evaluation(dataset, nbr_val_samples, args)
    
    # save video
    save_video = args.save_video
    if save_video:
        val_idx = 0 #np.random.choice(len_dataset)
        hres, lres, point_coord, point_value = dataset[val_idx]
        if args.normalize_channels:
            mean, std = dataset.channel_mean, dataset.channel_std
        else:
            mean = std = None
        pde_layer = get_3d_pde_layer(mean=mean, std=std)
        res_dict = model_inference(args, lres, pde_layer)
        export_video(args, res_dict, hres, lres, dataset)
        visualize_3d(args, res_dict, hres, lres, dataset)

if __name__ == '__main__':
    main()
