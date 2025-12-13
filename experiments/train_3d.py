"""Training script for RB2 experiment.
"""
import argparse
import json
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
np.set_printoptions(precision=4)

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# import our modules
import sys
sys.path.append("../../src")
import train_utils as utils
from unet3d import UNet3d
from implicit_net import ImNet
from local_implicit_grid import query_local_implicit_grid
from nonlinearities import NONLINEARITIES
import dataloader_spacetime as loader
from physics import get_rb2_pde_layer, get_3d_pde_layer

# pylint: disable=no-member


def loss_functional(loss_type):
    """Get loss function given function type names."""
    if loss_type == 'l1':
        return F.l1_loss
    if loss_type == 'l2':
        return F.mse_loss
    # else (loss_type == 'huber')
    return F.smooth_l1_loss


def train(args, unet, imnet, train_loader, epoch, global_step, device,
          logger, writer, optimizer, pde_layer):
    """Training function."""
    unet.train()
    imnet.train()
    tot_loss = 0
    count = 0
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)
    loss_func = loss_functional(args.reg_loss_type)
    for batch_idx, data_tensors in enumerate(train_loader):
        # send tensors to device
        
        data_tensors = [t.to(device) for t in data_tensors]
        _, input_grid, point_coord, point_value = data_tensors
        optimizer.zero_grad()
        latent_grid = unet(input_grid)  # [batch, N, C, T, X, Y]
        # permute such that C is the last channel for local implicit grid query
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, N, T, X, Y, C]

        # define lambda function for pde_layer
        fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(fwd_fn)
        pred_value, residue_dict = pde_layer(point_coord, return_residue=True)

        # function value regression loss
        reg_loss = loss_func(pred_value, point_value)

        # pde residue loss
        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)
        pde_loss = loss_func(pde_tensors, torch.zeros_like(pde_tensors))
        loss = args.alpha_reg * reg_loss + args.alpha_pde * pde_loss

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_value_(unet.module.parameters(), args.clip_grad)
        torch.nn.utils.clip_grad_value_(imnet.module.parameters(), args.clip_grad)

        optimizer.step()
        tot_loss += loss.item()
        count += 1
        if batch_idx % args.log_interval == 0:
            # logger log
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss Sum: {:.6f}\t"
                "Loss Reg: {:.6f}\tLoss Pde: {:.6f}".format(
                    epoch, batch_idx * len(input_grid), len(train_loader) * len(input_grid),
                    100. * batch_idx / len(train_loader), loss.item(),
                    args.alpha_reg * reg_loss, args.alpha_pde * pde_loss))
            # tensorboard log
            writer.add_scalar('train/reg_loss_unweighted', reg_loss, global_step=int(global_step))
            writer.add_scalar('train/pde_loss_unweighted', pde_loss, global_step=int(global_step))
            writer.add_scalar('train/sum_loss', loss, global_step=int(global_step))
            writer.add_scalars('train/losses_weighted',
                               {"reg_loss": args.alpha_reg * reg_loss,
                                "pde_loss": args.alpha_pde * pde_loss,
                                "sum_loss": loss}, global_step=int(global_step))

        global_step += 1
    tot_loss /= count
    return tot_loss

def eval(args, unet, imnet, eval_loader, epoch, global_step, device,
          logger, writer, optimizer, pde_layer):
    """Training function."""
    unet.eval()
    imnet.eval()
    tot_loss = 0
    count = 0
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)
    loss_func = loss_functional(args.reg_loss_type)
    for batch_idx, data_tensors in enumerate(eval_loader):
        # send tensors to device
        
        data_tensors = [t.to(device) for t in data_tensors]
        _, input_grid, point_coord, point_value = data_tensors

        latent_grid = unet(input_grid)  # [batch, N, C, T, X, Y]
        # permute such that C is the last channel for local implicit grid query
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, N, T, X, Y, C]

        # define lambda function for pde_layer
        fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # update pde layer and compute predicted values + pde residues
        pde_layer.update_forward_method(fwd_fn)
        pred_value, residue_dict = pde_layer(point_coord, return_residue=True)

        # function value regression loss
        reg_loss = loss_func(pred_value, point_value)

        # pde residue loss
        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)
        pde_loss = loss_func(pde_tensors, torch.zeros_like(pde_tensors))
        loss = args.alpha_reg * reg_loss + args.alpha_pde * pde_loss

        tot_loss += loss.item()
        count += 1
        global_step += 1
        break # Only evaluate on 1 batch
    tot_loss /= count
    logger.info(f'Validation loss is = {tot_loss}')
    return tot_loss

def plot_losses(training_losses, evaluation_losses, epoch, save_path, logger, show=False):
    # Basic validation
    if len(training_losses) != epoch or len(evaluation_losses) != epoch:
        raise ValueError("Loss lists must have length equal to 'epoch'.")

    epoch_vec = range(epoch)

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_vec, training_losses, label="Training Loss", linewidth=2)
    plt.plot(epoch_vec, evaluation_losses, label="Evaluation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path + '/Training_VS_Validation_Loss.png')
    logger.info(f'Saved Training_VS_Validation_Loss.png to {save_path}')
    if show:
        plt.show()

    plt.close()

def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument("--batch_size_per_gpu", type=int, default=10, metavar="N",
                        help="input batch size for training (default: 10)")
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--pseudo_epoch_size", type=int, default=3000, metavar="N",
                        help="number of samples in an pseudo-epoch. (default: 3000)")
    parser.add_argument("--lr", type=float, default=1e-2, metavar="R",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--data_folder_training", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--data_folder_evaluation", type=str, default="./data",
                        help="path to data folder (default: ./data)")
    parser.add_argument("--train_data", type=str, default="rb2d_ra1e6_s42.npz",
                        help="name of training data (default: rb2d_ra1e6_s42.npz)")
    parser.add_argument("--eval_data", type=str, default="rb2d_ra1e6_s42.npz",
                        help="name of training data (default: rb2d_ra1e6_s42.npz)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log_dir", type=str, required=True, help="log directory for run")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint if resume is needed")
    parser.add_argument("--ny", default=128, type=int, help="resolution of high res crop in y.")
    parser.add_argument("--nx", default=128, type=int, help="resolution of high res crop in x.")
    parser.add_argument("--nz", default=128, type=int, help="resolution of high res crop in z.")
    parser.add_argument("--downsamp_xyz", default=8, type=int,
                        help="down sampling factor in x and z for low resolution crop.")
    parser.add_argument("--n_samp_pts_per_crop", default=1024, type=int,
                        help="number of sample points to draw per crop.")
    parser.add_argument("--lat_dims", default=32, type=int, help="number of latent dimensions.")
    parser.add_argument("--unet_nf", default=16, type=int,
                        help="number of base number of feature layers in unet.")
    parser.add_argument("--unet_mf", default=256, type=int,
                        help="a cap for max number of feature layers throughout the unet.")
    parser.add_argument("--imnet_nf", default=32, type=int,
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--reg_loss_type", default="l1", type=str,
                        choices=["l1", "l2", "huber"],
                        help="number of base number of feature layers in implicit network.")
    parser.add_argument("--alpha_reg", default=1., type=float, help="weight of regression loss.")
    parser.add_argument("--alpha_pde", default=1., type=float, help="weight of pde residue loss.")
    parser.add_argument("--num_log_images", default=2, type=int, help="number of images to log.")
    parser.add_argument("--pseudo_batch_size", default=1024, type=int,
                        help="size of pseudo batch during eval.")
    parser.add_argument("--normalize_channels", dest='normalize_channels', action='store_true')
    parser.set_defaults(normalize_channels=True)
    parser.add_argument("--no_normalize_channels", dest='normalize_channels', action='store_false')
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', action='store_true')
    parser.add_argument("--no_lr_scheduler", dest='lr_scheduler', action='store_false')
    parser.set_defaults(lr_scheduler=True)
    parser.add_argument("--clip_grad", default=1., type=float,
                        help="clip gradient to this value. large value basically deactivates it.")
    parser.add_argument("--lres_filter", default='none', type=str,
                        help=("type of filter for generating low res input data. "
                              "choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'."))
    parser.add_argument("--lres_interp", default='linear', type=str,
                        help=("type of interpolation scheme for generating low res input data."
                              "choice of 'linear', 'nearest'"))
    parser.add_argument('--nonlin', type=str, default='leakyrelu', choices=list(NONLINEARITIES.keys()),
                        help='Nonlinear activations for continuous decoder.')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    # adjust batch size based on the number of gpus available
    if torch.cuda.device_count() > 0:
        args.batch_size = torch.cuda.device_count() * args.batch_size_per_gpu
    else:
        args.batch_size = args.batch_size_per_gpu

    # log and create snapshots
    os.makedirs(args.log_dir, exist_ok=True)
    #filenames_to_snapshot = glob("*.py") + glob("*.sh")
    #utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), 'w') as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tensorboard'))

    # random seed for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create dataloaders
    trainset = loader.Spatial3D_DataLoader(
        data_dir=args.data_folder_training, data_filename=args.train_data,
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xyz=args.downsamp_xyz,
        normalize_output=args.normalize_channels, normalize_hres=True, return_hres=args.normalize_channels,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp
    )

    
    evalset = loader.Spatial3D_DataLoader(
        data_dir=args.data_folder_evaluation, data_filename=args.eval_data,
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xyz=args.downsamp_xyz,
        normalize_output=args.normalize_channels, normalize_hres=True, return_hres=args.normalize_channels,
        lres_filter=args.lres_filter, lres_interp=args.lres_interp, trainset_mean=trainset._mean, trainset_std=trainset._std
    )

    train_sampler = RandomSampler(trainset, replacement=True, num_samples=args.pseudo_epoch_size)
    eval_sampler = RandomSampler(evalset, replacement=True, num_samples=args.pseudo_epoch_size)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler, **kwargs)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                             sampler=eval_sampler, **kwargs)
    # setup model
    unet = UNet3d(in_features=4, out_features=args.lat_dims, igres=trainset.scale_lres,
                  nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4, nf=args.imnet_nf, 
                  activation=NONLINEARITIES[args.nonlin])
    all_model_params = list(unet.parameters())+list(imnet.parameters())

    if args.optim == "sgd":
        optimizer = optim.SGD(all_model_params, lr=args.lr)
    else:
        optimizer = optim.Adam(all_model_params, lr=args.lr, weight_decay=1e-3)

    start_ep = 0
    global_step = int(0)

    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        unet.load_state_dict(resume_dict["unet_state_dict"])
        imnet.load_state_dict(resume_dict["imnet_state_dict"])
        optimizer.load_state_dict(resume_dict["optim_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    unet = nn.DataParallel(unet)
    unet.to(device)
    imnet = nn.DataParallel(imnet)
    imnet.to(device)

    model_param_count = lambda model: sum(x.numel() for x in model.parameters())
    logger.info("{}(unet) + {}(imnet) paramerters in total".format(model_param_count(unet),
                                                                   model_param_count(imnet)))
    print(f'Unet parameters = {model_param_count(unet)}')
    print(f'Imnet parameters = {model_param_count(imnet)}')

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    # get pdelayer for the RB2 equations
    if args.normalize_channels:
        mean = trainset.channel_mean
        std = trainset.channel_std
    else:
        mean = std = None

    pde_layer = get_3d_pde_layer(mean=mean, std=std, z_crop=1., y_crop=1., x_crop=1.)

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    # training loop
    patience = 30
    epochs_no_improve = 0 
    
    evaluation_losses = []
    training_losses = []

    tracked_stats = eval(args, unet, imnet, eval_loader, 0, global_step, device,logger, writer, optimizer, pde_layer)
    logger.info(f'This loss is on training set ↓')
    training_loss = eval(args, unet, imnet, train_loader, 0, global_step, device,logger, writer, optimizer, pde_layer)
    
    training_losses.append(training_loss)
    evaluation_losses.append(tracked_stats)
    for epoch in range(start_ep + 1, args.epochs + 1):
        training_loss = train(args, unet, imnet, train_loader, epoch, global_step, device, logger, writer,
                     optimizer, pde_layer)
        evaluation_loss = eval(args, unet, imnet, eval_loader, epoch, global_step, device,logger, writer, optimizer, pde_layer)
        training_losses.append(training_loss)
        evaluation_losses.append(evaluation_loss)
        
        if args.lr_scheduler:
            scheduler.step(evaluation_loss)

        if evaluation_loss < tracked_stats:
            tracked_stats = evaluation_loss
            is_best = True
            epochs_no_improve = 0
        else:
            is_best = False
            epochs_no_improve += 1

        utils.save_checkpoint({
            "epoch": epoch,
            "unet_state_dict": unet.module.state_dict(),
            "imnet_state_dict": imnet.module.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "tracked_stats": tracked_stats,
            "global_step": global_step,
        }, is_best, epoch, checkpoint_path, "_pdenet", logger)

        if epochs_no_improve > patience:
            break
    plot_losses(training_losses, evaluation_losses, len(training_losses), args.log_dir, logger, show=False)
if __name__ == "__main__":
    main()
