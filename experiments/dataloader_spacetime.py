"""3D spatila Dataloader"""
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import warnings
# pylint: disable=too-manz-arguments, too-manz-instance-attributes, too-manz-locals
    
class Spatial3D_DataLoader(Dataset):
    """Pytorch Dataset instance for loading 4D dataset into 3D spatial.

    Loads a 3d space cubic cutout from the whole simulation.
    """
    def __init__(self, data_dir="./", data_filename="",
                 nx=128, ny=128, nz=128, n_samp_pts_per_crop=1024,
                 downsamp_xyz=4, normalize_output=False, normalize_hres=False,
                 return_hres=False, lres_filter='none', lres_interp='linear', trainset_mean = None, trainset_std = None):
        """

        Initialize DataSet
        Args:
          data_dir: str, path to the dataset folder, default="./"
          data_filename: str, name of the dataset file, default="rb2d_ra1e6_s42"
          nx: int, number of 'pixels' in x dimension for high res dataset.
          nz: int, number of 'pixels' in z dimension for high res dataset.
          ny: int, number of 'pixels' in y dimension for high res dataset.
          n_samp_pts_per_crop: int, number of sample points to return per crop.
          downsamp_xyz: int, downsampling factor for the spatial dimensions.
          normalize_output: bool, whether to normalize the range of each channel to [0, 1].
          normalize_hres: bool, normalize high res grid.
          return_hres: bool, whether to return the high-resolution data.
          lres_filter: str, filter to apply on original high-res image before interpolation.
                       choice of 'none', 'gaussian', 'uniform', 'median', 'maximum'.
          lres_interp: str, interpolation scheme for generating low res.
                       choice of 'linear', 'nearest'.
        """
        self.data_dir = data_dir
        self.data_filename = data_filename
        self.nx_hres = nx
        self.nz_hres = nz
        self.ny_hres = ny
        self.nx_lres = int(nx/downsamp_xyz)
        self.nz_lres = int(nz/downsamp_xyz)
        self.ny_lres = int(ny/downsamp_xyz)
        self.n_samp_pts_per_crop = n_samp_pts_per_crop
        self.downsamp_xyz = downsamp_xyz
        self.normalize_output = normalize_output
        self.normalize_hres = normalize_hres
        self.return_hres = return_hres
        self.lres_filter = lres_filter
        self.lres_interp = lres_interp

        # warn about median filter
        if lres_filter == 'median':
            warnings.warn("the median filter is very slow...", RuntimeWarning)

        # concatenating pressure, x-velocity, y-velocity and z-velocity as a 4 channel array: puvw
        # shape: (T, X, Y, Z)
        if data_dir != '':
            p = np.load(os.path.join(self.data_dir, 'p.npy'))  
            u = np.load(os.path.join(self.data_dir, 'u.npy'))
            v = np.load(os.path.join(self.data_dir, 'v.npy'))
            w = np.load(os.path.join(self.data_dir, 'w.npy'))
        elif data_filename != '':
            npdata = np.load(self.data_filename)
            p, u, v, w = npdata['p'], npdata['u'], npdata['v'], npdata['w']  # each [T, X, Y, Z]
        else:
            raise ValueError('Need either a file or a directory with 4 files (p,u,v,w)')

        T, X, Y, Z = p.shape
        snapshots = []
        for t in range(T):
            snap = np.stack([p[t], u[t], v[t], w[t]], axis=0)  # [C=4, X, Y, Z]
            snap = snap.transpose(0, 3, 2, 1)                  # [C, Z, Y, X]
            snapshots.append(snap)

        self.data = np.stack(snapshots, axis=0)  # [T, C, Z, Y, X]
        nt_data, nc_data, nz_data, ny_data, nx_data = self.data.shape


        # assert nx, nz, nt are viable
        if (nx > nx_data) or (nz > nz_data) or (ny > ny_data):
            raise ValueError('Resolution in each spatial temporal dimension x ({}), z({}), t({})'
                             'must not exceed dataset limits x ({}) z ({}) t ({})'.format(
                                 nx, ny, nz, nx_data, ny_data, nz_data))
        if (ny % downsamp_xyz != 0) or (nx % downsamp_xyz != 0) or (nz % downsamp_xyz != 0):
            raise ValueError('nx, nz and nt must be divisible by downsamp factor.')

        self.nx_start_range = np.arange(0, nx_data-nx+1)
        self.nz_start_range = np.arange(0, nz_data-nz+1)
        self.ny_start_range = np.arange(0, ny_data-ny+1)
        self.rand_grid = np.stack(np.meshgrid(self.nz_start_range,
                                              self.ny_start_range,
                                              self.nx_start_range, indexing='ij'), axis=-1)
        # (xaug, zaug, taug, 3)
        self.rand_start_id = self.rand_grid.reshape([-1, 3])
        self.scale_hres = np.array([self.nz_hres, self.ny_hres, self.nx_hres], dtype=np.int32)
        self.scale_lres = np.array([self.nz_lres, self.ny_lres, self.nx_lres], dtype=np.int32)

        # compute channel-wise mean and std
        if trainset_mean is None:
            self._mean = np.mean(self.data, axis=(0, 2, 3, 4))  # shape (4,)
            self._std  = np.std(self.data, axis=(0, 2, 3, 4))   # shape (4,)
        else:
            self._mean = trainset_mean
            self._std = trainset_std

    def __len__(self):
        return self.data.shape[0] * self.rand_start_id.shape[0]

    def filter(self, signal): ## NOT UPDATED FOR X Y Z but not used
        """Filter a given signal with a choice of filter type (self.lres_filter).
        """
        signal = signal.copy()
        filter_size = [1, self.downsamp_xyz*2-1, self.downsamp_xyz*2-1, self.downsamp_xyz*2-1]

        if self.lres_filter == 'none' or (not self.lres_filter):
            output = signal
        elif self.lres_filter == 'gaussian':
            sigma = [0, int(self.downsamp_t/2), int(self.downsamp_xz/2), int(self.downsamp_xz/2)]
            output = ndimage.gaussian_filter(signal, sigma=sigma)
        elif self.lres_filter == 'uniform':
            output = ndimage.uniform_filter(signal, size=filter_size)
        elif self.lres_filter == 'median':
            output = ndimage.median_filter(signal, size=filter_size)
        elif self.lres_filter == 'maximum':
            output = ndimage.maximum_filter(signal, size=filter_size)
        else:
            raise NotImplementedError(
                "lres_filter must be one of none/gaussian/uniform/median/maximum")
        return output

    def __getitem__(self, idx):
        """Get the random cutout data cube corresponding to idx.

        Args:
          idx: int, index of the crop to return. must be smaller than len(self).

        Returns:
          space_time_crop_hres (*optional): array of shape [4, nt_hres, nz_hres, nx_hres],
          where 4 are the phys channels pbuw.
          space_time_crop_lres: array of shape [4, nt_lres, nz_lres, nx_lres], where 4 are the phys
          channels pbuw.
          point_coord: array of shape [n_samp_pts_per_crop, 3], where 3 are the t, x, z dims.
                       CAUTION - point_coord are normalized to (0, 1) for the relative window.
          point_value: array of shape [n_samp_pts_per_crop, 4], where 4 are the phys channels pbuw.
        """
        t_id = idx // len(self.rand_start_id)   # pick which snapshot in time
        crop_id = idx % len(self.rand_start_id) # pick which spatial crop

        z_id, y_id, x_id = self.rand_start_id[crop_id]
        snapshot = self.data[t_id]  # [C, Z, Y, X]

        space_time_crop_hres = snapshot[:,
                                z_id:z_id+self.nz_hres,
                                y_id:y_id+self.ny_hres,
                                x_id:x_id+self.nx_hres]  # [C, Z, Y, X]
        
        # Data augmentation (desactivate if data is not symmetric)
        if np.random.rand() < 0.5:
            space_time_crop_hres = np.flip(space_time_crop_hres, axis=1).copy()  # flip Z
        if np.random.rand() < 0.5:
            space_time_crop_hres = np.flip(space_time_crop_hres, axis=2).copy()  # flip Y
        if np.random.rand() < 0.5:
            space_time_crop_hres = np.flip(space_time_crop_hres, axis=3).copy()  # flip X

        # create low res grid from hi res space time crop
        # apply filter
        space_time_crop_hres_fil = self.filter(space_time_crop_hres)

        interp = RegularGridInterpolator(
            (np.arange(self.nz_hres), np.arange(self.ny_hres), np.arange(self.nx_hres)),
            values=space_time_crop_hres_fil.transpose(1, 2, 3, 0), method=self.lres_interp)

        lres_coord = np.stack(np.meshgrid(np.linspace(0, self.nz_hres-1, self.nz_lres),
                                          np.linspace(0, self.ny_hres-1, self.ny_lres),
                                          np.linspace(0, self.nx_hres-1, self.nx_lres),
                                          indexing='ij'), axis=-1)
        space_time_crop_lres = interp(lres_coord).transpose(3, 0, 1, 2)  # [c, z, y, x]

        # create random point samples within space time crop
        point_coord = np.random.rand(self.n_samp_pts_per_crop, 3) * (self.scale_hres - 1)
        point_value = interp(point_coord)
        point_coord = point_coord / (self.scale_hres - 1)

        if self.normalize_output:
            space_time_crop_lres = self.normalize_grid(space_time_crop_lres)
            point_value = self.normalize_points(point_value)
        if self.normalize_hres:
            space_time_crop_hres = self.normalize_grid(space_time_crop_hres)

        return_tensors = [space_time_crop_lres, point_coord, point_value]

        # cast everything to float32
        return_tensors = [t.astype(np.float32) for t in return_tensors]

        if self.return_hres:
            return_tensors = [space_time_crop_hres] + return_tensors
        return tuple(return_tensors)

    @property
    def channel_mean(self):
        """channel-wise mean of dataset."""
        return self._mean

    @property
    def channel_std(self):
        """channel-wise mean of dataset."""
        return self._std

    @staticmethod
    def _normalize_array(array, mean, std):
        """normalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            dev = array.device
            std = torch.tensor(std, device=dev)
            mean = torch.tensor(mean, device=dev)
        return (array - mean) / std

    @staticmethod
    def _denormalize_array(array, mean, std):
        """normalize array (np or torch)."""
        if isinstance(array, torch.Tensor):
            dev = array.device
            std = torch.tensor(std, device=dev)
            mean = torch.tensor(mean, device=dev)
        return array * std + mean

    def normalize_grid(self, grid):
        """Normalize grid over physics channels."""
        if grid.ndim == 4:  
            # [C, Z, Y, X]
            mean_bc = self.channel_mean[:, None, None, None]
            std_bc  = self.channel_std[:, None, None, None]
        elif grid.ndim == 5:  
            # [B, C, Z, Y, X]
            mean_bc = self.channel_mean[None, :, None, None, None]
            std_bc  = self.channel_std[None, :, None, None, None]
        else:
            raise ValueError(f"Unexpected grid shape: {grid.shape}")
        return (grid - mean_bc) / std_bc


    def normalize_points(self, points):
        """Normalize points.

        Args:
          points: np array or torch tensor of shape [..., 4], 4 are the num. of phys channels.
        Returns:
          channel normalized points of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]  # unsqueeze from the front
        std_bc = self.channel_std[(None,)*(g_dim-1)]  # unsqueeze from the front
        return self._normalize_array(points, mean_bc, std_bc)

    def denormalize_grid(self, grid):
        """Denormalize grid.

        Args:
          grid: np array or torch tensor of shape [4, ...], 4 are the num. of phys channels.
        Returns:
          channel denormalized grid of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(grid.shape)
        mean_bc = self.channel_mean[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        std_bc = self.channel_std[(...,)+(None,)*(g_dim-1)]  # unsqueeze from the back
        return self._denormalize_array(grid, mean_bc, std_bc)


    def denormalize_points(self, points):
        """Denormalize points.

        Args:
          points: np array or torch tensor of shape [..., 4], 4 are the num. of phys channels.
        Returns:
          channel denormalized points of same shape as input.
        """
        # reshape mean and std to be broadcastable.
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]  # unsqueeze from the front
        std_bc = self.channel_std[(None,)*(g_dim-1)]  # unsqueeze from the front
        return self._denormalize_array(points, mean_bc, std_bc)


