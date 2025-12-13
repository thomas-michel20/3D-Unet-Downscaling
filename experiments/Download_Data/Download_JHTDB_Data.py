"""
instantiate dataset
"""
from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *
import numpy as np
import os

path = os.getcwd()

auth_token = '' # Need to ask for access
dataset_title = 'isotropic1024coarse'
output_path = './giverny_output'

# instantiate the dataset.
dataset = turb_dataset(dataset_title = dataset_title, output_path = output_path, auth_token = auth_token)

"""
initialize getData parameters (except time and points)
"""
variable = 'velocity'
temporal_method = 'none'
spatial_method = 'lag8'
spatial_operator = 'field'

"""
example point distributions (2D plane, 3D box, random, time series) are provided below...
""";

"""
time series demo point(s)
    - time : the start time of the time series (snapshot number for datasets without a full time evolution).
    - time_end : the end time of the time series (snapshot number for datasets without a full time evolution).
    - delta_t : time step.
    - points : the points array.
"""
# create a temporary variable for the temporal method to set to 'pchip' for this time series example.
temporal_method_tmp = 'pchip'

nx = 16
ny = 16
nz = 16
n_points = nx * ny * nz
x_points = np.linspace(0.0, 2*np.pi, nx, dtype = np.float64)
y_points = np.linspace(0.0, 2*np.pi, ny, dtype = np.float64)
z_points = np.linspace(0.0, 2*np.pi, nz, dtype = np.float64)

points = np.array([axis.ravel() for axis in np.meshgrid(x_points, y_points, z_points, indexing = 'ij')], dtype = np.float64).T

time = 0.0
delta_t = 0.03
time_end = 7.0 + delta_t
n_intervals = int((time_end - time + delta_t) / (2 * delta_t))

time_val = 7.5 + delta_t
delta_t_val = 0.1
time_end_val = 9.5 + delta_t_val
n_intervals_val = int((time_end_val - time_val + delta_t_val) / (2 * delta_t_val))

result_pressure = []
result_velocity = []
times = []
log_path = "JHTDB_download.txt"

with open(log_path, "w") as log:
# process interpolation/differentiation of points.
    for i in range(n_intervals):
        time_start = time + i * 2 * delta_t
        print(time_start, file=log, flush=True)   
        time_end_loop = time_start + delta_t
        option = [time_end_loop, delta_t]
        variable = 'pressure'
        result_p_temp, times_temp = getData(dataset, variable, time_start, temporal_method_tmp, spatial_method, spatial_operator, points, option, return_times = True)
        variable = 'velocity'
        result_velocity_temp, times_temp = getData(dataset, variable, time_start, temporal_method_tmp, spatial_method, spatial_operator, points, option, return_times = True)
        result_pressure = result_pressure + result_p_temp
        result_velocity = result_velocity + result_velocity_temp
        times = np.append(times, times_temp)
    print(f'Training : Total number of timesteps extracted {len(times)} (from {times[0]} to {times[-1]})', file=log)

write_interpolation_tsv_file(dataset, points, result_pressure, output_filename = 'pressure_file')
write_interpolation_tsv_file(dataset, points, result_velocity, output_filename = 'velocity_file')

velocity_file = path + '/giverny_output/velocity_file.tsv'
pressure_file = path + '/giverny_output/pressure_file.tsv'
vel_df = pd.read_csv(filepath_or_buffer=velocity_file, sep='\t', skiprows=1)
p_df = pd.read_csv(filepath_or_buffer=pressure_file, sep='\t', skiprows=1)
p_df = p_df.drop(['uy', 'uz'],axis=1).rename(columns={"ux": "p"})
df = vel_df.merge(right=p_df, on=['x_point', 'y_point', 'z_point', 'time']) 
times = np.sort(df['time'].unique())
xs = np.sort(df['x_point'].unique())
ys = np.sort(df['y_point'].unique())
zs = np.sort(df['z_point'].unique())

T, X, Y, Z = len(times), len(xs), len(ys), len(zs)

# Create an index mapping from coordinate -> position in array
time_index = {t: i for i, t in enumerate(times)}
x_index = {x: i for i, x in enumerate(xs)}
y_index = {y: i for i, y in enumerate(ys)}
z_index = {z: i for i, z in enumerate(zs)}

# Allocate arrays
u = np.zeros((T, X, Y, Z))
v = np.zeros((T, X, Y, Z))
w = np.zeros((T, X, Y, Z))
p = np.zeros((T, X, Y, Z))

# Fill arrays
for _, row in df.iterrows():
    ti = time_index[row['time']]
    xi = x_index[row['x_point']]
    yi = y_index[row['y_point']]
    zi = z_index[row['z_point']]
    
    u[ti, xi, yi, zi] = row['ux']
    v[ti, xi, yi, zi] = row['uy']
    w[ti, xi, yi, zi] = row['uz']
    p[ti, xi, yi, zi] = row['p']

np.savez(f"data/3d_isotropic_train_{nx}_{delta_t}_{time_end}.npz", u=u, v=v, w=w, p=p)

print('training.npz was written')

result_pressure = []
result_velocity = []
times = []
with open(log_path, "w") as log:
# process interpolation/differentiation of points.
    for i in range(n_intervals_val):
        time_start = time_val + i * 2 * delta_t_val
        time_end_loop = time_start + delta_t_val
        option = [time_end_loop, delta_t_val]
        variable = 'pressure'
        result_p_temp, times_temp = getData(dataset, variable, time_start, temporal_method_tmp, spatial_method, spatial_operator, points, option, return_times = True)
        variable = 'velocity'
        result_velocity_temp, times_temp = getData(dataset, variable, time_start, temporal_method_tmp, spatial_method, spatial_operator, points, option, return_times = True)
        result_pressure = result_pressure + result_p_temp
        result_velocity = result_velocity + result_velocity_temp
        times = np.append(times, times_temp)
    print(f'Validation : Total number of timesteps extracted {len(times)} (from {times[0]} to {times[-1]})')

write_interpolation_tsv_file(dataset, points, result_pressure, output_filename = 'pressure_file_val')
write_interpolation_tsv_file(dataset, points, result_velocity, output_filename = 'velocity_file_val')

velocity_file = path + '/giverny_output/velocity_file_val.tsv'
pressure_file = path + '/giverny_output/pressure_file_val.tsv'
vel_df = pd.read_csv(filepath_or_buffer=velocity_file, sep='\t', skiprows=1)
p_df = pd.read_csv(filepath_or_buffer=pressure_file, sep='\t', skiprows=1)
p_df = p_df.drop(['uy', 'uz'],axis=1).rename(columns={"ux": "p"})
df = vel_df.merge(right=p_df, on=['x_point', 'y_point', 'z_point', 'time']) 
times = np.sort(df['time'].unique())
xs = np.sort(df['x_point'].unique())
ys = np.sort(df['y_point'].unique())
zs = np.sort(df['z_point'].unique())

T, X, Y, Z = len(times), len(xs), len(ys), len(zs)

# Create an index mapping from coordinate -> position in array
time_index = {t: i for i, t in enumerate(times)}
x_index = {x: i for i, x in enumerate(xs)}
y_index = {y: i for i, y in enumerate(ys)}
z_index = {z: i for i, z in enumerate(zs)}

# Allocate arrays
u = np.zeros((T, X, Y, Z))
v = np.zeros((T, X, Y, Z))
w = np.zeros((T, X, Y, Z))
p = np.zeros((T, X, Y, Z))

# Fill arrays
for _, row in df.iterrows():
    ti = time_index[row['time']]
    xi = x_index[row['x_point']]
    yi = y_index[row['y_point']]
    zi = z_index[row['z_point']]
    
    u[ti, xi, yi, zi] = row['ux']
    v[ti, xi, yi, zi] = row['uy']
    w[ti, xi, yi, zi] = row['uz']
    p[ti, xi, yi, zi] = row['p']

np.savez(f"data/3d_isotropic_val_{nx}_{delta_t}_{time_end}_2.npz", u=u, v=v, w=w, p=p)

print('validation.npz was written')

