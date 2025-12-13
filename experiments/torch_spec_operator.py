# corrected_fft_helpers.py
import torch
import numpy as np

# -----------------------
# Helpers: complex <-> real/imag pair
# -----------------------
def complex_from_realimag(x):
    """
    Convert tensor with last dim [real, imag] to complex tensor.
    Input shape: [..., 2]
    Output shape: [...] complex
    """
    return x[..., 0] + 1j * x[..., 1]


def realimag_from_complex(z):
    """
    Convert complex tensor to last-dim [real, imag].
    Input shape: [...] complex
    Output shape: [..., 2]
    """
    return torch.stack([z.real, z.imag], dim=-1)


# -----------------------
# 3D padded rfft / irfft (operate with last-dim real/imag convention)
# -----------------------
def pad_rfft3(f, onesided=True):
    """
    Padded batch real fft (3D)
    :param f: real tensor of shape [..., res0, res1, res2]
    :return: tensor of shape [..., res0, res1, res2//2+1, 2] (real/imag last dim)
    Process:
        1) rfft along last dim (z) -> complex [..., r0, r1, r2h]
        2) fft along -2 (y)
        3) fft along -3 (x)
       After each transform zero the Nyquist index along the transformed axis the
       same way your original code attempted.
    """
    # sizes
    n0, n1, n2 = f.shape[-3:]
    # half indices used in original code
    h0, h1, h2 = int(n0 // 2), int(n1 // 2), int(n2 // 2)

    # 1) rfft along z (last axis)
    F = torch.fft.rfft(f, dim=-1)  # complex tensor [..., r0, r1, r2h]
    # zero Nyquist (if present) along last axis at index h2
    # note: for n2 even, index h2 == n2/2 corresponds to last index (r2h-1)
    if 0 <= h2 < F.shape[-1]:
        F = F.clone()
        F[..., h2] = 0 + 0j

    # 2) fft along y (axis -2)
    F = torch.fft.fft(F, dim=-2)   # complex tensor [..., r0, r1, r2h]
    if 0 <= h1 < F.shape[-2]:
        F = F.clone()
        # zero the h1 slice along axis -2
        F.select(dim=-2, index=h1).zero_()

    # 3) fft along x (axis -3)
    F = torch.fft.fft(F, dim=-3)   # complex tensor [..., r0, r1, r2h]
    if 0 <= h0 < F.shape[-3]:
        F = F.clone()
        F.select(dim=-3, index=h0).zero_()

    # convert to real/imag last-dim format
    return realimag_from_complex(F)


def pad_irfft3(F):
    """
    Padded batch inverse real fft (3D)
    :param F: tensor of shape [..., res0, res1, res2//2+1, 2] (real/imag)
    :return: real tensor of shape [..., res0, res1, res2]
    Inverse of pad_rfft3: do ifft along -3, ifft along -2, irfft along -1.
    """
    # Convert to complex
    Z = complex_from_realimag(F)  # complex [..., r0, r1, r2h]

    # inverse order of pad_rfft3
    z = torch.fft.ifft(Z, dim=-3)   # ifft along x
    z = torch.fft.ifft(z, dim=-2)   # ifft along y

    # irfft along last axis returns real-valued signal of length n2
    # need original size: infer res from F's shape
    res2h = F.shape[-2]  # r2h
    # reconstruct n2 from r2h (for even n2: r2h = n2/2 + 1 -> n2 = 2*(r2h - 1))
    n2 = int(2 * (res2h - 1))
    f_real = torch.fft.irfft(z, n=n2, dim=-1)  # real tensor [..., r0, r1, n2]
    return f_real


# -----------------------
# 2D padded fft / ifft helpers (operate with last-dim real/imag convention)
# -----------------------
def pad_fft2(f):
    """
    Padded batch real 2D FFT (for mean in z-plane)
    :param f: real tensor of shape [..., res0, res1]
    :return: complex represented as [..., res0, res1, 2] (real/imag)
    Implementation: compute 2D FFT, zero Nyquist indices along axes similar to original.
    """
    n0, n1 = f.shape[-2:]
    h0, h1 = int(n0 // 2), int(n1 // 2)

    # compute 2D FFT (complex)
    F2 = torch.fft.fft2(f, dim=(-2, -1))  # complex [..., r0, r1]

    # zero Nyquist slices if present
    if 0 <= h1 < F2.shape[-1]:
        F2 = F2.clone()
        F2.select(dim=-1, index=h1).zero_()
    if 0 <= h0 < F2.shape[-2]:
        F2 = F2.clone()
        F2.select(dim=-2, index=h0).zero_()

    return realimag_from_complex(F2)


def pad_ifft2(F):
    """
    Padded batch inverse 2D FFT
    :param F: complex in real/imag format [..., res0, res1, 2]
    :return: real tensor [..., res0, res1]
    """
    Z = complex_from_realimag(F)
    f_real = torch.fft.ifft2(Z, dim=(-2, -1)).real
    return f_real


# -----------------------
# Frequency helper functions
# -----------------------
def rfftfreqs(res, dtype=torch.float32, exact=True):
    """
    Return frequency tensors for rfft grid.
    res: sequence length n_dims (e.g. [r0, r1, r2])
    returns tensor shape [ndim, r0, r1, r2h]
    """
    n_dims = len(res)
    freqs = []
    # for all dims except last, use full fftfreq
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1 / r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    # last dim: rfftfreq
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_)[:-1], dtype=dtype))

    # create meshgrid with 'ij' indexing to maintain order
    omega = torch.meshgrid(*freqs, indexing='ij')
    omega = torch.stack(list(omega), dim=0)  # [ndim, r0, r1, r2h]
    return omega


def fftfreqs(res, dtype=torch.float32):
    """
    Return full FFT frequency grid (for complex full FFT).
    returns tensor shape [ndim, r0, r1, ...]
    """
    n_dims = len(res)
    freqs = []
    for dim in range(n_dims):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1 / r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    omega = torch.meshgrid(*freqs, indexing='ij')
    omega = torch.stack(list(omega), dim=0)
    return omega


# -----------------------
# Imaginary multiplier helper (operates on last-dim real/imag)
# -----------------------
def img(x, deg=1):
    """
    Multiply tensor x by i**deg, assuming last dimension is [real, imag].
    deg modulo 4.
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        # i * (a + i b) = -b + i a
        # x[..., [1,0]] swaps imag/real: gives [imag, real]; then set real = -imag
        res = x[..., [1, 0]].clone()
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        # -1 * (a + i b) = -a + i(-b)
        res = -x
    elif deg == 3:
        # i**3 = -i: -i * (a + i b) = b - i a
        res = x[..., [1, 0]].clone()
        res[..., 1] = -res[..., 1]
    return res


# -----------------------
# Reconstruction / spectral ops
# -----------------------
def reconstruct(uv, w_):
    """
    Reconstruct entire field from uv and mean w in z
    :param uv: tensor of shape (batch, 2, res0, res1, res2)  (real-valued)
    :param w_: tensor of shape (batch, 1, res0, res1)      (real-valued mean in z)
    returns: uvw real tensor shape [batch, 3, res0, res1, res2]
    """
    # ensure shapes
    res = uv.shape[-3]
    # spectral transform of uv: get real/imag last-dim
    UV = pad_rfft3(uv)  # [batch, 2, r0, r1, r2h, 2]
    U = UV[:, 0:1]      # [batch, 1, r0, r1, r2h, 2]
    V = UV[:, 1:2]

    # spectral transform of mean plane w_ (2D)
    W0 = pad_fft2(w_)   # [batch, 1, r0, r1, 2]

    # frequencies
    K = rfftfreqs([res] * 3, dtype=uv.dtype).to(uv.device)  # [3, r0, r1, r2h]
    # avoid division by 0 on K[2] at (0,0,0)
    K = K.clone()
    K[2, 0, 0, 0] += 1.0
    K = K.unsqueeze(-1)  # [3, r0, r1, r2h, 1]

    # convert U,V to complex for arithmetic
    Uc = complex_from_realimag(U)  # complex [..., r2h]
    Vc = complex_from_realimag(V)

    # compute W in spectral domain: W = - (K0*U + K1*V) / K2
    K0 = K[0].to(uv.device)
    K1 = K[1].to(uv.device)
    K2 = K[2].to(uv.device)
    Wc = - (K0 * Uc + K1 * Vc) / K2  # complex

    # convert back to real/imag pair
    W = realimag_from_complex(Wc)  # [batch,1,r0,r1,r2h,2]

    # assign mean (zero-frequency along last axis) to the plane W0
    # W[..., 0, :] selects the last-axis index 0 (k=0 frequency) slice
    # W0 shape: [batch,1,r0,r1,2] -> broadcast to fit
    W = W.clone()
    # ensure W0 matches indexing order: W[..., 0, :] -> shape [batch,1,r0,r1,2]
    W[..., 0, :] = W0  # assign mean plane

    # inverse transform to physical space
    w = pad_irfft3(W)  # [batch, 1, r0, r1, r2]

    # concatenate uv and reconstructed w
    uvw = torch.cat((uv, w), dim=1)  # [batch, 3, r0, r1, r2]

    # debug prints similar to original (comment out in production)
    # compute Div = -img(K * F) where F = [U,V,W] in real/imag format
    F = torch.cat((U, V, W), dim=1)  # [batch,3, r0,r1,r2h,2]
    Kstack = rfftfreqs([res] * 3, dtype=uv.dtype).to(uv.device).unsqueeze(-1)  # [3,r0,r1,r2h,1]
    Div = -img(Kstack * F)  # elementwise mult in real/imag format
    # print first complex components if needed:
    # print(Div[0,2,:,:,0,0])
    # print(Div[0,2,:,:,0,1])

    return uvw


def spec_grad(S, dim=[0, 1]):
    """
    Compute spectral gradient of scalar field (or Jacobian of vector field)
    Assumes last-dim real/imag format.
    :param S: scalar field of shape [batch, res, res, res2h, 2]
              or vector field [batch, dim, res, res, res2h, 2]
    :param dim: list/tuple of axes to compute gradient along (0->x,1->y)
    returns: -i * K * S on requested dims
    """
    assert (len(S.shape) in [5, 6])
    assert (isinstance(dim, list) or isinstance(dim, tuple))
    is_scalar = (len(S.shape) == 5)
    res = S.shape[-3]
    # K shape [2, res, res, r2h]
    K = rfftfreqs([res, res, S.shape[-2]], dtype=S.dtype).to(S.device)
    # pick only requested dim(s)
    if is_scalar:
        # return shape [batch, len(dim), res, res, r2h, 2]
        outs = []
        for d in dim:
            Kd = K[d]  # [res,res,r2h]
            Kd = Kd.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # broadcast
            outs.append(-img(Kd * S.unsqueeze(1)).squeeze(1))
        return torch.stack(outs, dim=1)
    else:
        # vector field: S shape [batch, dimv, res, res, r2h, 2]
        # return -i * K[dim] * S (broadcasted)
        Kd = K[dim]  # K for requested dims
        Kd = Kd.unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # [1,1,len(dim),res,res,r2h,1] maybe adjust below
        # We will compute for each requested dim separately; simpler approach:
        outs = []
        for d in dim:
            Kcur = K[d].unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # [1,1,res,res,r2h,1]
            outs.append(-img(Kcur * S))
        # stack over the new axis representing gradient directions
        return torch.stack(outs, dim=2)  # [batch, dimv, len(dim), res, res, r2h, 2]


def spec_div(F):
    """
    Compute spectral divergence
    :param F: vector field tensor of shape (batch, dim=3, res, res, r2h, 2)
    returns Div: [batch, res, res, r2h, 2]
    """
    res = F.shape[2]
    K = rfftfreqs([res] * 3, dtype=F.dtype).unsqueeze(-1).to(F.device)  # [3,res,res,r2h,1]
    Div = torch.sum(-img(K * F), dim=1)  # sum over vector components
    return Div


def spec_curl(F):
    """
    Compute spectral curl
    :param F: vector field tensor [batch, 3, res, res, r2h, 2]
    returns curl same shape
    """
    # compute Jacobian-like spec_grad but adapted
    # J shape [batch, 3, 3, res, res, r2h, 2]
    # For simplicity, compute derivatives of each component with respect to each axis
    J = spec_grad(F, dim=[0, 1, 2])  # careful with returned shape; adapt indexing below
    # The shape returned by our spec_grad in vector case is [batch, dimv, len(dim), res, res, r2h, 2]
    # dimv = 3, len(dim)=3 -> J[b,comp,axis,...]
    # Curl components: cF_i = eps_{i j k} d_j F_k
    # that expansion:
    # cF_x = d_y W - d_z V -> J[:, :, 1, ...] for derivative index 1 (y) etc.
    # rearrange indices accordingly
    # J[:, comp, axis, ...]
    c_x = J[:, 2, 1] - J[:, 1, 2]  # d_y W - d_z V
    c_y = J[:, 0, 2] - J[:, 2, 0]  # d_z U - d_x W
    c_z = J[:, 1, 0] - J[:, 0, 1]  # d_x V - d_y U
    cF = torch.stack([c_x, c_y, c_z], dim=1)
    return cF


def phys_div(f):
    """
    Compute physical divergence (real space)
    :param f: real vector field tensor of shape [batch, 3, res, res, res]
    """
    F = pad_rfft3(f)  # [batch,3,res,res,r2h,2]
    Div = spec_div(F)
    div = pad_irfft3(Div)
    return div


def phys_proj(f):
    """
    Project physical vector field to be solenoidal
    :param f: real vector field [batch, dim, res, res, res]
    returns projected real field [batch, dim, res, res, res]
    """
    F = pad_rfft3(f)
    F_ = spec_proj(F)
    f_ = pad_irfft3(F_)
    return f_


def spec_proj(F):
    """
    Project spectral vector field to be solenoidal
    :param F: vector field tensor of shape (batch, dim, res, res, r2h, 2)
    :return sF: solenoidal field F in same spectral format
    """
    res = F.shape[2]
    # divF: [batch, res, res, r2h, 2]
    divF = spec_div(F)
    K = rfftfreqs([res] * 3, dtype=F.dtype).unsqueeze(-1).to(F.device)  # [3,res,res,r2h,1]
    Lap = -torch.sum(K ** 2, dim=0)  # [res,res,r2h,1]
    Lap = Lap.clone()
    Lap[0, 0, 0] = 1.0  # prevent division by 0
    Phi = divF / Lap  # [batch, res, res, r2h, 2]
    # arbitrary gauge value
    Phi = Phi.clone()
    Phi[:, 0, 0, 0] = 0.0
    sF = F - spec_grad(Phi, dim=[0, 1, 2])  # spec_grad returns shape with derivative axes
    # spec_grad returned stacked axes; we must subtract appropriate representation
    # spec_grad(Phi, dim=[0,1,2]) shapes are different from F; but the operation intended:
    # F - grad(Phi) where grad(Phi) returns vector field. We'll instead explicitly compute vector grad:
    # compute spectral gradient of scalar Phi -> returns [batch, len(dim), res,res,r2h,2] where len(dim)=3
    gradPhi = spec_grad(Phi, dim=[0, 1, 2])  # [batch, 3, res, res, r2h, 2] for scalar input
    sF = F - gradPhi
    return sF
