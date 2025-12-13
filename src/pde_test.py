"""Unit Test for regular_nd_grid_interpolation."""

# pylint: disable=import-error, no-member, too-many-arguments, no-self-use

import unittest
import torch
import pde
import numpy as np
from parameterized import parameterized


def generate_test_data_heat_eqn():
    rayleigh = 1e6
    prandtl = 1
    t_crop=2. 
    z_crop=1. 
    x_crop=2.
    y_crop = 2.
    
    P = (rayleigh * prandtl)**(-1/2)
    R = (rayleigh / prandtl)**(-1/2)
    nu = 1
    in_vars = 't, x, y, z'
    out_vars = 'p, u, v, w'
    nt, nx, ny, nz = 1./t_crop, 1./x_crop, 1./y_crop, 1./z_crop

    eqn_strs = [
        f'{nt}*dif(u,t) - {nu}*(({nx})**2*dif(dif(u,x),x)+({ny})**2*dif(dif(u,y),y)+({nz})**2*dif(dif(u,z),z)) + dif(p,x) + (u*{nx}*dif(u,x)+v*{ny}*dif(u,y)+w*{nz}*dif(u,z))',
        f'{nt}*dif(v,t) - {nu}*(({nx})**2*dif(dif(v,x),x)+({ny})**2*dif(dif(v,y),y)+({nz})**2*dif(dif(v,z),z)) + dif(p,y) + (u*{nx}*dif(v,x)+v*{ny}*dif(v,y)+w*{nz}*dif(v,z))',
        f'{nt}*dif(w,t) - {nu}*(({nx})**2*dif(dif(w,x),x)+({ny})**2*dif(dif(w,y),y)+({nz})**2*dif(dif(w,z),z)) + dif(p,z) + (u*{nx}*dif(w,x)+v*{ny}*dif(w,y)+w*{nz}*dif(w,z))',
        f'{nx}*dif(u,x) + {ny}*dif(v,y) + {nz}*dif(w,z)' # continuity
    ]
    eqn_names = ['transport_eqn_u', 'transport_eqn_v', 'transport_eqn_w', 'continuity']

    # arbitrary forward function, where inpt[0], inpt[1], inpt[2] correspond to x, y, t
    def fwd_fn(inpt):
        t, x, y, z = inpt[..., 0:1], inpt[..., 1:2], inpt[..., 2:3], inpt[..., 3:4]
        p = x + y + z + t
        u = x**2 + y*t
        v = y**2 + z*t
        w = z**2 + x*t
        return torch.cat([p, u, v, w], axis=-1)

    # input tensor
    inpt = torch.tensor([[1., 2., 3., 4.]])
    
    x, y, t = inpt[..., 0:1], inpt[..., 1:2], inpt[..., 2:3]
    g = x + 3*y**2 - 2 - 6*t
    expected_grads = {eqn_names[0]: g, eqn_names[1]: g}
    expected_val = fwd_fn(inpt)

    return in_vars, out_vars, eqn_strs, eqn_names, fwd_fn, inpt, expected_grads, expected_val

class PDELayerTest(unittest.TestCase):
    """Unit test for pde layer"""

    @parameterized.expand((
        generate_test_data_heat_eqn(),
        ))
    def test_pde_layer(self, in_vars, out_vars, eqn_strs, eqn_names,
                       fwd_fn, inpt, expected_grads, expected_val):
        """unit test for pde layer."""

        pdel = pde.PDELayer(in_vars=in_vars, out_vars=out_vars)
        for eqn_str, eqn_name in zip(eqn_strs, eqn_names):
            pdel.add_equation(eqn_str, eqn_name)
        pdel.update_forward_method(fwd_fn)
        val, grads = pdel(inpt)
        np.testing.assert_allclose(val.detach().numpy(), expected_val.detach().numpy(), atol=1e-4)
        for eqn_name in eqn_names:
            np.testing.assert_allclose(grads[eqn_name].detach().numpy(),
                                       expected_grads[eqn_name].detach().numpy())


if __name__ == '__main__':
    unittest.main()