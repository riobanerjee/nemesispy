#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def gen_vmrmap1(h2o,co2,co,ch4,nlon,nlat,npress,
        h2_frac = 0.84):
    """
    Generate a 3D gas abundance map.
    The abundance map is defined on a (longitude,latitude,pressure) grid.

    Parameters
    ---------

    Returns
    -------
        vmr_grid
    """
    he_frac = 1 - h2_frac
    vmr_grid = np.ones((nlon,nlat,npress,6))
    vmr_grid[:,:,:,0] *= 10**h2o
    vmr_grid[:,:,:,1] *= 10**co2
    vmr_grid[:,:,:,2] *= 10**co
    vmr_grid[:,:,:,3] *= 10**ch4
    vmr_grid[:,:,:,4] *= he_frac * (1-10**h2o-10**co2-10**co-10**ch4)
    vmr_grid[:,:,:,5] *= h2_frac * (1-10**h2o-10**co2-10**co-10**ch4)
    return vmr_grid

def gen_vmr(NLAYER,gases, h2_frac = 0.8547):

    vmrlayer = np.zeros([len(gases) + 2])
    vmrlayer[0:len(gases)] = 10.0 ** np.array(gases)
    vmr_bg = 1.0 - np.sum(vmrlayer)
    vmrlayer[len(gases)] = h2_frac * vmr_bg	# H2
    vmrlayer[len(gases)+1] = (1.0-h2_frac) * vmr_bg	# He			
    vmr = np.tile(vmrlayer, (NLAYER, 1))
    return vmr

def gen_vmr_bg(NLAYER,gases,bg_index=-1):

    vmrlayer = np.zeros([len(gases) + 1])
    vmrlayer[0:len(gases)] = 10.0 ** np.array(gases)
    if np.sum(vmrlayer) > 1.0:
        vmrlayer = vmrlayer / np.sum(vmrlayer)
    vmrlayer[len(gases)] = 1.0 - np.sum(vmrlayer)
    vmr = np.tile(vmrlayer, (NLAYER, 1))
    return vmr

def gen_vmr_clr(NLAYER,gases):
    # Requires CLR priors

    vmrlayer = 10.0 ** np.array(gases)
    if np.abs(np.sum(vmrlayer)-1.0) > 1e-3:
        vmrlayer = vmrlayer / np.sum(vmrlayer)
    vmr = np.tile(vmrlayer, (NLAYER, 1))
    return vmr

def mov_avg(a, n=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def gen_vmr_2layer(NLAYER, P_layer, two_layer_top=None, two_layer_bot=None,
                    P_transition=None, one_layer_gas=None, h2_frac=0.8547):

    nvmr = len(two_layer_top) + len(one_layer_gas)
    vmr = np.zeros((NLAYER, nvmr+2))

    smooth_window = 10
    for i in range(len(two_layer_top)):
        P_layer_idx = np.abs(P_layer - P_transition[i]).argmin()
        start_layer = max(int(P_layer_idx-smooth_window/2), 0)
        end_layer = min(int(P_layer_idx+smooth_window/2), NLAYER-1)


        Pnodes = [P_layer[0], P_layer[start_layer],
                P_layer[end_layer], P_layer[-1]]

        Cnodes = [two_layer_bot[i], two_layer_bot[i],
                two_layer_top[i], two_layer_top[i]]

        chemprofile = 10**np.interp((np.log(P_layer[::-1])),
                                    np.log(Pnodes[::-1]),
                                    Cnodes[::-1])
        

        wsize = NLAYER * (smooth_window / 100.0)

        if (wsize % 2 == 0):
            wsize += 1

        C_smooth = 10**mov_avg(np.log10(chemprofile), int(wsize))

        border = int((len(chemprofile) - len(C_smooth)) / 2)

        vmr[:, i] = chemprofile[::-1]

        vmr[border:-border, i] = C_smooth[::-1]


    for i in range(len(two_layer_top), nvmr):
        chemprofile = 10.0**float(one_layer_gas[i-len(one_layer_gas)]) * np.ones(NLAYER)
        vmr[:, i] = chemprofile


    vmr_bg = 1.0 - np.sum(vmr, axis=1)
    h2_profile = h2_frac * vmr_bg	# H2
    vmr[:, nvmr] = h2_profile
    he_profile = (1.0-h2_frac) * vmr_bg	# He
    vmr[:, nvmr+1] = he_profile
    # vmr[len(two_layer_top):nvmr, :] = np.tile(10.0 ** np.array(one_layer_gas), (NLAYER, 1))

    return vmr