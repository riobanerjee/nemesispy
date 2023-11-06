import numpy as np
from numba import jit
from nemesispy.radtran.calc_mmw import calc_mmw

@jit(nopython=True)
def calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power, haze_mult=1.0):
    """
    Grey cloud deck at Ptop with haze layer (by scaling Rayleigh scattering)

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths grid for calculating spectra.
        Unit: um
    P_layer(NLAYER) : ndarray
        Pressure at each layer.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Temperature at each layer.
        Unit: K
    mmw(NLAYER) : ndarray
        Mean molecular weight at each layer.
        Unit: g/mol
    Ptop : float
        Pressure at the top of the cloud deck.
        Unit: Pa
    power : float
        Power law index for scaling Rayleigh scattering.
        Unit: None
    haze_mult : float
        Multiplicative factor for scaling Rayleigh scattering.
        Unit: None
    
    Returns
    -------
    tau_cloud(NWAVE, NLAYER) : ndarray
        Optical depth of the cloud deck.
        Unit: None

    """

    NLAYER = len(P_layer)
    NWAVE = len(wave_grid)
    tau_cloud = np.zeros((NWAVE,NLAYER))
    wv_0 = 3.5 #um
    kappa_zero =  1.0
    ctop = 0
    if Ptop is not None:
        for ilayer in range(NLAYER):
            if P_layer[ilayer] > Ptop:
                ctop = ilayer
                tau_cloud[:,ilayer] = 1.0e10 #increase?
    rho0 = P_layer[0] * mmw[0] / (8.314 * T_layer[0])

    if power is not None:
        # rho = P_layer[ilayer] * 0.1013 * mmw / (8.314 * T_layer[ilayer])
        
    #         nd = rho * Q[iwave, ilayer]
    #         tau_cloud[iwave, ilayer] = xsec * rho * Q[iwave, ilayer]
        for ilayer in range(ctop, NLAYER):
            rho = P_layer[ilayer] * mmw[ilayer] / (8.314 * T_layer[ilayer])
            rho_scaled = rho / rho0
            for iwave in range(NWAVE):
                scat = rho_scaled * kappa_zero * (wave_grid[iwave] / wv_0) ** power 
                tau_cloud[iwave, ilayer] = tau_cloud[iwave, ilayer] + haze_mult * scat

    return tau_cloud


    # for ilayer in range(NLAYER):
    #     for iwave in range(NWAVE):
    #         if H_layer[ilayer] > Hknee:
    #             Q[iwave, ilayer] = 1.0
    #         else:
    #             Q[iwave ,ilayer] = 1.0e30
    #         rho = P_layer[ilayer] * 0.1013 * mmw / (8.314 * T_layer[ilayer])
    #         nd = rho * Q[iwave, ilayer]
    #         tau_cloud[iwave, ilayer] = xsec * rho * Q[iwave, ilayer]
    

# Models from cloud2con Michiel Min
def cloud_deck(P_layer, P_deck, phi):
    """
    Calculates something using the cloud deck model, can't see though it to the bottom

    Parameters
    ----------
    P_deck : float
        Pressure at the cloud deck
        Units: Pa
        Low: 1e2 Pa
        High: 1e8 Pa
    phi : float
        Cloud deck parameter
        Units: Pa
        Low: 1e2 Pa
        High: 1e8 Pa

    """

    f_cloud = (g/(k_cloud * phi)) * np.exp((P_layer - P_deck)/phi) / (1.0 - np.exp(-P_deck/phi))
    return f_cloud

def cloud_slab(P_layer, P_top, P_bottom, tau_cloud):
    """
    Calculates something using the cloud slab model, can see though it to the bottom

    Parameters
    ----------
    P_bottom : float
        Pressure at the bottom of the cloud slab
        Units: Pa
        Low: 1e2 Pa
        High: 1e8 Pa
    P_top : float
        Pressure at the top of the cloud slab
        Units: Pa
        Low: 1e2 Pa
        High: 1e8 Pa
    tau_cloud : float
        Optical depth of the cloud slab
        Units: None
        Low: 1e-6
        High: 1e2

    """

    f_cloud = 2 * g * P_layer * tau_cloud / (k_cloud * (P_bottom**2 - P_top**2))
    return f_cloud


def cloud_layer_finite(P_layer, P_top, P_bottom, tau_cloud, xi):
    """
    Calculates something using the cloud layer model, can see though it to the bottom

    Parameters
    ----------
    P_bottom : float
        Pressure at the bottom of the cloud layer
        Units: Pa
        Low: 1e2 Pa
        High: 1e8 Pa
    P_top : float
        Pressure at the top of the cloud layer
        Units: Pa
        Low: 1e2 Pa
        High: 1e8 Pa
    tau_cloud : float
        Optical depth of the cloud layer
        Units: None
        Low: 1e-6
        High: 1e2
    xi : float
        Exponent of the cloud layer
        Units: None
        Low: 0.1
        High: 10.0

    """

    f_cloud = g * xi * tau_cloud * P_layer ** (xi-1) / (k_cloud * (P_bottom**xi - P_top**xi))
    return f_cloud

def cloud_layer_infinite(P_layer, P_bottom, xi):
    """
    Calculates something using the cloud layer model, can see though it to the bottom

    Parameters
    ----------
    P_bottom : float
        Pressure at the bottom of the cloud layer
        Units: Pa
        Low: 1e2 Pa
        High: 1e8 Pa
    xi : float
        Exponent of the cloud layer
        Units: None
        Low: 0.1
        High: 10.0

    """

    f_cloud = g * xi * P_layer ** (xi-1) / (k_cloud * P_bottom**xi)
    return f_cloud

def parameterised_opacity(wave_grid, kappa_0, wv_0, p):
    """
    Calculates parameterised opacity

    Parameters
    ----------
    kappa_0 : float
        Something to do with opacity
        Units: cm^2/g
        Low: 10
        High: 1000
    wv_0 : float
        Wavelength at which kappa_0 is defined
        Units: um
        Low: 1e-2
        High: 1e2
    p : float
        Power law index
        Units: None
        Low: 0
        High: 4

    """

    kappa_abs = kappa_0 / (1 + (wave_grid / wv_0)**p)
    return kappa_abs

# def parameterised_refr_index(wave_grod, r_eff0, v_eff, gamma, n, k):

# def real_lab_materials(wave_grid, r_eff0, v_eff, gamma, compositons):

# What am I gonna do with mass fraction of cloud particles f_cloud?