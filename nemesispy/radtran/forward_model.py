import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.utils import calc_mmw
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import calc_layer # average
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.radiance import calc_radiance, calc_planck
from nemesispy.radtran.read import read_cia
from nemesispy.radtran.trig import gauss_lobatto_weights, interpolate_to_lat_lon

from scipy.special import legendre
def calc_grav(h, lat, M_plt, R_plt,
    J2=0, J3=0, J4=0, flatten=0, rotation=1e5):
    """


    Parameters
    ----------
    h : TYPE
        DESCRIPTION.
    lat : TYPE
        DESCRIPTION.
    M_plt : TYPE
        DESCRIPTION.
    R_plt : TYPE
        DESCRIPTION.
    J2 : TYPE, optional
        DESCRIPTION. The default is 0.
    J3 : TYPE, optional
        DESCRIPTION. The default is 0.
    J4 : TYPE, optional
        DESCRIPTION. The default is 0.
    flatten : TYPE, optional
        DESCRIPTION. The default is 0.
    rotation : TYPE, optional
        DESCRIPTION. The default is 1e5.

    Returns
    -------
    gtot : TYPE
        DESCRIPTION.

    """

    G = 6.67430e-11
    xgm = G*M_plt
    xomega = 2.*np.pi/(rotation*24*3600)
    xellip = 1/(1-flatten)
    xcoeff = np.zeros(3)
    xcoeff[0] = J2
    xcoeff[1] = J3
    xcoeff[2] = J4
    xradius = R_plt


    # account for latitude dependence
    lat = 2 * np.pi * lat/360 # convert to radiance
    ## assume for now that we use planetocentric latitude to begin with?
    # latc = np.arctan(np.tan(lat)/xellip**2.)   #Converts planetographic latitude to planetocentric
    # slatc = np.sin(latc)
    # clatc = np.cos(latc)
    slat = np.sin(lat)
    clat = np.cos(lat)
    # calculate the ratio of radius at equator to radius at current latitude
    Rr = np.sqrt(clat**2 + (xellip**2. * slat**2))
    # print('Rr',Rr)
    r = (h+xradius)/Rr
    radius = (xradius/Rr)

    # calculate legendre polynomials
    pol = np.zeros(6)
    for i in range(6):
        Pn = legendre(i+1)
        pol[i] = Pn(slat)
    # print('pol',pol)

    # Evaulate radial contribution from summation for first three terms
    # then suubtract centrifugal effects

    # note that if Jcoeffs are 0 then the following calculation can be
    # skipped, i.e. g = 1 if Jcoeffs are 0 and period is 1e5 (i.e. unknown)
    g = 1
    for i in range(3):
        ix = i + 1
        g = g - ((2*ix+1)*Rr**(2*ix)*xcoeff[ix-1]*pol[2*ix-1])

    gradial = (g*xgm/r**2) - (r*xomega**2 * clat**2.)
    # print('graial',gradial)

    # Evaluate latitudinal conttribution for first three terms,
    # then add centriifugal effects

    gtheta1 = 0.
    for i in range(3):
        ix = i+1
        gtheta1 = gtheta1 \
        -(4*ix**2*Rr**(2*ix)*xcoeff[ix-1]*(pol[2*ix-1-1]-slat*pol[2*ix-1])/clat)

    # print('xomega',xomega)
    # print('gtheta1',gtheta1)
    gtheta = (gtheta1 * xgm/r**2) + (r*xomega**2 * clat * slat)

    # combine the two components
    gtot = np.sqrt(gradial**2 + gtheta**2)

    return gtot

class ForwardModel():

    def __init__(self):
        """
        Attributes to store data that doesn't change during a retrieval
        """
        # planet and planetary system data
        self.M_plt = None
        self.R_plt = None
        self.M_star = None # currently not used
        self.R_star = None # currently not used
        self.T_star = None # currently not used
        self.semi_major_axis = None # currently not used
        self.NLAYER = None
        self.is_planet_model_set = False

        # opacity data
        self.gas_id_list = None
        self.iso_id_list = None
        self.wave_grid = None
        self.g_ord = None
        self.del_g = None
        self.k_table_P_grid = None
        self.k_table_T_grid = None
        self.k_gas_w_g_p_t = None
        self.cia_nu_grid = None
        self.cia_T_grid = None
        self.k_cia_pair_t_w = None
        self.is_opacity_data_set = False

        # debug data
        self.U_layer = None
        self.del_S = None
        self.del_H = None

    def set_planet_model(self, M_plt, R_plt, gas_id_list, iso_id_list, NLAYER,
        R_star=None, T_star=None, semi_major_axis=None):
        """
        Set the basic system parameters for running the
        """
        self.M_plt = M_plt
        self.R_plt = R_plt
        self.R_star = R_star
        self.T_star = T_star
        self.semi_major_axis = semi_major_axis
        self.gas_id_list = gas_id_list
        self.iso_id_list = iso_id_list
        self.NLAYER = NLAYER

        self.is_planet_model_set = True

    def set_opacity_data(self, kta_file_paths, cia_file_path):
        """
        Read gas ktables and cia opacity files and store as class attributes.
        """
        gas_id_list, iso_id_list, wave_grid, g_ord, del_g, k_table_P_grid,\
            k_table_T_grid, k_gas_w_g_p_t = read_kls(kta_file_paths)
        """
        Some gases (e.g. H2 and He) have no k table data so gas id lists need
        to be passed somewhere else.
        """
        # self.gas_id_list = gas_id_list
        # self.iso_id_list = iso_id_list
        self.wave_grid = wave_grid
        self.g_ord = g_ord
        self.del_g = del_g
        self.k_table_P_grid = k_table_P_grid
        self.k_table_T_grid = k_table_T_grid
        self.k_gas_w_g_p_t = k_gas_w_g_p_t

        cia_nu_grid, cia_T_grid, k_cia_pair_t_w = read_cia(cia_file_path)
        self.cia_nu_grid = cia_nu_grid
        self.cia_T_grid = cia_T_grid
        self.k_cia_pair_t_w = k_cia_pair_t_w

        self.is_opacity_data_set = True

    def test_point_spectrum(self,U_layer,P_layer,T_layer,VMR_layer,del_S,
        scale,solspec,path_angle=None):
        """
        wrapper for calc_radiance
        """
        # print('test scale',scale)
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, DEL_S=del_S)
        # debug
        # print('P_layer',P_layer)
        # print('T_layer',T_layer)
        # print('U_layer',U_layer)
        # print('del_S',del_S)
        return point_spectrum

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model, path_angle,
        solspec=None):
        """
        First get layer properties from model inputs
        Then calculate the spectrum at a single point on the disc.
        """

        H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S,del_H\
            = calc_layer(self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1, H_0=0.0, NSIMPS=101)

        if solspec.any() == None:
            solspec = np.ones(len(self.wave_grid))
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, DEL_S=del_H)
        self.U_layer = U_layer
        self.del_S = del_S
        self.del_H = del_H
        return point_spectrum

    def calc_disc_spectrum(self,phase,nmu,global_H_model,global_P_model,
        global_T_model,global_VMR_model,global_model_longitudes,
        global_model_lattitudes,solspec=None):

        nav, wav = gauss_lobatto_weights(phase, nmu)
        # print('calc_disc_spectrum')
        # print('nav, wav',nav,wav)
        """Want a monotonic array for interpolation"""
        fov_longitudes = wav[1,:]
        # convert to [-180,180]
        for index, ilon in enumerate (fov_longitudes):
            if ilon>180:
                fov_longitudes[index] = ilon - 360

        fov_lattitudes = wav[0,:]
        fov_emission_angles = wav[3,:]
        fov_weights = wav[-1,:]

        # n_fov_loc = len(fov_longitudes)
        fov_locations = np.zeros((nav,2))
        fov_locations[:,0] = fov_longitudes
        fov_locations[:,1] = fov_lattitudes

        fov_H_model = interpolate_to_lat_lon(fov_locations, global_H_model,
            global_model_longitudes, global_model_lattitudes)

        fov_P_model = interpolate_to_lat_lon(fov_locations, global_P_model,
            global_model_longitudes, global_model_lattitudes)

        fov_T_model = interpolate_to_lat_lon(fov_locations, global_T_model,
            global_model_longitudes, global_model_lattitudes)

        fov_VMR_model = interpolate_to_lat_lon(fov_locations, global_VMR_model,
            global_model_longitudes, global_model_lattitudes)

        disc_spectrum = np.zeros(len(self.wave_grid))

        total_weight = 0
        for ilocation in range(nav):
            # print('ilocation',ilocation)
            H_model = fov_H_model[ilocation]
            # print('fov_H_model',fov_H_model)
            # print('H_model', H_model)
            P_model = fov_P_model[ilocation]
            # print('P_model',P_model)
            T_model = fov_T_model[ilocation]
            VMR_model = fov_VMR_model[ilocation]
            path_angle = fov_emission_angles[ilocation]
            weight = fov_weights[ilocation]
            point_spectrum = self.calc_point_spectrum(
                H_model, P_model, T_model, VMR_model, path_angle,
                solspec=solspec)
            total_weight += weight
            # print('H_model',H_model)
            # print('P_model',P_model)
            # print('T_model',T_model)
            # print('VMR_model',VMR_model)
            # print('path_angle',path_angle)
            # print('point_spectrum',point_spectrum)
            # print('weight',weight)
            # print('total_weight',total_weight)
            disc_spectrum += point_spectrum * weight

        return disc_spectrum

    def run_point_spectrum(self, H_model, P_model, T_model,\
            VMR_model, path_angle, solspec=None):
        if self.is_planet_model_set == False:
            raise('Planet model has not been set yet')
        elif self.is_opacity_data_set == False:
            raise('Opacity data has not been set yet')
        point_spectrum = self.calc_point_spectrum(H_model, P_model, T_model,\
            VMR_model, path_angle, solspec=solspec)
        return point_spectrum

    """TBD"""
    def run_disc_spectrum(self,phase,nmu,global_H_model,global_P_model,
        global_T_model,global_VMR_model,global_model_longitudes,
        global_model_lattitudes,solspec=None):

        if self.is_planet_model_set == False:
            raise('Planet model has not been set yet')

        elif self.is_opacity_data_set == False:
            raise('Opacity data has not been set yet')

        disc_spectrum = self.calc_disc_spectrum(phase,nmu,global_H_model,
            global_P_model, global_T_model,global_VMR_model,global_model_longitudes,
            global_model_lattitudes,solspec=solspec)

        return disc_spectrum