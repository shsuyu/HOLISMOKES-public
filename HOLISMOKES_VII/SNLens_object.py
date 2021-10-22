# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:33:16 2018

@author: shuber
"""

import numpy as np
from scipy import interpolate
from astropy import units as u
from astropy import constants as const

class constants():
    def __init__(self):
        self.solar_lum=const.L_sun
        self.M_sun=const.M_sun
        self.c=const.c 
        self.G=const.G
        self.one_sigma=0.682689492
        self.two_sigma=0.954499736
        self.three_sigma=0.997300204
        
class SNmicrolensing():
    def __init__(self):
        #variables for microlensing map
        self.kappa = None
        self.gamma = None
        self.mass_function = None
        self.s = None
        self.res = None
        self.ss = None
        self.mean_mass=None
        self.z_l=None
        self.z_s=None
        self.D_l=None
        self.D_s=None
        self.D_ls=None
        self.R_E=None
        self.pixelsize_micro_map=None
        self.macro_magnification=None
        self.microlensing_map=None
        #variables for specific intsity
        self.redshift_included=0
        self.time_bin_center=None
        self.delta_t=None
        self.p_bin_center=None
        self.delta_p=None
        self.p_max=None
        self.lam_bin_center=None
        self.delta_lam=None        
        self.R_bin_center=None
        self.delta_R=None
        self.R_max=None
        
        self.SN_I_dic=None
        self.max_deviation_polar_cartesian=None
        self.time_bin_plot=None
        self.s_SN_model=None
        self.Dataname=None
        self.d_threshold_R={"me": 4., "n1": 4., "w7": 4, "SEDONA": 2, "su": 4., "77": 4., "ww": 4.}        

        self.resolution_plot=200 #for plt.savefig()
        self.string_choosen_filter="LSST"
        self.d_zero_points = {"u":-14,"g":-14,"r":-14,"i":-14,"z":-14,"y":-14} 
        self.string_choosen_magnitude="AB" 
        self.mapsize=2600
        self.mapsize=650
        self.redshift_for_output_name=0.0
        self.d_SN_names = {"me": "Merger", "w7": "W7", "n1": "N100", "su": "sub-Ch"}

        self.d_filter={}
        self.d_plotmin={}
        self.d_plotmax={} 

    def f_filter_LSST(self,filter_key,multiplicator = 1.):
    #this functions returns a function which contains the throughput for LSST filters as a function of wavelength where wavelength has to be in angstrom
    #in additon this functions returns the min and max value of the interpolation region
    #filter_key has to be a string and can be u,g,r,i,z,y    
    #the multiplicator is there for testing if redshift is included correctly
        filter_data  = open("./filter_information/total_"+str(filter_key)+".dat","r") 
        counter=0
        l_lam=[]
        l_filter=[]
        for line in filter_data:
            if counter > 6: #first seven lines contain something else  
                numbers_str = line.split() #numbers_str[0] gives one number as type string 
                numbers_float = [float(x) for x in numbers_str] #converts strings into floats
                l_lam.append(numbers_float[0]*10 * multiplicator) #times 10 so it is in angstrom
                l_filter.append(numbers_float[-1]) 
            counter +=1
        a_lam=np.asarray(l_lam)
        interp_min=a_lam[0]*u.angstrom
        interp_max=a_lam[-1]*u.angstrom
        a_filter=np.asarray(l_filter)
        f=interpolate.interp1d(a_lam,a_filter,kind="linear")
        return (f,interp_min,interp_max)
    
    def f_filters(self,filter_key):
    #in additon this functions returns the min and max value of the interpolation region
    #filter_key has to be a string and can be u,g,r,i,z,y    
    #the multiplicator is there for testing if redshift is included correctly
        filter_data  = open("./filter_information/total_"+str(filter_key)+".dat","r") 
        l_lam=[]
        l_filter=[]
        for line in filter_data:
                numbers_str = line.split() #numbers_str[0] gives one number as type string 
                numbers_float = [float(x) for x in numbers_str] #converts strings into floats
                l_lam.append(numbers_float[0]) 
                l_filter.append(numbers_float[-1]) 
        a_lam=np.asarray(l_lam)
        interp_min=a_lam[0]*u.angstrom
        interp_max=a_lam[-1]*u.angstrom
        a_filter=np.asarray(l_filter)
        f=interpolate.interp1d(a_lam,a_filter,kind="linear")
        return (f,interp_min,interp_max)

    def f_magnitude(self,flux_lambda,filter_band):
    #function to calculate magnituds in the vega or AB system system. 
    #flux_lambda has units erg/s/cm^2/A and same shape as self.lam_bin_center
        filter_function=self.d_filter[filter_band]
        plotmin=self.d_plotmin[filter_band]
        plotmax=self.d_plotmax[filter_band]   
        if plotmax >= self.amount_lam_bins:
            raise ValueError("The SNe data you are using does not cover the wavelength range you require! Use calculate_I2_red.py to calculate a new file with higher wavelength range!")
        
        if self.string_choosen_magnitude == "AB":
            flux_lambda_filter=filter_function(self.lam_bin_center[plotmin:plotmax])*flux_lambda[plotmin:plotmax].to(u.erg/u.s/u.cm**2/u.angstrom)    
            flux_filter=np.sum(flux_lambda_filter*self.lam_bin_center[plotmin:plotmax].to(u.angstrom)*self.delta_lam[plotmin:plotmax].to(u.angstrom),axis=0)
            normalization=np.sum(filter_function(self.lam_bin_center[plotmin:plotmax]) * const.c*self.delta_lam[plotmin:plotmax].to(u.angstrom)/self.lam_bin_center[plotmin:plotmax].to(u.angstrom))
            #normalization=1*u.angstrom/u.s
            mag=-2.5*np.log10(flux_filter/normalization/u.erg*u.cm**2)-48.6
        elif self.string_choosen_magnitude == "Vega": 
            flux_lambda_filter=filter_function(self.lam_bin_center[plotmin:plotmax])*flux_lambda[plotmin:plotmax]    
            flux_filter=np.sum(flux_lambda_filter*self.delta_lam[plotmin:plotmax],axis=0)
            mag=-2.5*np.log10(flux_filter/(u.erg/u.centimeter**2/u.s))+self.d_zero_points[filter_band]
        
        return mag
        
