#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:36:58 2021

@author: Simon Huber
"""

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from astropy import units as u

def f_get_system(system_number,image_number):
    # system_number_my OM10 version: 1_187, 2_18701, 3_18702, 4_18703, 5_18704,
    # 6_187, 7_187, 8_187, 9_18707
    
    sn = system_number

    if sn == 1 or sn == 2 or sn == 3 or sn == 4 or sn == 5:
        s = 0.600
        if image_number == 1:        
            kappa = 0.250895
            gamma = 0.274510
        elif image_number == 2:
            kappa = 0.825271
            gamma = 0.814777
        if sn == 1:
            source_redshift_microlensing_calculation = 0.76
            lens_redshift = 0.252
        elif sn == 2:
            source_redshift_microlensing_calculation = 0.55
            lens_redshift = 0.252
        elif sn == 3:
            source_redshift_microlensing_calculation = 0.99
            lens_redshift = 0.252
        elif sn == 4:
            source_redshift_microlensing_calculation = 0.76
            lens_redshift = 0.16
        elif sn == 5:
            source_redshift_microlensing_calculation = 0.76
            lens_redshift = 0.48
            
    if sn == 6 or sn == 7 or sn == 8:
        source_redshift_microlensing_calculation = 0.76
        lens_redshift = 0.252
        if image_number == 1:        
            kappa = 0.250895
            gamma = 0.274510
        elif image_number == 2:
            kappa = 0.825271
            gamma = 0.814777
        if sn == 6:
            s = 0.3
        elif sn == 7:
            s = 0.59
        elif sn == 8:
            s = 0.9
            
    if sn == 9:
        source_redshift_microlensing_calculation = 0.76
        lens_redshift = 0.252
        s = 0.6
        if image_number == 1:        
            kappa = 0.434950 
            gamma = 0.414743
        elif image_number == 2:
            kappa = 0.431058 
            gamma = 0.423635
        if image_number == 3:        
            kappa = 0.566524 
            gamma = 0.536502
        elif image_number == 4:
            kappa = 1.282808 
            gamma = 1.252791
            
    return kappa,gamma,s,source_redshift_microlensing_calculation,lens_redshift

class mlcs():
    def __init__(self,supernova_model,N_sim,kappa,gamma,s,source_redshift,lens_redshift):

        self.supernova_model = supernova_model
        self.N_sim = N_sim
        self.kappa = kappa
        self.gamma = gamma
        self.s = s
        self.source_redshift = source_redshift
        self.lens_redshift = lens_redshift

        self.data_version = "IRreduced"
        self.time_bins = np.arange(6,44)
        
    def _f_get_light_curve_dic(self):
        pickel_name = "k%f_g%f_s%.3f_redshift_source_%.3f_lens%.3f_Nsim_%i" % (self.kappa,self.gamma,self.s,self.source_redshift,
                                                                               self.lens_redshift,self.N_sim)

        input_folder = "%slight_curves/" %(input_data_path)

        open_pickle = "%s%s" %(input_folder,pickel_name)
        with open("%s.pickle" % (open_pickle),"rb") as handle:
            d_light_curves = pickle.load(handle)
            
        return d_light_curves
    
    def f_load_microlensed_lightcurve(self,filter_,micro_config):
                
        d_light_curves = self._f_get_light_curve_dic()
            
        key_micro_light_curve = "micro_light_curve_%s%i%s" % (self.supernova_model,micro_config,filter_)
        
        magnitude = d_light_curves[key_micro_light_curve]
        
        time = d_light_curves["time_bin_center"]
        
        return time, magnitude
    
    def f_load_macrolensed_lightcurve(self,filter_):
                
        d_light_curves = self._f_get_light_curve_dic()

        key_macro_light_curve = "macro_light_curve_%s%s" % (supernova_model,filter_)
        
        magnitude = d_light_curves[key_macro_light_curve]
        
        time = d_light_curves["time_bin_center"]
        
        return time, magnitude
    
    def _f_get_flux_dic(self,time_bin):
        file_name = "%s_k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (self.supernova_model,
                                                                               self.kappa,self.gamma,
                                                                               self.s,source_redshift,
                                                                               self.lens_redshift,self.N_sim)
        
        
        with open("%sLSNeIa_class/%s.pickle"%(input_data_path,file_name),"rb") as handle:
            SNmicro = pickle.load(handle)      
            
        #open all pickels for d_flux
        name_new_directory = "k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (self.kappa,self.gamma,self.s,
                                                                                     self.source_redshift,
                                                                                     self.lens_redshift,self.N_sim)
        
        data_input_folder = "%sspectra/%s/%s" %(input_data_path,self.supernova_model,name_new_directory) 
        time = SNmicro.time_bin_center[time_bin].to(u.day)
        time_name_to_open_flux = time.value
        pickel_name = "time_%.2f" % (time_name_to_open_flux)
        open_pickle = "%s/%s" %(data_input_folder,pickel_name)
        print open_pickle
        with open("%s.pickle" % open_pickle, "rb") as handle:
            d_flux = pickle.load(handle)
            
        return d_flux, time
        
        
    def f_load_microlensed_flux(self,micro_config,time_bin):
        
        d_flux, time = self._f_get_flux_dic(time_bin=time_bin)
  
        key_micro = "micro_flux_%i" % micro_config
        
        flux = d_flux[key_micro]
        
        wavelength = d_flux["lam_bin_center"]
        
        return wavelength, flux, time
    
    def f_load_macrolensed_flux(self,time_bin):
        
        d_flux, time = self._f_get_flux_dic(time_bin=time_bin)
  
        key_micro = "macro_flux"
        
        flux = d_flux[key_micro]
        
        wavelength = d_flux["lam_bin_center"]
        
        return wavelength, flux, time
            
            
if __name__ == "__main__":      
    system_number=9 # for the options see the readme file
    image_number=4# for the options see the readme file
    supernova_model="me" # available options "me", "n1", "su", "ww"
    
    input_data_path = "./data_release_holismokes7/" 
    
    kappa, gamma, s, source_redshift, lens_redshift = f_get_system(system_number=system_number,image_number=image_number)
    
    #source_redshift = 0 # modify this if you have produced data via public_redshifted_spectra_light_curve.py
    
    mlc = mlcs(supernova_model = supernova_model, N_sim = 10000, kappa = kappa,
               gamma = gamma, s = s, source_redshift = source_redshift,
               lens_redshift = lens_redshift)

    # get microlensed and macrolensed light curves
    micro_config = 9999 # random micro position, values from 0 to 9999
    filter_ = "g" # available filters "u","g","r","i","z","y","J","H"
    time, magnitude_micro = mlc.f_load_microlensed_lightcurve(filter_ = filter_, micro_config = micro_config)
    
    time, magnitude_macro = mlc.f_load_macrolensed_lightcurve(filter_ = filter_)
    
    plt.figure(1)
    plt.title("light curves")
    plt.plot(time,magnitude_micro,label = "micro, config %i" % micro_config)
    plt.plot(time,magnitude_macro,label = "macro")
    plt.xlabel("time after explosion [%s] (observer frame)" % time.unit)
    plt.ylabel("magnitude")
    plt.legend()
    
    # get microlensed and macrolensed spectra
    micro_config = 9999 # random micro position, values from 0 to 9999
    time_bin = 6 #use values from 6 to 43
    wavelength, flux_micro, time_spectra_micro = mlc.f_load_microlensed_flux(micro_config = micro_config, time_bin = time_bin)
    
    wavelength, flux_macro, time_spectra_macro = mlc.f_load_macrolensed_flux(time_bin = time_bin)

    plt.figure(2)
    plt.title("spectra taken at %.1f %s after explosion" % (time_spectra_micro.value,time_spectra_micro.unit))
    plt.plot(wavelength, flux_micro, label = "micro, config %i" % micro_config)
    plt.plot(wavelength, flux_macro, label = "macro")
    plt.xlabel("wavelength [%s]" % wavelength.unit)
    plt.ylabel("flux [%s]" % flux_micro.unit)
    plt.legend()


