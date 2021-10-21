#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:32:31 2020

@author: Simon Huber
"""

import numpy as np
from astropy import units as u
import cPickle as pickle
from astropy import constants as const
import scipy
import os





# It is not necessary to modify the following code, your adjustments can
# be made after the line "from here on please make your choices"
class flat_LCDM():
    def __init__(self,H_0=72 *u.km/u.megaparsec/u.s,Omega_m=0.26):
        self.H_0=H_0
        self.Omega_m=Omega_m
        self.Omega_vac=1-self.Omega_m 
        self.c=constants().c

    def f_coordinate_distance(self,z1,z2):
        def integrand(z):
            return 1/(np.sqrt((1+z)**3*self.Omega_m+self.Omega_vac)) 
        I=scipy.integrate.quad(integrand,z1,z2)
        coordinate_dis=(constants().c*I[0]/self.H_0).to(u.megaparsec)
        return coordinate_dis
        
    def f_angular_diameter_distance(self,z1,z2):
        coordinate_dis=self.f_coordinate_distance(z1,z2)
        ang_dia_dis=(coordinate_dis/(1+z2)).to(u.megaparsec)
        return ang_dia_dis
        
    def f_luminosity_distance(self,z1,z2):
        coordinate_dis=self.f_coordinate_distance(z1,z2)
        lum_dis=(coordinate_dis*(1+z2)).to(u.megaparsec)
        return lum_dis
    
class constants():
    def __init__(self):
        self.solar_lum=const.L_sun
        self.M_sun=const.M_sun
        self.c=const.c 
        self.G=const.G
        self.one_sigma=0.682689492
        self.two_sigma=0.954499736
        self.three_sigma=0.997300204
        
        
def f_modify_redshift(SNmicro_class,source_redshift_microlensing_calculation,source_redshift_output):

    SNmicro_class.redshift_included = source_redshift_output
    SNmicro_class.redshift_for_output_name = source_redshift_output
    
    #SNmicro_class.SN_I_dic = 0                              
    #SNmicro_class.SN_I_dic["I_lam"] = SNmicro_class.SN_I_dic["I_lam"]*(1+source_redshift_microlensing_calculation)**3 / (1+source_redshift_output)**3                       



    SNmicro_class.time_bin_center=SNmicro_class.time_bin_center/(1+source_redshift_microlensing_calculation)*(1+SNmicro_class.redshift_included)
    SNmicro_class.delta_t=SNmicro_class.delta_t/(1+source_redshift_microlensing_calculation)*(1+SNmicro_class.redshift_included)
    SNmicro_class.p_bin_center=SNmicro_class.p_bin_center*(1+source_redshift_microlensing_calculation)/(1+SNmicro_class.redshift_included)
    SNmicro_class.delta_p=SNmicro_class.delta_p*(1+source_redshift_microlensing_calculation)/(1+SNmicro_class.redshift_included) 
    SNmicro_class.p_max=SNmicro_class.p_max*(1+source_redshift_microlensing_calculation)/(1+SNmicro_class.redshift_included)

    SNmicro_class.lam_bin_center=SNmicro_class.lam_bin_center/(1+source_redshift_microlensing_calculation)*(1+SNmicro_class.redshift_included)

    SNmicro_class.delta_lam=SNmicro_class.delta_lam/(1+source_redshift_microlensing_calculation)*(1+SNmicro_class.redshift_included)
    SNmicro_class.R_bin_center=SNmicro_class.time_bin_center[:,None]*SNmicro_class.p_bin_center
    SNmicro_class.delta_R=SNmicro_class.time_bin_center[:,None]*SNmicro_class.delta_p
    SNmicro_class.R_max=SNmicro_class.R_max 

    SNmicro_class.time_bin_plot=np.arange(8,40)

    SNmicro_class.z_s=source_redshift_output
    SNmicro_class.D_s=flat_LCDM().f_angular_diameter_distance(0,SNmicro_class.z_s)
    SNmicro_class.D_ls=flat_LCDM().f_angular_diameter_distance(SNmicro_class.z_l,SNmicro_class.z_s)
    SNmicro_class.R_E=np.sqrt(4*constants().G*SNmicro_class.mean_mass*constants().M_sun/constants().c**2*SNmicro_class.D_ls*SNmicro_class.D_s/SNmicro_class.D_l).to(u.cm)
    SNmicro_class.pixelsize_micro_map=SNmicro_class.ss*SNmicro_class.R_E/SNmicro_class.res

    return SNmicro_class

def f_modify_microlensed_flux_approximately(flux,source_redshift_microlensing_calculation,source_redshift_output):
    if source_redshift_output == 0:
        ang_dis_output = 10*u.parsec    
    else:
        ang_dis_output = flat_LCDM().f_angular_diameter_distance(0,source_redshift_output)        
    ang_dis_microlensing_calculation = flat_LCDM().f_angular_diameter_distance(0,source_redshift_microlensing_calculation)

    
    flux_mod = flux* ang_dis_microlensing_calculation**2 / ang_dis_output**2*(1+source_redshift_microlensing_calculation)**5 / (1+source_redshift_output)**5      
    return flux_mod

    

class redshifted_mlcs():
    def __init__(self,N_sim,kappa,gamma,s,source_redshift_microlensing_calculation,source_redshift_output,lens_redshift):

        self.N_sim = N_sim
        self.kappa = kappa
        self.gamma = gamma
        self.s = s
        self.source_redshift_microlensing_calculation = source_redshift_microlensing_calculation
        self.source_redshift_output = source_redshift_output
        self.lens_redshift = lens_redshift
        if source_redshift_output >= 0.4:
            # for lower source redshift there is not enough data in the spectra available
            self.filters = ["u","g","r","i","z","y","J","H"]
        else:
            self.filters = ["u","g","r","i","z","y"]

        self.data_version = "IRreduced"
        self.time_bins = np.arange(6,44)
        self.SN_models = ["me", "n1", "ww", "su"]
    




    def f_get_microlensed_lightcurve(self):

        d_store_results = {}
        
        for supernova_model in self.SN_models:
            print supernova_model
            file_name = "%s_k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (supernova_model,
                                                                                   self.kappa,self.gamma,
                                                                                   self.s,source_redshift_microlensing_calculation,
                                                                                   self.lens_redshift,self.N_sim)
                
            with open("%sLSNeIa_class/%s.pickle"%(input_data_path,file_name),"rb") as handle:
                SNmicro = pickle.load(handle)
    
                    
            #changes the source redshift from source_redshift_microlensing_calculation to source_redshift_output
            #the microlensing was calculated for source_redshift_microlensing_calculation and can not be changed afterwards
            #therefore if source_redshift_microlensing_calculation not equal to source_redshift_output the microlensing
            #calculation is only approximated, because the size of the SN changes and therefore also the microlensing probability
            #However Huber et al 2020 (arXiv:2008.10393) in Figure 8 shows that there is only little dependcy of microlensing on the size of the SN
            SNmicro = f_modify_redshift(SNmicro_class=SNmicro,source_redshift_microlensing_calculation = source_redshift_microlensing_calculation,source_redshift_output=source_redshift_output)        
            SNmicro.time_bin_plot = self.time_bins
    
    

            d_mag_macro = {}
    
                                                     
            for filter_ in self.filters:
                d_mag_macro[filter_] = []
                # assumes LSST filters
                if filter_ == "J" or filter_ == "H" or filter_ == "K":
                    f_filter=SNmicro.f_filters(filter_)                
                else:
                    f_filter=SNmicro.f_filter_LSST(filter_)
    
                SNmicro.d_filter[filter_]=f_filter[0]
                SNmicro.d_plotmin[filter_]=np.where(SNmicro.lam_bin_center >= f_filter[1])[0][0]
                SNmicro.d_plotmax[filter_]=np.where(SNmicro.lam_bin_center <= f_filter[2])[0][-1] + 1  
                if SNmicro.d_plotmax[filter_] == SNmicro.amount_lam_bins:
                    raise ValueError("The SNe data you are using does not cover the wavelength range you require!")
    
            #open all pickels for d_flux
            name_new_directory = "k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (self.kappa,self.gamma,self.s,
                                                                                         self.source_redshift_microlensing_calculation,
                                                                                         self.lens_redshift,self.N_sim)
            data_input_folder = "%sspectra/%s/%s" %(input_data_path,supernova_model,name_new_directory) 
            d_flux_time_bin = {}
            for time_bin in self.time_bins:
                time_name_to_open_flux = ((SNmicro.time_bin_center[time_bin]).to(u.day)*(1+self.source_redshift_microlensing_calculation)/(1+self.source_redshift_output)).value
                pickel_name = "time_%.2f" % (time_name_to_open_flux)
                open_pickle = "%s/%s" %(data_input_folder,pickel_name)
                with open("%s.pickle" % open_pickle, "rb") as handle:
                    d_flux = pickle.load(handle)
                d_flux_time_bin[time_bin] = d_flux
    
            for config in range(0,self.N_sim):
            #for config in range(0,3): 
                d_mag = {}
                for filter_ in self.filters:
                    d_mag[filter_] = []      
                for time_bin in self.time_bins:
                                    
                    d_flux = d_flux_time_bin[time_bin]
    
                    if config == 0:
                        macrolensed_flux = d_flux["macro_flux"] 
                        macrolensed_flux = f_modify_microlensed_flux_approximately(flux = macrolensed_flux,
                                                                        source_redshift_microlensing_calculation = self.source_redshift_microlensing_calculation,
                                                                        source_redshift_output = self.source_redshift_output)
                        for filter_ in self.filters:
                            mag_macro = round(SNmicro.f_magnitude(macrolensed_flux,filter_).value,3) 
                            d_mag_macro[filter_].append(mag_macro)
    
                    for filter_ in self.filters:
                        key_micro = "micro_flux_%i" % config
                        microlensed_flux = d_flux[key_micro]
    
                        microlensed_flux = f_modify_microlensed_flux_approximately(flux = microlensed_flux,
                                                    source_redshift_microlensing_calculation = self.source_redshift_microlensing_calculation,
                                                    source_redshift_output = self.source_redshift_output)
                        mag = round(SNmicro.f_magnitude(microlensed_flux,filter_).value,3)                    
                        d_mag[filter_].append(mag)
    
                for filter_ in self.filters:
                    key_micro_light_curve = "micro_light_curve_%s%i%s" % (supernova_model,config,filter_)
                    d_store_results[key_micro_light_curve] = d_mag[filter_]
                    
            for filter_ in self.filters:
                key_macro_light_curve = "macro_light_curve_%s%s" % (supernova_model,filter_)
                d_store_results[key_macro_light_curve] = d_mag_macro[filter_]
    
            time_bin_center = SNmicro.time_bin_center[self.time_bins].to(u.day)
            d_store_results["time_bin_center"] = time_bin_center
            
        pickel_name = "k%f_g%f_s%.3f_redshift_source_%.3f_lens%.3f_Nsim_%i" % (self.kappa,self.gamma,self.s,self.source_redshift_output,
                                                                               self.lens_redshift,self.N_sim)
            
        output_folder = "%slight_curves/" %(output_data_path)
        try:
            os.mkdir(output_folder)
        except:
            pass

        safe_pickle = "%s%s" %(output_folder,pickel_name)
        with open(safe_pickle + ".pickle", "wb") as handle:
            pickle.dump(d_store_results, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    def f_get_microlensed_spectra(self):
        
        for supernova_model in self.SN_models:
            print supernova_model
            file_name = "%s_k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (supernova_model,
                                                                                   self.kappa,self.gamma,
                                                                                   self.s,source_redshift_microlensing_calculation,
                                                                                   self.lens_redshift,self.N_sim)
    
            with open("%sLSNeIa_class/%s.pickle"%(input_data_path,file_name),"rb") as handle:
                SNmicro = pickle.load(handle)
    
                    
            #changes the source redshift from source_redshift_microlensing_calculation to source_redshift_output
            #the microlensing was calculated for source_redshift_microlensing_calculation and can not be changed afterwards
            #therefore if source_redshift_microlensing_calculation not equal to source_redshift_output the microlensing
            #calculation is only approximated, because the size of the SN changes and therefore also the microlensing probability
            #However Huber et al 2020 (arXiv:2008.10393) in Figure 8 shows that there is only little dependcy of microlensing on the size of the SN
            SNmicro = f_modify_redshift(SNmicro_class=SNmicro,
                                        source_redshift_microlensing_calculation = source_redshift_microlensing_calculation,
                                        source_redshift_output=source_redshift_output)        
            SNmicro.time_bin_plot = self.time_bins

            #open all pickels for d_flux
            name_new_directory = "k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (self.kappa,self.gamma,self.s,
                                                                                         self.source_redshift_microlensing_calculation,
                                                                                         self.lens_redshift,self.N_sim)
            data_input_folder = "%sspectra/%s/%s" %(input_data_path,supernova_model,name_new_directory) 
            d_flux_time_bin = {}
            for time_bin in self.time_bins:
                time_name_to_open_flux = ((SNmicro.time_bin_center[time_bin]).to(u.day)*(1+self.source_redshift_microlensing_calculation)/(1+self.source_redshift_output)).value
                pickel_name = "time_%.2f" % (time_name_to_open_flux)
                open_pickle = "%s/%s" %(data_input_folder,pickel_name)
                with open("%s.pickle" % open_pickle, "rb") as handle:
                    d_flux = pickle.load(handle)
                d_flux_time_bin[time_bin] = d_flux
                

    
            system_name = "k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (self.kappa,self.gamma,self.s,self.source_redshift_output,
                                                                                   self.lens_redshift,self.N_sim)
    
            output_folder = "%sspectra/%s/%s" %(output_data_path,supernova_model,system_name)
            try:
                try:
                    os.mkdir("%sspectra/" %(output_data_path))
                except:
                    pass
                os.mkdir("%sspectra/%s/" %(output_data_path,supernova_model))
            except:
                pass
            try:
                os.mkdir(output_folder)
            except:
                pass       
        
            d_store_modified_flux = {}
            for time_bin in self.time_bins:
                for config in range(0,self.N_sim):
                #for config in range(0,3):                   
                    d_flux = d_flux_time_bin[time_bin]
    
                    if config == 0:
                        macrolensed_flux = d_flux["macro_flux"] 
                        macrolensed_flux = f_modify_microlensed_flux_approximately(flux = macrolensed_flux,
                                                                        source_redshift_microlensing_calculation = self.source_redshift_microlensing_calculation,
                                                                        source_redshift_output = self.source_redshift_output)
                        d_store_modified_flux["macro_flux"] = macrolensed_flux
                        
                    key_micro = "micro_flux_%i" % config
                    microlensed_flux = d_flux[key_micro]

                    microlensed_flux = f_modify_microlensed_flux_approximately(flux = microlensed_flux,
                                                source_redshift_microlensing_calculation = self.source_redshift_microlensing_calculation,
                                                source_redshift_output = self.source_redshift_output)
                    d_store_modified_flux[key_micro] = microlensed_flux

                d_store_modified_flux["lam_bin_center"] = SNmicro.lam_bin_center

                time_name_to_store_flux = ((SNmicro.time_bin_center[time_bin]).to(u.day)).value                
                pickel_name = "time_%.2f" % (time_name_to_store_flux)
                safe_pickle = "%s/%s" %(output_folder,pickel_name)
                with open(safe_pickle + ".pickle", "wb") as handle:
                    pickle.dump(d_store_modified_flux, handle, protocol=pickle.HIGHEST_PROTOCOL) 

            file_name_class = "%s_k%f_g%f_s%.3f_redshift_source%.3f_lens%.3f_Nsim_%i" % (supernova_model,self.kappa,self.gamma,
                                                                                   self.s,self.source_redshift_output,self.lens_redshift,self.N_sim)
            data_storage_folder = "%sLSNeIa_class/" %(output_data_path)
            try:
                os.mkdir(data_storage_folder)
            except:
                pass 
            with open(data_storage_folder + file_name_class + ".pickle", "wb") as handle:
                pickle.dump(SNmicro, handle, protocol=pickle.HIGHEST_PROTOCOL) 

"""           
import sys
system_number=int(sys.argv[1])
image_number=int(sys.argv[2])
"""

import public_spectra_light_curve

# from here on please make your choices

system_number=1
image_number=1



input_data_path = ".../data_release_holismokes7/" 
output_data_path = ".../data_release_holismokes7_output/" 

kappa, gamma, s, source_redshift_microlensing_calculation, lens_redshift = public_spectra_light_curve.f_get_system(system_number=system_number,image_number=image_number)


source_redshift_output = source_redshift_microlensing_calculation # exact microlensing calculation
source_redshift_output = 0 # rest frame, microlensing approximated see readme

mlc = redshifted_mlcs(N_sim = 10000, kappa = kappa,
                      gamma = gamma, s = s,
                      source_redshift_microlensing_calculation = source_redshift_microlensing_calculation,
                      source_redshift_output = source_redshift_output,
                      lens_redshift = lens_redshift)

mlc.f_get_microlensed_lightcurve()

mlc.f_get_microlensed_spectra()




