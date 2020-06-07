# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:05:21 2020

@author: leezi
"""
from matplotlib.widgets import Cursor
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy import units as un
import astropy.constants as const
from scipy.signal import fftconvolve, gaussian
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from . import fit as ft
mpl.use('Tkagg')

class spec_data(object):
    
    def __init__(self, name):
        self.name = name
        self.spec = pd.DataFrame
        
    def add_spec(self, wavelength, flux, noise, z):
        
        wavelength = np.array(wavelength)
        flux = np.array(flux)
        noise = np.array(noise)
        
        self.spec.wavelength = wavelength
        self.spec.flux = np.array(flux)
        self.spec.noise = np.array(noise)
        self.spec.z = z
        self.spec.restframe = wavelength/(1+z)

    def select_range(self,line,span=3):
        
        linedata = pd.read_csv('lines.csv')
        idx = linedata['lines']==line
        center = float(linedata['wavelength'][idx]) 
        idx2 = np.where((self.spec.restframe-center<=span)&(self.spec.restframe-center>=-span))
        
        return idx2
    
    def plot_spec(self,line,span=3):
#         plt.ion()
        idx = self.select_range(line,span)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.plot(self.spec.restframe[idx],self.spec.flux[idx])
#         plt.show()
#         plt.ioff()
        return ax
    
    def select_component(self,line,span=3):
        ax = self.plot_spec(line,span)
        ax.set_title('Mark center(s) of component(s)',color = 'red')
        cursor = Cursor(plt.gca(), horizOn=False, color='r', lw=1)
        pos = plt.ginput(0,0)
        pos = np.array(pos)
        
#         component_center = pos[:,0]
#         ax.vlines(component_center,0,1,color='r',linestyle=':')
#         plt.show()
        return pos[:,0]
    
    def add_mask(self,line,span=3):
        
        ax = self.plot_spec(line,span)
        ax.set_title('Choose two points to mask bad data \n (press enter to exit if no data need to be masked)',color = 'red')
        cursor = Cursor(plt.gca(), horizOn=False, color='r', lw=1)
        idx = np.array([])
        
        while True:
            pos= plt.ginput(2,0)
            if pos != []:
                mask_x = np.array(pos)[:,0]

                idx_temp = np.where((self.spec.restframe>min(mask_x))&(self.spec.restframe<max(mask_x)))
                idx = np.append(idx,idx_temp).astype('int')
                key = input('press e to save mask and exit \n press c to cancel mask \n press any key to continue masking')
                if key != 'e':
                    continue
                else:
                    self.spec.flux[idx] = 1
#                     ax = self.plot_spec(line,span)
#                     ax.set_title('Masked spectrum')
                    break
            else:
                plt.close()
                break

    def initial_guess(self,line,span=3,sigma=20,N=1e14):
        lam_center = self.select_component(line,span)
        p = np.zeros((len(lam_center),5))
        p[:,0] = lam_center
        p[:,1:3] = sigma
        p[:,3:5] = N
        return p
    
    def fit(self,line,span=3,sigma=20,N=1e14,mask=False):
        
        idx = self.select_range(line,span)
        wave = self.spec.restframe[idx]
        f = self.spec.flux[idx]
        noise = self.spec.noise[idx]
        lam = np.linspace(min(wave),max(wave),500)
        
        if mask:
            self.add_mask('C IV',span)

        plt.close()
        paras = self.initial_guess(line,span,sigma,N)
        p = paras.flatten()
        bounds = ft.set_bounds(paras)
        popt,pcov = curve_fit(ft.multicomponet,wave,f,p0=p, bounds = bounds,sigma=noise)
        paras_fit = popt.reshape(len(popt)//5,5)
        plt.vlines(paras_fit[:,0],1,1.08,color = 'blue',linestyle='--')
        plt.vlines(paras_fit[:,0]+2.575,1,1.08,color = 'blue',linestyle='--')
        print('N1=%.3f'%np.log10(sum(paras_fit[:,3])))
        print('N2=%.3f'%np.log10(sum(paras_fit[:,4])))
        plt.plot(lam,ft.multicomponet(lam,*popt))
        plt.show()

        return paras_fit,pcov
        

