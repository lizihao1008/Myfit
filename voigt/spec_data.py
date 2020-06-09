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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
        self.spec.flux_origin = np.array(flux)
        
    def select_range(self,line,span=3):
        
        linedata = pd.read_csv('lines.csv')
        idx = linedata['lines']==line
        center = float(linedata['wavelength'][idx]) 
        idx2 = np.where((self.spec.restframe-center<=span)&(self.spec.restframe-center>=-span))
        
        return idx2
    
    def plot_spec(self,line,span=3,out=False):
        mpl.use('tkagg')
        idx = self.select_range(line,span)
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.plot(self.spec.restframe[idx],self.spec.flux[idx],color='black',drawstyle='steps-mid')
        if out: 
            return ax,self.spec.restframe[idx],self.spec.flux[idx]
        else:
            return
    def select_component(self,line,span=3):
        ax,_,_ = self.plot_spec(line,span,out=True)
        ax.set_title('Mark center(s) of component(s)',color = 'red')
        cursor = Cursor(plt.gca(), horizOn=False, color='r', lw=1)
        pos = plt.ginput(0,0)
        pos = np.array(pos)
        
#         component_center = pos[:,0]
#         ax.vlines(component_center,0,1,color='r',linestyle=':')
#         plt.show()
        return pos[:,0]
    
    def add_mask(self,line,span=3):
        
        ax,_,_ = self.plot_spec(line,span,out=True)
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
            return
        
    def convolve_resolution(self,resolution):
        if isinstance(resolution,int):
            kernel = 1549*(1+self.spec.z)/resolution/2.35482
            a = int(10*kernel) + 21
            LSF = gaussian(a, kernel)
            LSF = LSF/LSF.sum()
            profile_broad = fftconvolve(self.spec.flux_origin, LSF, 'valid')
            self.spec.flux = np.interp(self.spec.restframe, self.spec.restframe[a//2:-a//2+1], profile_broad,left=1,right=1)
        else:
            raise TypeError('Type of resoluton should be integer')
        return
    
    def reset_resolution(self):
        self.spec.flux = self.spec.flux_origin
        return
    
    def initial_guess(self,line,components='ui',span=3,sigma=20,N=1e14):
        
        if components is 'ui':
            lam_center = self.select_component(line,span)
        else:
            lam_center = np.array(components)
        p = np.zeros((len(lam_center),5))
        p[:,0] = lam_center
        p[:,1:3] = sigma
        p[:,3:5] = N
        return p
    
    def fit(self,line,resolution=False,components='ui',span=3,sigma=20,N=1e14,mask=False,plot=True,print_result=True):
        if resolution:
            if isinstance(resolution,int):
                global r,z
                r = resolution
                z = self.spec.z
            else:
                raise TypeError('Type of resoluton should be integer')
        else:
            r = False
        idx = self.select_range(line,span)
        wave = self.spec.restframe[idx]
        f = self.spec.flux[idx]
        noise = self.spec.noise[idx]
        lam = np.linspace(min(wave),max(wave),500)
        
        if mask:
            self.add_mask('C IV',span)
            plt.close()
        paras = self.initial_guess(line,components,span,sigma,N)
        # print(paras)
        p = paras.flatten()
        bounds = ft.set_bounds(paras)
        popt,pcov = curve_fit(ft.multicomponet,wave,f,p0=p, bounds = bounds,sigma=noise)
        paras_fit = popt.reshape(len(popt)//5,5)
        
        if print_result:
            print('lg(N1)(total)=%.3f'%np.log10(sum(paras_fit[:,3])))
            print('lg(N2)(total)=%.3f'%np.log10(sum(paras_fit[:,4])))
            print('--------------------')
            for i in range(len(paras_fit)):
                print('component %d:\n sigma1=%.3f sigma2=%.3f lg(N1)=%.3f lg(N2)=%.3f'
                      %(i+1,paras_fit[i,1],paras_fit[i,2],np.log10(paras_fit[i,3]),np.log10(paras_fit[i,4])))
                print('--------------------')
        if plot:
            if components != 'ui':
                # mpl.use('agg')
                ax = self.plot_spec(line,span)
            plt.vlines(paras_fit[:,0],1,1.08,color = 'blue',linestyle='--')
            plt.vlines(paras_fit[:,0]+2.575,1,1.08,color = 'blue',linestyle='--')
    
    
            plt.plot(lam,ft.multicomponet(lam,*popt))
            plt.show()

        return paras_fit,pcov,paras[:,0]
    

# z = z
# r = r

