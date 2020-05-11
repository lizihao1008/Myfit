# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:28:23 2020

@author: leezi
"""
__version__ = '0.1'

import pandas as pd
from astropy.io import fits
from astropy import units as un
import astropy.constants as const
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve, gaussian
import numpy as np
import matplotlib.pyplot as plt

def readdr7(file):
    
    s = fits.open(file)
    z = s[0].header['z']
    c0=s[0].header['COEFF0']
    c1=s[0].header['COEFF1']
    
    flux=s[0].data[0]*un.erg/un.AA/un.cm**2/un.s
    wavelength = 10.**(c0 + c1 * np.arange(1,len(flux)+1))*un.AA
    restframe = wavelength/(1+z)
    
    spec = pd.DataFrame
    spec.flux = flux
    spec.wavelength = wavelength
    spec.restframe = restframe
    spec.z = z
    spec.radv = 299792.458*un.km/un.s*z/(1+z)
    spec.data = s[2].data
    spec.name = s[0].header['NAME']
    spec.noise = s[0].data[2]
    spec.name = file.split('\\')[-1]
    s.close()
    
    return spec 

#The function of theoretical relationship among flux, velocity dispersion and column density.
def Ilam(lam,dlam,sigma1,sigma2,N1,N2):
    
    lam = (lam-dlam)/1e8 #Transfer angstrom to cm to be compatible with Gaussian units
    lam1 = 1548.2/1e8
    lam2 = 1550.77/1e8
    
    f1 = 0.189
    f2 = 0.095
    
    gamma1 = 2.643E+08
    gamma2 = 2.628E+08

    sigma1 = sigma1*1e5 #Transfer km/s to cm/s
    sigma2 = sigma2*1e5
    
    a1 = gamma1*lam1/4/np.pi/sigma1/2**0.5
    a2 = gamma2*lam2/4/np.pi/sigma2/2**0.5
    
    c = const.c.to('cm/s').value
    u1 = c*(lam**2-lam1**2)/(lam**2+lam1**2)/sigma1/2**0.5
    u2 = c*(lam**2-lam2**2)/(lam**2+lam2**2)/sigma2/2**0.5
    
    h1 = np.exp(-u1**2)
    h2 = np.exp(-u2**2)
    
    H1 = h1-a1/u1**2/np.pi**0.5*(h1**2*(4*u1**4+7*u1**2+4+1.5*u1**(-2))-1.5*u1**(-2)-1)
    H2 = h2-a2/u2**2/np.pi**0.5*(h2**2*(4*u2**4+7*u2**2+4+1.5*u2**(-2))-1.5*u2**(-2)-1)
    
    phi1 = c/lam1/sigma1/(2*np.pi)**0.5*H1
    phi2 = c/lam2/sigma2/(2*np.pi)**0.5*H2
    
    tau1 = (np.pi*const.e.esu**2/const.m_e.to('g')/c**2).value*f1*lam1**2*phi1*N1
    tau2 = (np.pi*const.e.esu**2/const.m_e.to('g')/c**2).value*f2*lam2**2*phi2*N2
    
    I = np.e**(-tau1-tau2)
    kernel = 2/2.35482
    a = int(10*kernel) + 21
    LSF = gaussian(a, kernel)
    LSF = LSF/LSF.sum()
    profile_broad = fftconvolve(I, LSF, 'valid')
    profile_obs = np.interp(lam, lam[a//2:-a//2+1], profile_broad,left=1,right=1)
    
    return profile_obs

def select_range(line,spec):
    linedata = pd.read_csv('lines.csv')
    idx = linedata['lines']==line
    center = float(linedata['wavelength'][idx]) 
    idx2 = np.where((spec.restframe.value-center<=20)&(spec.restframe.value-center>=-15))
    
    return idx2

def normal_spec(flux,noise,spec,line):
    
    linedata = pd.read_csv('lines.csv')
    idx = linedata['lines']==line
    center = float(linedata['wavelength'][idx])
    idx1 = np.where((spec.restframe.value-center>5)*(spec.restframe.value-center<20))
    idx2 = np.where((spec.restframe.value-center<-5)*(spec.restframe.value-center>-20))
    n = (sum(spec.flux[idx1])+sum(spec.flux[idx2])).value/(len(idx1[0])+len(idx2[0]))
    
    flux = flux/n
    noise = noise/n

    return flux,noise

def plotlines(spec,*names):
    linedata = pd.read_csv('lines.csv')

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)
    ax.plot(spec.restframe, spec.flux)
    
    for name in names:
        idx = linedata['lines']==name
        for x in linedata['wavelength'][idx]:
            idx_temp = np.where(((x-spec.restframe.value)<10)&((x-spec.restframe.value)>0))[0]
            if len(spec.flux[idx_temp])==0:
                continue
            ymin = sum(spec.flux[idx_temp].value)/len(spec.flux[idx_temp])
            ax.vlines(x,ymin+3,ymin+10,color='red',linewidth=0.8)
            print('Plotting '+name+' line, wavelength =%.2f'%x)
            ax.annotate(name, xy=(x, ymin+10), xytext=(-15, 10), xycoords = 'data',textcoords='offset points', fontsize='x-large')
    return

def compute_N(spec,line):
    idx = select_range(line,spec)
    linedata = pd.read_csv('lines.csv')
    center = float(linedata['wavelength'][linedata['lines']==line]) 
    lam = spec.restframe[idx].value
    noise = spec.noise[idx]
    flux = spec.flux[idx].value
    flux_normal,noise_normal = normal_spec(flux,noise,spec,line)
    popt,pcov = curve_fit(Ilam,lam,flux_normal,p0=[3,60,60,1e15,1e15], sigma=noise_normal, 
                          bounds=([-5, 0,0, 1e12,1e12], [5, 500,500, 1e16,1e16]))


    print('lgN1=%4f'%(np.log10(popt[3])))
    print('lgN2=%4f'%np.log10(popt[4]))
    err = Ilam(lam,*popt)-flux_normal
    lamfit = np.logspace(np.log10(min(lam)),np.log10(max(lam)),200)

    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    plt.subplots_adjust(bottom=0.1, top=1, hspace=1)
    ax2 = plt.subplot2grid((4, 4), (3, 0), rowspan=1, colspan=4)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(bottom=0.1, top=0.8, hspace=0)

    ax1.plot(lamfit,Ilam(lamfit,*popt),'r--',linewidth = 3)
    ax1.plot(lam,flux_normal,drawstyle='steps-mid',color='k')
    ax1.errorbar(lam,flux_normal,noise_normal,fmt='.',ecolor='black',color='black',linewidth=0.8,capsize=2)
    ax2.plot(lam,err,'green')
    ax2.hlines(0,min(lam),max(lam), color = 'black',linestyles='--')

    ax1.text(center+7,0,r'$lgN_{CIV} = %.4f(\lambda = 1548.20 \AA)$'%np.log10(popt[3])+'\n'
             +r'$lgN_{CIV} = %.4f (\lambda = 1550.77 \AA)$'%np.log10(popt[4])+'\n'
             +r'$\sigma_1=%.2f km/s(\lambda = 1548.20 \AA)$'%popt[1]+'\n'
             +r'$\sigma_2=%.2f km/s(\lambda = 1550.77 \AA)$'%popt[2],fontsize = 14)

    ax1.set_title(spec.name,fontsize = 18)
    ax1.set_ylabel(r"Normalized flux",fontsize=14)
    ax2.set_ylabel("Residual",fontsize=14)
    ax2.set_xlabel("Rest frame wavelength ($\AA$)",fontsize=14)
    ax1.set_ylim(-0.2,1.52)
    ax2.set_ylim(-0.55,0.7)
    # ax1.set_xlim(1540,1565)
    # ax2.set_xlim(1540,1565)

    ax1.grid()
    ax2.grid()

    plt.savefig('column density fitting of '+spec.name.split('.')[0]+'.png')
    
    return