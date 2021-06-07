from nemesis.NEMESIS_pipeline import full_pipeline, Transit_Pipeline
import os,sys
import time as clock
import pandas as pd
import numpy as np
import requests


# example star (TOI 270) TIC 259377017
ID = 259377017
Sector = 5

# pipeline settings for light curve extraction

#square pixel size of FFIs to use
cutoutsize=11

#Minimum Median Photon Counts to consider for a target star, few counts likely = very noisy light curves
minimum_photon_counts=1

#number of standard deviations from brightest target pixel to consider for aperture mask
threshold=7.5

#Pixel Level Decorrelation Nth order
pld_order=3

#Number of Principal Component Analysis terms to use
n_pca_terms=3

# Number of standard deviations to remove below/above the median flux baseline
Nsigma_low=2.0
sigma_high=2.0

# You can turn outlier removal on or off with a "yes" or "no" answer
remove_outliers='yes'
Nsigma_low,Nsigma_high= 2.0,2.0

# data points to remove in window size around momentum dumps
before_after_in_minutes=30.0

# Cadence refers to 30 minute observations ("long") or 2 minute observations ("short")
# cadence='long'
#cadence='short'


keep_FITS=False #will delete entire FITS files for all LCs...
keep_imagedata=False
verbose=True


# Transit Searching Pipeline Settings
N_transits=3
minP=1
oversampling_factor=9
SDEthreshold=6
duration_grid_step=1.05
path = os.getcwd()+'/'

print('TESTING LONG CADENCE FIRST')
cadence='long'

start=clock.time()
try:
    full_pipeline(ID,cutoutsize,Sector,minimum_photon_counts,threshold,pld_order,n_pca_terms, Nsigma_low, Nsigma_high, remove_outliers, before_after_in_minutes, path, cadence, verbose,keep_FITS=keep_FITS, keep_imagedata=keep_imagedata)
    print('')
    end=clock.time()
    runtime=end-start
    if runtime>60:
        print('light curve extraction runtime: '+str(runtime/60)+' minutes')
    if runtime<60:
        print('light curve extraction runtime: '+str(runtime)+' seconds')
    print(' ')
    print('Now doing Transit Search')
    print(' ')
    if cadence=='long':
        Path = path+'Sector_'+str(Sector)+'/FFI_PLD_LCs/'
        savefigpath = path+'Sector_'+str(Sector)+'/FFI_PLD_plots/'
    if cadence=='short':
        Path = path+'Sector_'+str(Sector)+'/TPF_PLD_LCs/'
        savefigpath = path+'Sector_'+str(Sector)+'/TPF_PLD_plots/'
    try:
        LC_df = pd.read_csv(Path+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_final_LC.txt')
        time=np.array(LC_df['Time'].to_list())
        flux=np.array(LC_df['Detrended Flux'].to_list())
        error=np.array(LC_df['Detrended Error'].to_list())
        ###
        input_LC = pd.DataFrame({'Time':time,'Flux':flux,'Error':error})
    except FileNotFoundError as FNFE:
        print(FNFE)
        print('no lightcurve for TIC '+str(ID)+' in Sector '+str(Sector))
        print('See above for details')
        print('')
    try: 
        PowerSpectrum_df, TransitModel_df, TransitParams_df, PowerSpectrum_df2, TransitModel_df2, TransitParams_df2 = Transit_Pipeline(SDEthreshold, ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path,keep_FITS=keep_FITS,keep_imagedata=keep_imagedata)
    except (requests.exceptions.HTTPError,requests.exceptions.ConnectionError) as E:
        print(E)
        gc.collect() #clearing garbage data
except (SystemExit,ConnectionError) as e:
    print(e)
    print('see output above')
    print(' ')
except (FileNotFoundError,KeyError) as E:
    print(E)
    print ('issues with astropy cache or MAST server, clearing cache now and trying in a minute...')
    os.system('rm -r ~/.astropy/cache/download/py3/lock/pid')
print('finished Sector '+str(Sector)+' light curves and Transit Searches!')
end=clock.time()
runtime=(end-start)
if runtime > 60:
    print("runtime: "+str(runtime/60)+" minutes")
if runtime < 60:
    print("runtime: "+str(runtime)+" seconds")
print(' ')
