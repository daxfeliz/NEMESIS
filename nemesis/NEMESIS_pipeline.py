# # Code Version 13: April 22nd, 2021


########## DEPENDENCIES FOR PIPELINE ###########
#Stuff for making directories and finding files
import glob, os, sys
from os import listdir
from os.path import isfile, join
import fnmatch
import time as clock
#Stuff for doing math and plotting
import numpy as np
import matplotlib
#matplotlib.use('Agg') #<--- for cluster only
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

#stuff for getting FFI data from MAST
import astropy
import astroquery
from astroquery.mast import Catalogs
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

#in case there are WiFi issues, these may help
from urllib.error import HTTPError
import requests

#stuff for detecting periodic transit events
from transitleastsquares import catalog_info
from transitleastsquares import period_grid
from transitleastsquares import transitleastsquares

#No one likes to see warnings (Feel free to comment this out if you do!)

import warnings
# warnings.filterwarnings(action='once') #useful to see a warning once but that's it
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

    
#for N order PLD
from itertools import combinations_with_replacement as CwR
from fbpca import pca
    
# for detrending
import wotan
from wotan import flatten

#for outlier removal
from wotan import slide_clip

#for saving
import pandas as pd

#for getting TPFs
import lightkurve as lk

# for clearning all that dataframe garbage memory that Python stores
import gc

#########################################################
#########################################################
########## CUSTOM PIPELINE FUNCTIONS BELOW ##############
#########################################################
########################################################

# Useful inverse-variance weighting binning function 
# (also used in AstroImageJ, courtesty of Karen Collins)

def Bin_func(time,flux,error,binsize):
    import math
    import numpy as np
    good = np.where(np.isfinite(time))
    timefit = time[good]
    fluxfit = flux[good]
    errfit  = error[good]
    timemax = np.max(timefit)
    timemin = np.min(timefit)
    npoints = len(timefit)
    nbins   = int(math.ceil((timemax - timemin)/binsize)) #binsize in days
    bintime = np.full((nbins,), np.nan)
    binflux = np.full((nbins,), np.nan)
    binerr  = np.full((nbins,), np.nan)
    for i in range(0,nbins-1):
        tobin = [np.where( (timefit >= (timemin + i*binsize)) & (timefit < (timemin + (i+1)*binsize)) )]
        if tobin[0] != -1:
    #     inverse variance weighted means
            binflux[i] = ((fluxfit[tobin]/(errfit[tobin]**2.0)).sum()) / ((1.0/errfit[tobin]**2.0).sum())
            bintime[i] = ((timefit[tobin]/(errfit[tobin]**2.0)).sum()) / ((1.0/errfit[tobin]**2.0).sum())
            binerr[i]  = 1.0 / (np.sqrt( (1.0/errfit[tobin]**2.0)).sum() )
    
    good2   = np.where(np.isfinite(bintime))
    bintime = bintime[good2]
    binflux = binflux[good2]
    binerr  = binerr[good2]
    
    return bintime, binflux, binerr


####Step 0
def Make_dirs(path,Sector,cadence):
    import os
    #Step 0: Creating directories to save figures and data
    path=path+'Sector_'+str(Sector)+'/'
    savefigpath1 = path+'FFI_PLD_plots/'
    savelcpath1 = path+'FFI_PLD_LCs/'
    savefigpath2 = path+'TPF_PLD_plots/'
    savelcpath2 = path+'TPF_PLD_LCs/'    
    downloadpath = path+'cache/'
    ###
    if cadence=='long':
        savefigpath=savefigpath1
        savelcpath=savelcpath1
        downloadpath=downloadpath
        if os.path.exists(savefigpath1)==True:
            pass
        else: 
            os.makedirs(savefigpath1)
        if os.path.exists(savelcpath1)==True:
            pass
        else:
            os.makedirs(savelcpath1) 
        if os.path.exists(downloadpath)==True:
            pass
        else: 
            os.makedirs(downloadpath)                        
    if cadence=='short':        
        savefigpath=savefigpath2
        savelcpath=savelcpath2
        downloadpath=downloadpath
        if os.path.exists(savefigpath2)==True:
            pass
        else: 
            os.makedirs(savefigpath2)
        if os.path.exists(savelcpath2)==True:
            pass
        else:
            os.makedirs(savelcpath2)
        if os.path.exists(downloadpath)==True:
            pass
        else: 
            os.makedirs(downloadpath)            
    ###  
    return path, savefigpath, savelcpath, downloadpath



####Step 1: Get Images
def center_cutout(hdu,cutoutsize,cadence):  
    from astropy.io import fits
    x=hdu[1].header['1CRPX4']
    y=hdu[1].header['2CRPX4']
    reference_pixel=[x,y]
    
    
    size=cutoutsize #new cutout

    col = int(x)
    row = int(y)
    s = (size/2, size/2)

    imshape = np.shape(hdu[1].data['FLUX'][1:]) #use 1st image?
    # Find the image edges
    col_edges = np.asarray([np.nanmax([0, col-s[0]]),
                            np.nanmin([col+s[0], imshape[1]])],
                           dtype=int)
    row_edges = np.asarray([np.nanmax([0, row-s[1]]),
                            np.nanmin([row+s[1], imshape[0]])],
                           dtype=int)

    primaryhdu = hdu[0].copy()

    #now we need coordinates
    from astropy.wcs import WCS
    w = WCS(hdu[2].header)
    X, Y = np.meshgrid(np.arange(imshape[2]), np.arange(imshape[1]))
    pos_corr1_pix = np.copy(hdu[1].data['POS_CORR1'])
    pos_corr2_pix = np.copy(hdu[1].data['POS_CORR2'])

    # We zero POS_CORR* when the values are NaN or make no sense (>50px)
    with warnings.catch_warnings():  # Comparing NaNs to numbers is OK here
        warnings.simplefilter("ignore", RuntimeWarning)
        bad = np.any([~np.isfinite(pos_corr1_pix),
                      ~np.isfinite(pos_corr2_pix),
                      np.abs(pos_corr1_pix - np.nanmedian(pos_corr1_pix)) > 50,
                      np.abs(pos_corr2_pix - np.nanmedian(pos_corr2_pix)) > 50], axis=0)
    pos_corr1_pix[bad], pos_corr2_pix[bad] = 0, 0

    # Add in POSCORRs
    X = (np.atleast_3d(X).transpose([2, 0, 1]) +
         np.atleast_3d(pos_corr1_pix).transpose([1, 2, 0]))
    Y = (np.atleast_3d(Y).transpose([2, 0, 1]) +
         np.atleast_3d(pos_corr2_pix).transpose([1, 2, 0]))

    # Pass through WCS
    ra, dec = w.wcs_pix2world(X.ravel(), Y.ravel(), 0)
#     ra, dec = w.wcs_pix2world(X.ravel(), Y.ravel(), 1)
    ra = ra.reshape((pos_corr1_pix.shape[0], imshape[1], imshape[2]))
    dec = dec.reshape((pos_corr2_pix.shape[0], imshape[1], imshape[2]))
    quality_mask = hdu[1].data['QUALITY']!=0
    r,d = ra[quality_mask], dec[quality_mask]

    hdu[2].header['RA_OBJ'] = np.nanmean(r[row_edges[0]:row_edges[1], col_edges[0]:col_edges[1]])
    hdu[2].header['DEC_OBJ'] = np.nanmean(d[row_edges[0]:row_edges[1], col_edges[0]:col_edges[1]])


    from copy import deepcopy
    hdus = [primaryhdu]


    # Copy the header
    primary_hdr = deepcopy(hdu[0].header)
    bintable_hdr = deepcopy(hdu[1].header)
    image_hdr = deepcopy(hdu[2].header)

    # hdus = fits.PrimaryHDU(data=hdu[0], header=primary_hdr)

    # Trim any columns that have the shape of the image, to be the new shape
    data_columns = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for idx, datacol in enumerate(hdu[1].columns):
            # If the column is 3D
            if (len(hdu[1].data[datacol.name].shape) == 3):
                # Make a copy, trim it and change the format
                datacol = deepcopy(datacol)
                datacol.array = datacol.array[:, row_edges[0]:row_edges[1], col_edges[0]:col_edges[1]]
                datacol._dim = '{}'.format(datacol.array.shape[1:]).replace(' ', '')
                datacol._dims = datacol.array.shape[1:]
                datacol._format = fits.column._ColumnFormat('{}{}'.format(np.product(datacol.array.shape[1:]),
                                                                          datacol._format[-1]))
                data_columns.append(datacol)
                bintable_hdr['TDIM{}'.format(idx)] = '{}'.format(datacol.array.shape[1:]).replace(' ', '')
                bintable_hdr['TDIM9'] = '{}'.format(datacol.array.shape[1:]).replace(' ', '')
                bintable_hdr['TDIM13'] = '{}'.format((0, datacol.array.shape[1])).replace(' ', '')
            else:
                data_columns.append(datacol)

    # Get those coordinates sorted for the corner of the TPF and the WCS
    bintable_hdr['1CRV*P'] = bintable_hdr['1CRV4P'] + col_edges[0]
    bintable_hdr['2CRV*P'] = bintable_hdr['2CRV4P'] + row_edges[0]
    bintable_hdr['1CRPX*'] = bintable_hdr['1CRPX4'] - col_edges[0]
    bintable_hdr['2CRPX*'] = bintable_hdr['2CRPX4'] - row_edges[0]


    # Make a table for the data
    data_columns[-1]._dim = '{}'.format((0, int(data_columns[5]._dim.split(',')[1][:-1]))).replace(' ', '')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BinTablehdu = fits.BinTableHDU.from_columns(data_columns, header=bintable_hdr)

    # Append it to the hdulist
    hdus.append(BinTablehdu)

    # Correct the aperture mask
    Imagehdu = hdu[2].copy()
    ar = Imagehdu.data
    ar = ar[row_edges[0]:row_edges[1], col_edges[0]:col_edges[1]]
    Imagehdu.header['NAXIS1'] = ar.shape[0]
    Imagehdu.header['NAXIS2'] = ar.shape[1]
    Imagehdu.data = ar
    hdus.append(Imagehdu)


    
    newfits = fits.HDUList(hdus)
    
    return newfits

def gethdu(ID,Sector,cutoutsize,cadence,minimum_photon_counts,verbose,downloadpath):
    #from NEMESIS_pipeline import center_cutout
    import lightkurve
    from lightkurve.search import _search_products 
    import os
    import requests
    import time as clock    
    starName="TIC "+str(ID)
    degrees = 21/3600 #21 arcsec to degrees
    if cadence=='long':
        ffi_or_tpf='FFI'
    if cadence=='short':
        ffi_or_tpf='Target Pixel'                    
    try:
        notworking=False
        start=clock.time()
        search_string=_search_products(starName, radius=degrees, \
                                       filetype=ffi_or_tpf, cadence=cadence, \
                                       mission=('TESS'), sector=Sector)
        while True:
            try:
                data = search_string.download(cutout_size=(cutoutsize+1,cutoutsize+1),\
                                              quality_bitmask='hardest',download_dir=downloadpath)
                notworking=1
                if notworking==True:
                    break
            except requests.exceptions.HTTPError as E:      
                print('')
                print(E)
                print('waiting 2 seconds then trying again to get through MAST')
                clock.sleep(2) 
                print(' ')                
        end=clock.time()
        runtime=end-start
        if runtime > 60:
            print('FFI download time: '+str(runtime/60)+' minutes')
        if runtime < 60:
            print('FFI download time: '+str(runtime)+' seconds')
#         lightkurve has its own cutout function but it seems iffy too...
        #force sizes to be even (sometimes it isn't...)
        x=data.hdu[1].header['1CRPX4']#-1
        y=data.hdu[1].header['2CRPX4']#-1
        reference_pixel=[x,y]
        try:
            data2=data.cutout(center=(reference_pixel[0],reference_pixel[1]),size=(cutoutsize,cutoutsize))
            hdu=data2.hdu
        except ValueError: #in case there are NaNs produced in centered cutout
            hdu=data.hdu

    #     os.system("rm -r " + downloadpath) #delete cache path

#         hdu=data.hdu        

        # NOT SURE THIS WORKS PROPERLY, LEAVE THIS OFF FOR NOW	
        # NOT SURE THIS WORKS PROPERLY, LEAVE THIS OFF FOR NOW	
        #hdu = center_cutout(hdu,cutoutsize=10,cadence=cadence)
        # NOT SURE THIS WORKS PROPERLY, LEAVE THIS OFF FOR NOW	
        # NOT SURE THIS WORKS PROPERLY, LEAVE THIS OFF FOR NOW	
        
        ### quick quality check for a minimum amount of desired max brightness
        if len(hdu[1].data['FLUX'])<1: 
            print('Weird image (maybe near edge of detector?)')
            return None,None,None,None,None
        elif (len(hdu[1].data['FLUX'])>1) & (np.nanmedian(hdu[1].data['FLUX']) < minimum_photon_counts):
                print('Images have median brightness less than '+str(minimum_photon_counts)+'!')
                return None,None,None,None,None           
        else:
            ###
            CCD=hdu[0].header['CCD']
            Camera=hdu[0].header['Camera']
            wcs = WCS(hdu[2].header)
            quality_mask = hdu[1].data['QUALITY']!=0

            # getting pixel coordinates
            # x=hdu[2].header['CRPIX1']
            # y=hdu[2].header['CRPIX2']                     
            # these are more accurate?
            if cadence=='short':
                x=hdu[1].header['1CRPX4']-1
                y=hdu[1].header['2CRPX4']-1
            if cadence=='long':
                x=hdu[1].header['1CRPX4']-1
                y=hdu[1].header['2CRPX4']-1
            reference_pixel=[x,y]

            return hdu,CCD,Camera,quality_mask,reference_pixel
    except (AttributeError, NameError, \
            UnboundLocalError,lightkurve.search.SearchError) as E: 
        print(E)
        print('NO HDU IN FFI/TPF FOR TIC '+str(ID)+' IN SECTOR '+ str(Sector))
        return None,None,None,None,None

    
    
def check_centroids(ID,Sector,cutoutsize,cadence,reference_pixel,savelcpath):
    #from NEMESIS_pipeline import readNDarr
    #from NEMESIS_pipeline import gethdu, Make_dirs,centroid_quadratic
    import pandas as pd   
    verbose=False

    pix_mask = readNDarr(savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_pix_mask")
    images = readNDarr(savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_image_fluxes")    
    time = pd.read_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_RAW_LC_systematics_removed.txt")['Time']
    
    centxs=[]
    centys=[]
    for x in range(len(images)):
        im=images[x]
        centxx,centyy = centroid_quadratic(im, pix_mask,reference_pixel)
        centxs=np.append(centxs,centxx)
        centys=np.append(centys,centyy)
        
    return centxs,centys



    
####Step 2 helper function    
def thresholdmask(hdu,reference_pixel,threshold,use_centroid=False):    
    import numpy as np
    ### threshold higher is more conservative, lower is more liberal
    ### define image from image data
    median_image = np.nanmedian(hdu[1].data['FLUX'], axis=0)
    vals=median_image[np.isfinite(median_image)].flatten()
    ###
    from astropy.stats.funcs import median_absolute_deviation as MAD
    ###
    # A value for the number of sigma by which a pixel needs to be
    # brighter than the median flux to be included in the aperture:
    ###
    MADcut = ( 1.4826*MAD(vals)*threshold +np.nanmedian(median_image))
    threshold_mask = np.nan_to_num(median_image) > MADcut
    

    if (reference_pixel == None):
        # return all regions above threshold
        return threshold_mask
    
   
    ###
    from scipy.ndimage import label #<--converts image to 1s and 0s as "labels"
    labels = label(threshold_mask)[0]
    label_args = np.argwhere(labels > 0)
    ###    
    ### For all pixels above threshold, compute distance to reference pixel:
    distances = [np.hypot(crd[0], crd[1]) for crd in label_args - np.array([reference_pixel[1], reference_pixel[0]])]
    ###
    ### Which label corresponds to the closest pixel?
    closest_arg = label_args[np.argmin(distances)]
    closest_label = labels[closest_arg[0], closest_arg[1]]
    ###
    threshmask=labels==closest_label    
    
    if use_centroid==False:
        reference_pixel = reference_pixel
    if use_centroid==True:
        reference_pixel = centroid_quadratic( np.nanmedian(hdu[1].data['FLUX']), threshold_mask,reference_pixel)    
    
    return threshmask#, reference_pixel



# Lightkurve method of finding centroids of brightest pixel (it's pretty good!)
def centroid_quadratic(data, mask, reference_pixel):#mask=None):
    """Computes the quadratic estimate of the centroid in a 2d-array.
    This method will fit a simple 2D second-order polynomial
    $P(x, y) = a + bx + cy + dx^2 + exy + fy^2$
    to the 3x3 patch of pixels centered on the brightest pixel within
    the image.  This function approximates the core of the Point
    Spread Function (PSF) using a bivariate quadratic function, and returns
    the maximum (x, y) coordinate of the function using linear algebra.
    For the motivation and the details around this technique, please refer
    to Vakili, M., & Hogg, D. W. 2016, ArXiv, 1610.05873.
    Caveat: if the brightest pixel falls on the edge of the data array, the fit
    will tend to fail or be inaccurate. 
    As used in the Lightkurve package of Barentsen et al.
    Parameters
    ----------
    data : 2D array
        The 2D input array representing the pixel values of the image.
    mask : array_like (bool), optional
        A boolean mask, with the same shape as `data`, where a **True** value
        indicates the corresponding element of data is masked.
    Returns
    -------
    column, row : tuple
        The coordinates of the centroid in column and row.  If the fit failed,
        then (NaN, NaN) will be returned.
    """
    # Step 1: identify the patch of 3x3 pixels (z_)
    # that is centered on the brightest pixel (xx, yy)
    if mask is not None:
        data = data * mask
    arg_data_max = np.nanargmax(data)
    yy, xx = np.unravel_index(arg_data_max, data.shape)
    
    # Make sure the 3x3 patch does not leave the image bounds
    if yy < 1:
        yy = 1
    if xx < 1:
        xx = 1
    if yy > (data.shape[0] - 2):
        yy = data.shape[0] - 2
    if xx > (data.shape[1] - 2):
        xx = data.shape[1] - 2

    z_ = data[yy-1:yy+2, xx-1:xx+2]

    # Next, we will fit the coefficients of the bivariate quadratic with the
    # help of a design matrix (A) as defined by Eqn 20 in Vakili & Hogg
    # (arxiv:1610.05873). The design matrix contains a
    # column of ones followed by pixel coordinates: x, y, x**2, xy, y**2.
    
    A = np.array([[1, -1, -1, 1,  1, 1],
                  [1,  0, -1, 0,  0, 1],
                  [1,  1, -1, 1, -1, 1],
                  [1, -1,  0, 1,  0, 0],
                  [1,  0,  0, 0,  1, 0],
                  [1,  1,  0, 1,  0, 0],
                  [1, -1,  1, 1, -1, 1],
                  [1,  0,  1, 0,  0, 1],
                  [1,  1,  1, 1,  1, 1]])
    
    # We also pre-compute $(A^t A)^-1 A^t$, cf. Eqn 21 in Vakili & Hogg.
    At = A.transpose()
    
    # In Python 3 this can become `Aprime = np.linalg.inv(At @ A) @ At`
    Aprime = np.matmul(np.linalg.inv(np.matmul(At, A)), At)

    # Step 2: fit the polynomial $P = a + bx + cy + dx^2 + exy + fy^2$
    # following Equation 21 in Vakili & Hogg.
    # In Python 3 this can become `Aprime @ z_.flatten()`
    a, b, c, d, e, f = np.matmul(Aprime, z_.flatten()) #dot product

    # Step 3: analytically find the function maximum,
    # following https://en.wikipedia.org/wiki/Quadratic_function
    det = 4 * d * f - e ** 2
    if abs(det) < 1e-6:
        return np.nan, np.nan  # No solution
    xm = - (2 * f * b - c * e) / det
    ym = - (2 * d * c - b * e) / det
    
    return np.array(xx + xm), np.array(yy + ym)    

#functions to calculate CDPP in a given window of time
def running_median(data, window_size):
    from collections import deque
    from bisect import insort, bisect_left
    from itertools import islice
    seq = iter(data)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result

#functions to calculate CDPP in a given window of time
def running_mean(data,window_size):
    import numpy as np
    if window_size > len(data):
        window_size = len(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

#functions to calculate CDPP in a given window of time
def CDPP(time,flux,error,method,unit,binsize=(1.0/24.0)): #1hr bin by default
    import numpy as np
    #Step 1: Calc number of data points per time bin
    cad = np.nanmedian(np.diff(time))
    Npts_per_timebin = int(binsize/cad)
    
    #Step 2: Estimate median or MAD of binned flux
    if method=='median':
        binmed = running_median(flux,Npts_per_timebin)
    if method=='mean':
        binmed = running_mean(flux,Npts_per_timebin)
        
    #Step 3: Calculated Combined Differential Photometric Precision (CDPP)
    if unit=='ppo':
        CDPP = np.nanstd(binmed) #in ppo per sqrt time bin
    if unit=='pph':
        CDPP = 1e2*np.nanstd(binmed) #in ppo per sqrt time bin        
    if unit=='ppt':        
        CDPP = 1e3*np.nanstd(binmed) #in ppt per sqrt time bin
    if unit=='ppm':
        CDPP = 1e6*np.nanstd(binmed) #in ppm per sqrt time bin
    return CDPP


def estimate_min_scatter(flux,initial_pix_mask,minNstd,maxNstd): #not sure if this is best, may be too strict...
    from scipy.signal import savgol_filter
    masks, scatters = [], []
    for i in range(minNstd, maxNstd):
        temp_pix_mask = thresholdmask(hdu,reference_pixel,i)
        tempflux = np.sum(flux[:,temp_pix_mask],axis=-1)        
        smooth = savgol_filter(f, 1001, polyorder=5)
        masks.append(msk)
        scatters.append(scatter)

    # Choose the aperture that minimizes the scatter
    pix_mask = masks[np.argmin(scatters)]
    return pix_mask

#### Step 2

#helper functions to save apertures, image fluxes
def saveNDarr(multiNDarr,path,filename):
    import pickle
    output = open(path+filename+'.pkl', 'wb')
    pickle.dump(multiNDarr, output)
    output.close()

def readNDarr(path,filename):
    import pickle
    pkl_file = open(path+filename+'.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()    
    return data


def query_region_for_nearby_stars(ID,radial_cone_in_arcsecs):
    #stuff for getting FFI data from MAST
    from astroquery.mast import Catalogs
    import numpy as np
    import time as clock
    import requests
    
    starName="TIC "+str(ID) 
    radial_cone = radial_cone_in_arcsecs/ 3600.0 # angular radius in degrees
    try:
        catalogData = Catalogs.query_object(starName, radius = radial_cone, catalog = "TIC")
    except requests.exceptions.ConnectionError as E:
        clock.sleep(5) #pause 5 seconds then try again
        catalogData = Catalogs.query_object(starName, radius = radSearch, catalog = "TIC")    
    #    
    return catalogData

def convert_TessMag_to_Flux(Tmag):
    f = 10**(-(Tmag/2.5))
    return f

def calc_flux_contamination(ID,radial_cone_in_arcseconds=63):
    
    catalogData = query_region_for_nearby_stars(ID,radial_cone_in_arcseconds)
    
    Tmag_target_star = catalogData[0]['Tmag']
    Flux_target_star = convert_TessMag_to_Flux(Tmag_target_star)
    Flux_all = []
    for t in range(len(catalogData)):
        Flux_all = np.append(Flux_all, convert_TessMag_to_Flux(catalogData[t]['Tmag']))
    Flux_total = np.sum(Flux_all)
    
    flux_contamination_ratio = Flux_target_star / Flux_total #this is light FROM the target star
    flux_contamination_ratio = 1 - Flux_target_star / Flux_total #this is light NOT FROM the target star (will produce ratio=0 for 1 - 10^(-Tmag,target/2.5) / 10^(-Tmag,target/2.5)
    
    return flux_contamination_ratio
    
    

def SAP(ID,Sector,cutoutsize,hdu,quality_mask,threshold,cadence,reference_pixel,verbose,savelcpath,use_SPOC_aperture='no'):
    import numpy as np
    quality_mask = hdu[1].data['QUALITY']!=0
    median_image = np.nanmedian(hdu[1].data['FLUX'][~quality_mask], axis=0)
    ###
    ###
    flux = hdu[1].data['FLUX'][~quality_mask]  
    rawtime = hdu[1].data['TIME'][~quality_mask]
    #include only finite values, excluding NaNs
    m = np.any(np.isfinite(flux),axis=(1,2)) 
    rawtime = np.ascontiguousarray(rawtime[m],dtype=np.float64)
    flux = np.ascontiguousarray(flux[m],dtype=np.float64)    
    ###
    ###
    # find dimmest pixels using inverted threshold mask
    bkg_mask = ~thresholdmask(hdu,reference_pixel=None,threshold=1/1000)
    ###
    # select brightest pixels using threshold mask
    try:
        T=threshold
        threshmask=thresholdmask(hdu,reference_pixel,T)
    except ValueError as e:
        try:
            T=threshold/2.0
            print('threshold too high, no pixels selected. Trying half of input threshold: ',str(T))
            threshmask=thresholdmask(hdu,reference_pixel,T)
        except ValueError as ee:
            try:
                T=threshold/3.0
                print('threshold STILL too high, no pixels selected. Trying third of input threshold: ',str(T))
                threshmask=thresholdmask(hdu,reference_pixel,T)
            except ValueError as eee:
                print('setting threshold=1 (last resort)')
                T=1.0
                try:
                    threshmask=thresholdmask(hdu,reference_pixel,T)
                except ValueError as eeee:
                    print('Ok, I tried...')
                    pass
    try:
        pix_mask=threshmask
        print('selected threshold: ',T)
    except UnboundLocalError as UE:
        print('unable to find suitable threshold mask')
        return
    
    if cadence=='short':
        if use_SPOC_aperture=='yes':
            print('using SPOC aperture')
            try:
                pipeline_mask = hdu[2].data & 2 > 0
            except TypeError:  # Early versions of TESScut returned floats in HDU 2
                pipeline_mask = np.ones(hdu[2].data.shape, dtype=bool)
            pix_mask = pipeline_mask
            bkg_mask = ~pipeline_mask
        if use_SPOC_aperture=='no':
            print('using threshold mask')
    ###
    ###
    #subtract background flux
#     flux -= np.median(flux[:, bkg_mask], axis=-1)[:, None, None]
#     fluxsum = np.nansum(flux[:, pix_mask], axis=-1)
#     is_allnan = ~np.any(np.isfinite(flux[:, pix_mask]), axis=1) #removing nans
#     fluxsum[is_allnan] = np.nan
#     sap_flux=fluxsum/np.nanmedian(fluxsum)
    
    
    rawflux = np.nansum(flux[:, pix_mask], axis=-1)
    bkgFlux = np.nansum(flux[:, bkg_mask], axis=-1)
    
    Npixbkg = len(np.where(bkg_mask == True)[0])
    Npixaper= len(np.where(pix_mask == True)[0])
    
    bkgFlux = bkgFlux/Npixbkg #normalize background
    
    rawsap_flux = rawflux - (bkgFlux * Npixaper)
    sap_flux = rawsap_flux / np.nanmedian(rawsap_flux)
    
    # Deblending by calculating flux contaminatio ratio for stars within 3 TESS pixels of target
    # If no othery stars are nearby, this ratio = 1
    flux_contamination_ratio = calc_flux_contamination(ID)
    # subtract and re-normalize sap_flux to complete the process
    sap_flux = sap_flux - flux_contamination_ratio
    sap_flux = sap_flux / np.nanmedian(sap_flux)

    
    nanmask = np.where(np.isfinite(sap_flux)==True)[0]
#     e0=hdu[1].data['FLUX_ERR'][~quality_mask]
#     error = np.nansum(e0[:, pix_mask]**2, axis=1)**0.5
#     is_allnan = ~np.any(np.isfinite(e0[:, pix_mask]), axis=1)
#     error[is_allnan] = np.nan

    # these provide LARGE errors for either or both TPFs/FFIs

    error = np.abs( sap_flux / np.nanmedian(np.nansum(sap_flux)/np.nanmedian(sap_flux)))
    error =error[nanmask]
    
    if verbose==True:
        print('len check ',' T', len(rawtime),' SAP',len(sap_flux), ' E ', len(error))
        print('shape check ',' T', np.shape(rawtime),' SAP',np.shape(sap_flux), ' E ', np.shape(error))
    ###
    ###
    ###
    SAP_LC = pd.DataFrame({"Time":rawtime,"RAW SAP Flux":rawsap_flux,"SAP Flux":sap_flux,"SAP Error":error, "Background Flux":bkgFlux})
    ###
    ### saving data before momentumdump removal at a later step
    ###
    if verbose==True:    
        print('RAW len check:', len(rawtime),len(sap_flux),len(error))
    SAP_LC.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_RAW_LC.txt",index=False)
    
    return bkg_mask, pix_mask ,flux, median_image, SAP_LC, flux_contamination_ratio
    
def get_coordinates(hdu, cadence='all'):
    """Returns two 3D arrays of RA and Dec values in decimal degrees.
    If cadence number is given, returns 2D arrays for that cadence. If
    cadence is 'all' returns one RA, Dec value for each pixel in every cadence.
    Uses the WCS solution and the POS_CORR data from TPF header.
    Parameters
    ----------
    cadence : 'all' or int
        Which cadences to return the RA Dec coordinates for.
    Returns
    -------
    ra : numpy array, same shape as tpf.flux[cadence]
        Array containing RA values for every pixel, for every cadence.
    dec : numpy array, same shape as tpf.flux[cadence]
        Array containing Dec values for every pixel, for every cadence.
    """
    from astropy.wcs import WCS
    import warnings
    wcs = WCS(hdu[2].header)
    X, Y = np.meshgrid(np.arange(hdu[1].data['FLUX'].shape[2]), np.arange(hdu[1].data['FLUX'].shape[1]))
    pos_corr1_pix = np.copy(hdu[1].data['POS_CORR1'])
    pos_corr2_pix = np.copy(hdu[1].data['POS_CORR2'])

    # We zero POS_CORR* when the values are NaN or make no sense (>50px)
    with warnings.catch_warnings():  # Comparing NaNs to numbers is OK here
        warnings.simplefilter("ignore", RuntimeWarning)
        bad = np.any([~np.isfinite(pos_corr1_pix),
                      ~np.isfinite(pos_corr2_pix),
                      np.abs(pos_corr1_pix - np.nanmedian(pos_corr1_pix)) > 50,
                      np.abs(pos_corr2_pix - np.nanmedian(pos_corr2_pix)) > 50], axis=0)
    pos_corr1_pix[bad], pos_corr2_pix[bad] = 0, 0

    # Add in POSCORRs
    X = (np.atleast_3d(X).transpose([2, 0, 1]) +
         np.atleast_3d(pos_corr1_pix).transpose([1, 2, 0]))
    Y = (np.atleast_3d(Y).transpose([2, 0, 1]) +
         np.atleast_3d(pos_corr2_pix).transpose([1, 2, 0]))

    # Pass through WCS
    quality_mask = hdu[1].data['QUALITY']!=0
#     ra, dec = wcs.wcs_pix2world(X.ravel(), Y.ravel(), 1)
    ra, dec = wcs.wcs_pix2world(X.ravel(), Y.ravel(), 0)
    ra = ra.reshape((pos_corr1_pix.shape[0], hdu[1].data['FLUX'].shape[1], hdu[1].data['FLUX'].shape[2]))
    dec = dec.reshape((pos_corr2_pix.shape[0], hdu[1].data['FLUX'].shape[1], hdu[1].data['FLUX'].shape[2]))
    ra, dec = ra[quality_mask], dec[quality_mask]
    if cadence != 'all':
        return ra[cadence], dec[cadence]
    return ra, dec

def plot_orientation(hdu, ax):
    
    nx, ny = hdu[1].data['FLUX'].shape[1:]
    col,row = hdu[1].header['1CRPX5'], hdu[1].header['2CRPX5']
    x0, y0 = 1,ny-2.5# + int(0.5 * nx), 0 + int(0.25 * ny)
    #print(x0,y0)
    # East
    tmp = get_coordinates(hdu, cadence='all')
    ra00, dec00 = tmp[0][0][0][0], tmp[1][0][0][0]
    ra10, dec10 = tmp[0][0][0][-1], tmp[1][0][0][-1]
    theta = np.arctan((dec10 - dec00) / (ra10 - ra00))
    if (ra10 - ra00) < 0.0:
        theta += np.pi
    # theta = -22.*np.pi/180.
    x1, y1 = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
    ax.arrow(x0, y0, x1, y1, head_width=0.2, color="black")
    ax.text(x0 + 1.5 * x1, y0 + 1.5 * y1, "E", color="black")
    # North
    theta = theta + 90.0 * np.pi / 180.0
    x1, y1 = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
    ax.arrow(x0, y0, x1, y1, head_width=0.2, color="black")
    ax.text(x0 + 1.5 * x1, y0 + 1.5 * y1, "N", color="black")

    
def get_TESS_sources(ID,hdu,downloadpath,magnitude_limit):
    #stuff for getting data from MAST
    import astropy
    from astroquery.mast import Catalogs
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia
    from astropy.wcs import WCS
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Angle
    from astroquery.vizier import Vizier
    import numpy as np
    import pandas as pd
    # changing cache directories
    Tesscut.cache_location=downloadpath
    Catalogs.cache_location=downloadpath
    Vizier.cache_location=downloadpath
    ###
    starName="TIC "+str(ID)
    pix_scale=21
    cone = 0.5*np.nanmax(hdu[1].data['FLUX'].shape[1:]) * pix_scale
    cone = 5*pix_scale
#     result = Catalogs.query_object(starName, catalog = "TIC",radius=Angle(cone, "arcsec"))
    ###
    ra=hdu[2].header['RA_OBJ']
    dec=hdu[2].header['DEC_OBJ']    
    # Get the positions of the Gaia sources
    frame='icrs'
    c1 = SkyCoord(ra,dec, frame=frame, unit='deg')
    result = Catalogs.query_region(c1, catalog = "TIC",radius=Angle(cone, "arcsec"))
    
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    
    print('')
#     print(result.columns)
#     result=result.filled()
    result=result.to_pandas()
    result = result[result['Tmag'] < magnitude_limit]
    ###
    if len(result) == 0:
        raise no_targets_found_message
        
    return result

def get_GAIA_sources(ID,hdu, downloadpath,magnitude_limit):
    #stuff for getting data from MAST
    import astropy
    from astroquery.mast import Catalogs
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia
    from astropy.wcs import WCS
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Angle
    from astroquery.vizier import Vizier
    import numpy as np
    import pandas as pd
    ###
    # changing cache directories
    Tesscut.cache_location=downloadpath
    Catalogs.cache_location=downloadpath
    Vizier.cache_location=downloadpath
    ###
    starName="TIC "+str(ID)
    pix_scale=21
    
    ra=hdu[2].header['RA_OBJ']
    dec=hdu[2].header['DEC_OBJ']    
    # Get the positions of the Gaia sources
    c1 = SkyCoord(ra,dec, frame='icrs', unit='deg')
    # Use pixel scale for query size
    pix_scale = 21.0
#     Vizier.ROW_LIMIT = -1 # doesn't include target star (?)
    cone = 0.5*np.nanmax(hdu[1].data['FLUX'].shape[1:]) * pix_scale
    cone = 5*pix_scale
    try:
        result = Vizier.query_region(c1, catalog=["I/345/gaia2"],radius=Angle(cone, "arcsec"))
#     result = Vizier.query_region(c1, catalog=["GAIA"],radius=Angle(cone, "arcsec"))
    except requests.exceptions.ConnectionError as CE:
        print(CE)
        import time as clock
        clock.sleep(10)
        print('trying Vizier query again')
        result = Vizier.query_region(c1, catalog=["I/345/gaia2"],radius=Angle(cone, "arcsec"))
        print('')
    #
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message
    result = result["I/345/gaia2"].to_pandas() #using GAIA DR2   
    result = result[result.Gmag < magnitude_limit]
    
    #rename RA DEC columns to match TESS query
    result=result.rename(columns={'RA_ICRS': 'ra','DE_ICRS': 'dec', 'pmRA':'pmRA', 'pmDE':'pmDEC'})
    
    return result


def Get_stellar_params(ID,downloadpath):
    from transitleastsquares import catalog_info
    #stuff for getting FFI data from MAST
    import astropy
    from astroquery.mast import Catalogs
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia
    from astropy.wcs import WCS
    from astropy.io import fits
    import astropy.units as u
    import numpy as np
    import time as clock
    import requests
    # changing cache directories
    Tesscut.cache_location=downloadpath
    Catalogs.cache_location=downloadpath
    
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)

        starName="TIC "+str(ID) 
        radSearch = 21/3600.0 # angular radius in degrees
        catalogData = Catalogs.query_object(starName, radius = radSearch, catalog = "TIC")
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)

        starName="TIC "+str(ID) 
        radSearch = 21/3600.0 # angular radius in degrees
        catalogData = Catalogs.query_object(starName, radius = radSearch, catalog = "TIC")
    ###
    ra = catalogData[0]['ra']
    dec = catalogData[0]['dec']
    coord = SkyCoord(ra, dec, unit = "deg")
    Tmag=catalogData[0]['Tmag']
    Gmag=catalogData[0]['GAIAmag']
    Vmag=catalogData[0]['Vmag']
    rmag=catalogData[0]['rmag']
    imag=catalogData[0]['imag']
    zmag=catalogData[0]['zmag'] 
    Jmag=catalogData[0]['Jmag']
    Hmag=catalogData[0]['Hmag']
    Kmag=catalogData[0]['Kmag']
    Teff=catalogData[0]['Teff'] 
    logg=catalogData[0]['logg']
    rho=catalogData[0]['rho']
    dist=catalogData[0]['d']
    ###
    ###
    if str(Vmag)==str('--'): Vmag=np.nan
    if str(Tmag)==str('--'): Tmag=np.nan
    if str(Gmag)==str('--'): Gmag=np.nan
    if str(rmag)==str('--'): rmag=np.nan
    if str(imag)==str('--'): imag=np.nan
    if str(zmag)==str('--'): zmag=np.nan
    if str(Jmag)==str('--'): Jmag=np.nan            
    if str(Hmag)==str('--'): Hmag=np.nan
    if str(Kmag)==str('--'): Kmag=np.nan
    if str(logg)==str('--'): logg=np.nan
    if str(rho)==str('--'): rho=np.nan
    if str(dist)==str('--'): dist=np.nan
    ###
    return Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist

def plot_catalog_sources(ID,hdu,ax, downloadpath,magnitude_limit,catalog,dot_scale=35,dolegend="no"):
    #stuff for getting data from MAST
    import astropy
    from astroquery.mast import Catalogs
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia
    from astropy.wcs import WCS
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Angle
    from astroquery.vizier import Vizier
    import numpy as np
    import pandas as pd
    
    # changing cache directories
    Tesscut.cache_location=downloadpath
    Catalogs.cache_location=downloadpath
    Vizier.cache_location=downloadpath
    
    if catalog=='TESS':
        result = get_TESS_sources(ID,hdu, downloadpath, magnitude_limit)
    if catalog=='GAIA':
        result = get_GAIA_sources(ID,hdu, downloadpath,magnitude_limit)
    
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
    
    if result is None:
        raise no_targets_found_message
        pass
    elif len(result) == 0:
        raise too_few_found_message
        pass
    else:
        radecs = np.vstack([result['ra'], result['dec']]).T
        wcs = WCS(hdu[2].header)
        coords = wcs.all_world2pix(radecs, 0) 
        #coords = wcs.wcs_world2pix(radecs, 1)  #test if this is better?
        ###
        year = ((np.nanmin(hdu[1].data['TIME'])+2457000 ) * u.day).to(u.year)
        ###
        pmra = ((np.nan_to_num(np.asarray(result['pmRA'])) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
        pmdec = ((np.nan_to_num(np.asarray(result['pmDEC'])) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
        result['ra'] += pmra
        result['dec'] += pmdec  
        ###
        if catalog=='TESS':
            df = pd.DataFrame(data=dict(x=coords[:, 0], y=coords[:, 1], mag=result['Tmag']))
        if catalog=='GAIA':
            df = pd.DataFrame(data=dict(x=coords[:, 0], y=coords[:, 1], mag=result['Gmag']))
        Nbins= 22#(int(magnitude_limit)+1)
        #print('Nbins',Nbins)
        bins = np.linspace(df.mag.min(), df.mag.max(), Nbins)
        grouped = df.groupby(np.digitize(df.mag, bins))
        ###
        ###
        #print('sizes',sizes)
        labels=[]
        for i in range(Nbins):
            labels=np.append(labels,str(i))
        cm = plt.get_cmap('gist_rainbow')
        ax.set_prop_cycle(color=[cm(1.*i/Nbins) for i in range(Nbins)])

        df=df.sort_values(by=['mag']).reset_index()
        del df['index']

        scale=dot_scale
        sizes=[scale*(len(df)+1)]    
        ### Get stellar params for target star
        Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist = Get_stellar_params(ID,downloadpath) 
        reference_pixel=[hdu[1].header['1CRPX4']-1,hdu[1].header['2CRPX4']-1]
        #print('targ gmag',Gmag)
        sizes = 128 / 2**((np.array(df['mag'])-Gmag))
    #     for i in range(len(df)):
    #         sizes = np.append(sizes,scale*(i+1))

        cad=(np.round(np.nanmedian(np.diff(hdu[1].data['TIME']))*60*24))
        if cad==30:
            offset=1
        if cad==2:
            offset=1.5
        offset=0

        #try to keep labels inside bounds of cutout image
        xlimit = np.shape(hdu[1].data['FLUX'][0])[0]-1
        ylimit = np.shape(hdu[1].data['FLUX'][0])[1]-1

        for x in reversed(range(len(df))): #s=base_ms / 2 ** dmag,
            if (df['x'][x]-offset > xlimit) or (df['y'][x]-offset > ylimit):
                pass
            else:
                ax.scatter(df['x'][x]-offset,df['y'][x]-offset,s=sizes[x],color='cyan')
                #print(sizes[x])
                ax.text(df['x'][x]-offset,df['y'][x]-offset, str(int(df['mag'][x])), color="red", zorder=100,fontsize=16)
    ###
    ax.text(reference_pixel[0],reference_pixel[1], str(int(Gmag)), color="black", zorder=100,fontsize=16)
    ###
    ###
    if dolegend=="yes":    
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
    
        if catalog=='TESS':
            if len(by_label.values())>4:
                ncols=2
            else:
                ncols=1
            ax.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.4,1.1),\
                      loc='upper right',fontsize=8,ncol=ncols,markerscale=1,\
                      labelspacing=2.5,title="TESS Mag")                   
        if catalog=='GAIA': 
            if len(by_label.values())>4:
                ncols=2
            else:
                ncols=1                        
            ax.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.4,1.1),\
                           loc='upper right',fontsize=8,ncol=ncols,markerscale=1,\
                           labelspacing=1.,title="GAIA Mag")   
    ###    
    #plot orientation of cutout (N,E)
    plot_orientation(hdu,ax)
    ###
    return bins, sizes,labels,grouped,result,df

def plot_cutouts(ID,Sector,cadence,hdu,pix_mask,bkg_mask,reference_pixel,fig,axes,savelcpath,downloadpath,do_colorbar='yes'):
    import astropy
    from astroquery.mast import Catalogs
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia
    from astropy.wcs import WCS
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Angle
    from astroquery.vizier import Vizier
    import numpy as np
    import pandas as pd
#     from NEMESIS_pipeline import centroid_quadratic,Get_stellar_params    
    
    # changing cache directories
    Tesscut.cache_location=downloadpath
    Catalogs.cache_location=downloadpath
    Vizier.cache_location=downloadpath
    

    if len(axes)==1:
        ax1=axes[0]
    if len(axes)==2:
        ax1=axes[0]
        ax2=axes[1]
    
    wcs = WCS(hdu[2].header)
    median_image = np.nanmedian(hdu[1].data['FLUX'],axis=0)
    cutoutsize=np.shape(hdu[1].data['FLUX'][0])[0]
    centx,centy = centroid_quadratic(median_image, pix_mask,reference_pixel)
    cxs,cys = check_centroids(ID,Sector,cutoutsize,cadence,reference_pixel,savelcpath)
    
    fs=12
    ms=20
    if cadence=='long':
        ax1.set_title('FFI cutout')
    if cadence=='short':
        ax1.set_title('TPF cutout')        
    medi_img1 = ax1.imshow(median_image, cmap="gray_r",alpha=0.9, aspect="auto")  
    medi_img1.set_clim(vmin=0, vmax=np.nanmax(median_image))
    ax1.plot(reference_pixel[0],reference_pixel[1],'bX',markersize=ms)
    ax1.plot(centx,centy,'yX',markersize=ms)
    ax1.plot(cxs,cys,color='purple',marker='.',markersize=3,linestyle='none')    
    if do_colorbar=="yes":        
        cbar = fig.colorbar(medi_img1,ax=ax1,pad=0.01)
        cbar.ax.set_ylabel('Median Photon Counts', fontsize = fs,rotation=270)
        cbar.ax.get_yaxis().labelpad = fs
    ax1.invert_yaxis()
    ax1.set_xlim(-0.5,np.shape(hdu[1].data['FLUX'][0])[0]-1)
    ax1.set_ylim(-0.5,np.shape(hdu[1].data['FLUX'][0])[1]-1)
    ax1.coords[0].set_axislabel('RA')
    ax1.coords[1].set_axislabel('DEC',minpad=-2) #minpad =labelpad
    ###
#     ax1.coords[0].set_major_formatter('hh:mm:ss')
#     ax1.coords[1].set_major_formatter('hh:mm:ss')
    ax1.coords[0].set_major_formatter('d.dddd')
    ax1.coords[1].set_major_formatter('d.dddd')    
    ###
    Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist = Get_stellar_params(ID,downloadpath)
    ###
    cm = plt.get_cmap('gist_rainbow')    
    #also plot RA,DEC from pixel coordinates (TAN), more digits than (_OBJ)
    ra2=hdu[2].header['CRVAL1'] 
    dec2=hdu[2].header['CRVAL2']
    
    px,py = wcs.all_world2pix([[ra2,dec2]],0)[0]
#     px,py = wcs.all_world2pix([[ra2,dec2]],1)[0]
    if len(axes)==1:
        try:
            bins, sizes,labels,grouped,result,df = plot_catalog_sources(ID,hdu,ax1,downloadpath,magnitude_limit=18,catalog='GAIA',dot_scale=10,dolegend="no")
        except ValueError as e:
            print(e)
        ax1.imshow(pix_mask, cmap="Reds", alpha=0.9, aspect="auto") #selected aperture
        ax1.imshow(bkg_mask, cmap="Purples", alpha=0.25, aspect="auto")
        
    if len(axes)==2:    
        try:
            bins, sizes,labels,grouped,result,df = plot_catalog_sources(ID,hdu,ax1,downloadpath,magnitude_limit=18,catalog='GAIA',dot_scale=10,dolegend="no")
        except ValueError as e:
            print(e)
        ax2.set_title('Aperture and Background Masks')
        medi_img2 = ax2.imshow(median_image, cmap="gray_r",alpha=0.9, aspect="auto")  
        medi_img2.set_clim(vmin=0, vmax=np.nanmax(median_image))
        ax2.imshow(pix_mask, cmap="Reds", alpha=0.9, aspect="auto") #selected aperture
        ax2.imshow(bkg_mask, cmap="Purples", alpha=0.25, aspect="auto")
        ax2.plot(reference_pixel[0],reference_pixel[1],'rX',markersize=ms)
        ax2.plot(centx,centy,'yX',markersize=ms)
        ax2.plot(cxs,cys,color='purple',marker='.',markersize=3,linestyle='none')
        if do_colorbar=="yes":        
            cbar = fig.colorbar(medi_img2,ax=ax2,pad=0.01)
            cbar.ax.set_ylabel('Median Photon Counts', fontsize = fs,rotation=270)
            cbar.ax.get_yaxis().labelpad = fs
        ax2.invert_yaxis()
        ax2.set_xlim(-0.5,np.shape(hdu[1].data['FLUX'][0])[0]-1)
        ax2.set_ylim(-0.5,np.shape(hdu[1].data['FLUX'][0])[1]-1)
        ax2.coords[0].set_axislabel('RA')
        ax2.coords[1].set_axislabel('DEC',minpad=-2) #minpad =labelpad
#         ax2.coords[0].set_major_formatter('hh:mm:ss')
#         ax2.coords[1].set_major_formatter('hh:mm:ss')
        ax2.coords[0].set_major_formatter('d.dddd')
        ax2.coords[1].set_major_formatter('d.dddd')
    
    

####Step 4 (Two funcs b/c they're long and a 3rd to apply one)
def momentumdump_check(SectorNum):
    SectorNum=float(SectorNum)
    if SectorNum==1 or SectorNum==2 or SectorNum==3 or SectorNum==4:
        mdumps=2.5
        if SectorNum==1:
            #orbit start times
            t_0 = 1325.29278
            t_1 = 1338.52153
        if SectorNum==2:
            #orbit start times
            t_0 = 1354.10102
            t_1 = 1368.59406
        if SectorNum==3:
            #orbit start times
            t_0 = 1382.03987+mdumps #since it's cut out in mdump removal below
            t_1 = 1396.60497 
        if SectorNum==4:
            #orbit start times
            t_0 = 1410.89974
            t_1 = 1424.54897
        ###            
        ###                
    elif SectorNum==5:
        mdumps=3.0
        #orbit start times
        t_0 = 1437.82566
        t_1 = 1451.54898 
        ###            
    elif SectorNum==6 or SectorNum==7 or SectorNum==8 or SectorNum==9 or SectorNum==10 or SectorNum==11 or SectorNum==12:
        mdumps=3.125
        if SectorNum==6:
            #orbit start times
            t_0 = 1465.21262
            t_1 = 1478.11304
        if SectorNum==7:
            #orbit start times
            t_0 = 1491.62553
            t_1 = 1504.69775
        if SectorNum==8:
            #orbit start times
            t_0 = 1517.34150
            t_1 = 1530.25816
        if SectorNum==9:
            #orbit start times
            t_0 = 1543.21648
            t_1 = 1557.00080
        if SectorNum==10:
            #orbit start times
            t_0 = 1569.43176
            t_1 = 1582.76231
        if SectorNum==11:
            #orbit start times
            t_0 = 1596.77203
            t_1 = 1610.77620
        if SectorNum==12:
            #orbit start times
            t_0 = 1624.94979
            t_1 = 1640.03312
    elif SectorNum==13: 
        mdumps=3.375
        #orbit start times
        t_0 = 1653.91505
        t_1 = 1668.61921
        ###        
    elif SectorNum==14:
        mdumps=4.4
        #orbit start times
        t_0 = 1683.34838
        t_1 = 1697.33865       
        ### 
    elif SectorNum==31:
        mdumps=6 # 6-ish, half way thru each orbit. We won't use this number,just need an output
        #orbit start times
        t_0 = 2144.50927
        t_1 = 2157.45371
        ###         
    return mdumps,t_0,t_1


# this will remove regions of high scatter or when Earth/Moon (EM) is in field of view
def momentumdump_removal(SectorNum,Camera,CCD,before_after_in_minutes, time):
    import numpy as np
#     print('mA: ',len(time))
    SectorNum=float(SectorNum)
    ###    
    if SectorNum==1:
        #orbit start times
        t_0 = 1325.29278
        t_1 = 1338.52153
        mdumps=2.5
        jittermask = (time < 1347) | (time > 1350)
        ###
        if Camera==1:
#             print('go')
            EM_glintmask = (time < time[-1]) | (time > (time[-1]-2)) #last 2 days no good
            mask = (jittermask) & (EM_glintmask)
        else:
            mask = (jittermask)               
            ###                
    if SectorNum==2:
        mdumps=2.5
        #orbit start times
        t_0 = 1354.10102
        t_1 = 1368.59406
        ###        
        #no jittermask for 2
        mask = np.ones(len(time), dtype=bool) #makes them all TRUE
        ###        
    if SectorNum==3:
        mdumps=2.5
        #orbit start times
        t_0 = 1382.03987+mdumps #since it's cut out below
        t_1 = 1396.60497 
        jittermask = (time < 1382) | (time > 1385.9)
        jittermask2 = (time < 1395.2325) | (time > 1396.6)
        jittermask3 = (time < 1406.2)
        ###
        mask = (jittermask) & (jittermask2) & (jittermask3)
        ###        
    if SectorNum==4:
        #orbit start times
        t_0 = 1410.89974
        t_1 = 1424.54897
        #momentem dump rate
        mdumps=2.5
        jittermask = (time > 1413) | (time < 1436.85)
        jittermask2 = (time < 1418.53691) | (time > 1424.5) #anomaly
        ###        
        if Camera==1: 
            EM_glintmask = (time < 1422.2297) | (time > 1423.502) 
            EM_glintmask2 = (time < 1436.1047) | (time > 1436.8353) 
            ###
            ###            
            mask = (jittermask) & (jittermask2) & (EM_glintmask) & (EM_glintmask2)
        else:
            #combine masks
            mask = (jittermask) & (jittermask2)         
            ###            
    if SectorNum==5:
        mdumps=3.0
        #orbit start times
        t_0 = 1437.82566
        t_1 = 1451.54898 
        ###        
        #no jittermask for Sector 5
        ###        
        if Camera==1: 
            EM_glintmask = (time < 1463.93945) | (time > 1464.25) 
            mask = (np.ones(len(time), dtype=bool)) & (EM_glintmask)
        else:
            mask = np.ones(len(time), dtype=bool) #makes them all TRUE
            ###            
    if SectorNum==6 or SectorNum==7:
        mdumps=3.125
        ###        
        if SectorNum==6:
            #orbit start times
            t_0 = 1465.21262
            t_1 = 1478.11304
        if SectorNum==7:
            #orbit start times
            t_0 = 1491.62553
            t_1 = 1504.69775
        ###            
        #no jittermask for Sector 6 OR 7
        mask = np.ones(len(time), dtype=bool) #makes them all TRUE
        ###        
    if SectorNum==8:
        mdumps=3.125
        #orbit start times
        t_0 = 1517.34150
        t_1 = 1530.25816
        ###        
        #no jittermask for Sector 8
        ###        
        EM_glintmask = (time < 1516) | (time > 1517.75) #earth glint at start of orbit        
        EM_glintmask2 = (time < 1529) | (time > 1536.2) #earth glint at start of orbit
        mask = (EM_glintmask) & (EM_glintmask2)
        ###        
    if SectorNum==9:
        mdumps=3.125
        #orbit start times
        t_0 = 1543.21648
        t_1 = 1557.00080
        ###        
        EM_glintmask = (time < 1542.21) | (time > 1544.0) #earth glint at start of orbit                
        EM_glintmask2 = (time < 1556.0) | (time > 1557.75) #earth glint at start of orbit
        mask = (EM_glintmask) & (EM_glintmask2)
        ###        
    if SectorNum==10:
        mdumps=3.125
        #orbit start times
        t_0 = 1569.43176
        t_1 = 1582.76231
        ###        
        EM_glintmask = (time < (1569.43-1)) | (time > (1571.0)) #earth glint at start of orbit
        EM_glintmask2 = (time < (1582.76-1)) | (time > (1584.5)) #earth glint at start of orbit
        mask = (EM_glintmask) & (EM_glintmask2)
        ###        
    if SectorNum==11:
        mdumps=3.125
        #orbit start times
        t_0 = 1596.77203
        t_1 = 1610.77620
        ###        
        EM_glintmask =  (time < (1596.77-1)) | (time > (1599)) #earth glint at start of orbit        
        EM_glintmask2 = (time < (1610)) | (time > (1613.75)) #earth glint at start of orbit
        mask = (EM_glintmask) & (EM_glintmask2)
        ###
    if SectorNum==12:
        mdumps=3.125  
        #orbit start times
        t_0 = 1624.94979
        t_1 = 1640.03312
        ###        
        EM_glintmask =  (time < (1624.949-1)) | (time > (1624.949+0.75)) #earth glint at start of orbit       
        EM_glintmask2 = (time < (1640.03-1)) |  (time > (1640.03+0.75)) #earth glint at start of orbit
        mask = (EM_glintmask) & (EM_glintmask2)
        ###
        ###        
    if SectorNum==13: 
        mdumps=3.375
        #orbit start times
        t_0 = 1653.91505
        t_1 = 1668.61921
        ###        
        jittermask = (time < 1665.2983) | (time > 1665.3501) 
        EM_glintmask = (time < (1653.915-1)) | (time > (1653.915+0.75)) #earth glint at start of orbit
        EM_glintmask2 = (time < (1668.61903-1)) | (time > (1668.619+0.75)) #earth glint at start of orbit
        mask = (jittermask) & (EM_glintmask) & (EM_glintmask2)
        ###
    if SectorNum==14:
        mdumps=4.4
        #orbit start times
        t_0 = 1683.34838
        t_1 = 1697.33865 
        #no jittermask for Sector 14 (...yet)
        mask = np.ones(len(time), dtype=bool) #makes them all TRUE
        ###        
        ### NEED TO ADD NORTH SECTORS!!!! ALSO EXTENDED MISSION SECTORS!
        ###
    if SectorNum==31:
        # DRN says 1 mdump occurred half way in each orbit...
        #orbit start times
        t_0 = 2144.50927
        t_0end =  2157.45371
        t_1 = 2158.85648
        t_1end = 2169.94398        
        mdumps = 0
        mask = np.ones(len(time), dtype=bool) #makes them all TRUE
        
        
    ### Need a special case for Sector 31 (and maybe others...)
    if SectorNum==31:
        mask_mdump = np.ones_like(time, dtype=bool)      
        timedump1 =(t_0end - t_0)/2 # halfway in orbit like DRN says...
        timedump2 =(t_1end - t_1)/2 # halfway in orbit like DRN says...
        for t in range(len(time)):
            if (time[t]>(timedump1-before_after_in_minutes/(60*24))) & (time[t]<(timedump1+before_after_in_minutes/(60*24))):
                    mask_mdump[t]=0
            if (time[t]>(timedump2-before_after_in_minutes/(60*24))) & (time[t]<(timedump2+before_after_in_minutes/(60*24))):
                    mask_mdump[t]=0
    else:
        Num_mdumps = int(np.round((np.nanmax(time) - np.nanmin(time))/mdumps,2))+1
        mask_mdump = np.ones_like(time, dtype=bool)      
        for N in range(Num_mdumps):    
            for t in range(len(time)):
                if t_0+(N)*mdumps<t_1:
                    timedump=t_0+(N)*mdumps
                    if (time[t]>(timedump-before_after_in_minutes/(60*24))) & (time[t]<(timedump+before_after_in_minutes/(60*24))):
                        mask_mdump[t]=0
                if t_1+(N)*mdumps <np.nanmax(time):
                    timedump=t_1+(N)*mdumps
                    if (time[t]>(timedump-before_after_in_minutes/(60*24))) & (time[t]<(timedump+before_after_in_minutes/(60*24))):
                        mask_mdump[t]=0
    ###
    ###    

        
    
    combo_mask = (mask_mdump) & (mask) 
    combo_mask2 = np.where(combo_mask==True)[0] #throwing out momentum dumps and bad data (False booleans)        
    mask_mdump=combo_mask2
    ###
    ###
    print('time before/after momentum dumps to remove: ',before_after_in_minutes, 'minutes')
    print('before momentumdump removal: ',len(time))    
    print('after: ',len(mask_mdump))    
#     return mask,mask_mdump,mdumps,t_0,t_1 #mask,mdumps,start of 1st/2nd orbits
    return mask_mdump,mdumps,t_0,t_1 #mask,mdumps,start of 1st/2nd orbits

####Still on Step 4 here...
def Applying_Mdump_removal(ID,Sector,Camera,CCD,before_after_in_minutes,SAP_LC,flux,savelcpath,verbose):
    import pandas as pd
    ###
    ###
    ### read in data from dataframe
    rawtime   = np.array(SAP_LC['Time'].to_list()) 
    sap_flux  = np.array(SAP_LC['SAP Flux'].to_list()) 
    sap_error = np.array(SAP_LC['SAP Error'].to_list()) 
    bkg_flux  = np.array(SAP_LC['Background Flux'].to_list()) 
    ###
    ###    
    ### making mask for momentum dumps and bad regions of data
    mask_mdump,mdumps,t_0,t_1 = momentumdump_removal(Sector,Camera,CCD,before_after_in_minutes, rawtime)
    ###
    ### 
    ### Applying mask and then saving masked data
    time=rawtime[mask_mdump]
    sap_flux=sap_flux[mask_mdump]
    sap_error=sap_error[mask_mdump]
    bkg_flux=bkg_flux[mask_mdump]
    flux=flux[mask_mdump] # removing frames with momentumdumps
    ###
    if verbose==True:
        print('clipped len check:', len(time),len(sap_flux),len(sap_error))
    clippedRAWLC_df = pd.DataFrame({"Time":time, "SAP Flux": sap_flux, "SAP Error":sap_error,"Background Flux":bkg_flux})
    ###
    clippedRAWLC_df.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_RAW_LC_systematics_removed.txt",index=False)
    ###
    if verbose==True:
        print('len check b',' T', len(time),' SAP',len(sap_flux), ' E ', len(sap_error))
    ###
    ### 
    return mask_mdump, mdumps,t_0,t_1, flux, SAP_LC, clippedRAWLC_df





# helper functions for smoothing and outlier removal based on stellar parameters


def SMA_AU_from_Period_to_stellar(Period,R_star,M_star):
    #assumes circular orbit
    #using Kepler's third law, calculate SMA
    #solar units
    RS = 6.955*10.0**10.0 #cm, solar radius
    MS = 1.989*10.0**33.0 #g, solar mass

    G = 6.6743*10.0**(-8.0) #cm^3 per g per s^2

    R = R_star*RS
    M = M_star*MS
    P=Period*60.0*24.0*60.0 #in seconds

    #SMA
    SMA_cm = ((G*M*(P**2))/(4*(np.pi**2)))**(1/3)   

    #note R_star is already in solar units so we need to convert to cm using
    # solar radius as a constant
    Stellar_Radius = R #now in cm

    SMA = SMA_cm / Stellar_Radius #now unitless (cm / cm)
    return SMA, SMA_cm

def Tdur(Period, R_star,M_star, R_planet_RE):    
    RE = 6.378*10.0**8 #cm
    RS = 6.955 *10.0**10 #cm    
    A = Period/np.pi #in days
        
    SMA_cm = SMA_AU_from_Period_to_stellar(Period,R_star,M_star)[1]
    
    B =(R_star*RS +R_planet_RE*RE)/ SMA_cm #in cm
    
    T_dur = A*np.arcsin(B) #in days
    return T_dur

def stellar_insolation(Ps, Ms, Rs, Teff):
    '''Return the insolation in Earth units'''
    #these are all cgs units:  cm, g, s    
    G=6.6743*10**(-8)
    Msun=1.989 *10**33
    Mearth=5.974*10**27
    Rsun=6.955*10**10
    Rearth=6.378*10**8
    AU=1.496*10**13
    pc=3.086 *10**18
    
    S0 = 1367#/1e0 # Watts per m^2 in watts per cm^2
    
    sigma = 5.67e-8

    L = 4*np.pi*(Rs*RS)**2 * sigma*Teff**4
    LE = 4*np.pi*(1*RS)**2 * sigma*5777**4
    
    Ps = np.ascontiguousarray(Ps)
    
    smas,smas_cm = SMA_AU_from_Period_to_stellar(Ps,Rs,Ms)
    smasE,smasE_cm = SMA_AU_from_Period_to_stellar(365.25,1,1)
    
    S= L / (4*np.pi*smas_cm**2)
    SE = LE / (4*np.pi*smasE_cm**2)
    
    
    S_in_earth_units = S/SE
    
    return S, S_in_earth_units


def BWMC_auto(ID,Sector,input_LC,savelcpath): #bt = break tolerance, pipeline uses window_size/2.0
    from wotan import flatten    
    import numpy as np
    
    #read in LC data
    time = np.array(input_LC['Time'].to_list())
    flux_raw = np.array(input_LC['Flux'].to_list())
    flux_error = np.array(input_LC['Error'].to_list())
    flux_model = np.array(input_LC['Model'].to_list())    
    centx = np.array(input_LC["Centroid X Positions"].to_list())
    centy = np.array(input_LC["Centroid Y Positions"].to_list())
    sap_flux=np.array(input_LC['SAP Flux'].to_list())
    sap_error=np.array(input_LC['SAP Error'].to_list())
    
    LCDur=(np.nanmax(time) - np.nanmin(time))
    maxP = LCDur/2 #longest period for 2 transits in a light curve (~14 days for TESS single sector LCs)
    R_planet_RE = 1
    
    #getting stellar parameters from TIC
    from transitleastsquares import catalog_info
    
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    
    # we want to keep the longest transit for an Earth-like planet for a single sector of data
    # using stellar parameters to determine transit duration
    window_size = 3*Tdur(maxP, R_star,M_star, R_planet_RE)
    
    flatten_lc, trend_lc = flatten(time, flux_raw, window_length=window_size, return_trend=True, break_tolerance=window_size/2.0,method='biweight',robust=True)
    T=time
    F=flatten_lc
    FE=flux_error
    #
    #checking for NaNs
    nanmask = np.where(np.isfinite(F)==True)[0]
    T = T[nanmask]
    F = F[nanmask]
    FE =FE[nanmask]
    F_raw = flux_raw[nanmask]
    flux_model = flux_model[nanmask]
    trend_lc=trend_lc[nanmask]
    centx=centx[nanmask]
    centy=centy[nanmask]   
    sap_flux=sap_flux[nanmask]
    sap_error=sap_error[nanmask]    
    #
    #redefining output terms
    time = T
    flux_detrended = F
    det_error = FE/trend_lc
    ###
    ###
    
    Det_LC = pd.DataFrame({'Time':time, 'SAP Flux':sap_flux,'SAP Error':sap_error,'PLD Flux':F_raw,'PLD Error':FE,'PLD Model':flux_model,'Detrended Flux':flux_detrended, 'Detrended Error':det_error,'Fitted Trend':trend_lc,"Centroid X Positions":centx,"Centroid Y Positions":centy})
    Det_LC.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_DET_LC.txt")
    return Det_LC, nanmask


#helper function for PLD modeler
def solver(X,flux):
    import warnings,sys
    # warnings.filterwarnings(action='once') #useful to see a warning once but that's it
    warnings.simplefilter("ignore", category=PendingDeprecationWarning)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    import numpy as np
    from numpy.linalg import LinAlgError
    a=np.dot(X.T,X)
    b=np.dot(X.T,flux)
    try:
        w=np.linalg.solve(a,b) 
        model=np.dot(X,w)
    except LinAlgError:
        print('np.linalg.solve gave Singular Matrix problem. Using np.linalg.lstsq')
        w=np.linalg.lstsq(a, b)[0]
        model=np.dot(X,w) 
    return model


def PLD_model_old(ID,Sector,flux,pix_mask,input_LC,savelcpath,pld_order=3, n_pca_terms=3):
    import warnings,sys
    # warnings.filterwarnings(action='once') #useful to see a warning once but that's it
    warnings.simplefilter("ignore", category=PendingDeprecationWarning)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    ###making solver function for SVD or Least Squares solutions
    import numpy as np
    from numpy.linalg import LinAlgError
    from sklearn.decomposition import PCA
    from itertools import combinations_with_replacement as CwR
    
    #read in LC data:
    time  = np.array(input_LC['Time'])
    input_flux  = np.array(input_LC['Flux'])
    error = np.array(input_LC['Error'])
    centx = np.array(input_LC["Centroid X Positions"])
    centy = np.array(input_LC["Centroid Y Positions"])    
    
    aperture = [pix_mask for i in range(len(flux))]
    fpix_rs = (flux*aperture).reshape(len(flux),-1)
    fpix_ap = np.zeros((len(flux),len(np.delete(fpix_rs[0],np.where(np.isnan(fpix_rs[0]))))))

    for c in range(len(fpix_rs)):
        naninds = np.where(np.isnan(fpix_rs[c]))
        fpix_ap[c] = np.delete(fpix_rs[c],naninds)

    newflux = np.sum(fpix_ap,axis=1)
    newX = fpix_ap / newflux.reshape(-1,1)
    
    # 1st order PLD
    f1 = fpix_ap / newflux.reshape(-1,1)
    pca = PCA(n_components = n_pca_terms)
    X1 = pca.fit_transform(f1)
    
    # Nth order PLD
    XN=[]
    for order in range(2,pld_order+1):
        fN = np.prod(list(CwR(X1.T, order)), axis=1).T #axis=+1?
        pca = PCA(n_components = n_pca_terms)
        X_n = pca.fit_transform(fN)
        XN.append(X_n)

    X_pld = np.concatenate(XN, axis=1) #axis=+1?
    
    
    pld_model = solver(X_pld , input_flux)+ np.nanmedian(input_flux)
    pld_detrended = input_flux/pld_model 
    pld_error = error/pld_model

    nanmask = np.where(np.isfinite(pld_detrended)==True)[0] #make nanmask on last step (PLD/DET/SAP)
    time = time[nanmask]
    input_flux = input_flux[nanmask]
    error = error[nanmask]
    pld_detrended = pld_detrended[nanmask]
    pld_error =pld_error[nanmask]
    pld_model = pld_model[nanmask]
    centx = centx[nanmask]
    centy = centy[nanmask]    

   
    PLD_LC = pd.DataFrame({'Time':time, 'SAP Flux':input_flux,'SAP Error':error,\
                           'PLD Flux':pld_detrended, 'PLD Error': pld_error,\
                           'PLD Model':pld_model, 'Centroid X Positions':centx,\
                          'Centroid Y Positions':centy})
    PLD_LC.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_PLD_LC.txt")
    return PLD_LC #,n_pca_terms



def PLD_model(ID,Sector,flux,pix_mask,input_LC,savelcpath,pld_order=3, n_pca_terms=3,cadence_mask=None):
    import warnings,sys
    # warnings.filterwarnings(action='once') #useful to see a warning once but that's it
    warnings.simplefilter("ignore", category=PendingDeprecationWarning)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
        
    from numpy.linalg import LinAlgError
    from itertools import combinations_with_replacement as multichoose
    import numpy as np
    
    pld_pixel_mask = pix_mask

    time  = np.array(input_LC['Time'])
    input_flux  = np.array(input_LC['Flux'])
    error = np.array(input_LC['Error'])
    centx = np.array(input_LC["Centroid X Positions"])
    centy = np.array(input_LC["Centroid Y Positions"]) 

    # create nan mask
    rawflux=input_flux
    rawflux_err=error
    nanmask = np.isfinite(time)
    nanmask &= np.isfinite(rawflux)
    nanmask &= np.isfinite(rawflux_err)
    nanmask &= np.abs(rawflux_err) > 1e-12

    # mask out nan values
    rawflux = rawflux[nanmask]
    rawflux_err = rawflux_err[nanmask]
    flux = flux[nanmask]
    # flux_err = flux_err[nanmask] # needs 'FLUX ERROR' from hdu...
    time = time[nanmask]

    # parse the PLD aperture mask
#         pld_pixel_mask = self.tpf._parse_aperture_mask(pld_aperture_mask) #not sure what this is

    # find pixel bounds of aperture on cutout
    xmin, xmax = min(np.where(pld_pixel_mask)[0]),  max(np.where(pld_pixel_mask)[0])
    ymin, ymax = min(np.where(pld_pixel_mask)[1]),  max(np.where(pld_pixel_mask)[1])

    # crop data cube to include only desired pixels
    # this is required for superstamps to ensure matrix is invertable
    flux_crop = flux[:, xmin:xmax+1, ymin:ymax+1]
    #flux_err_crop = self.flux_err[:, xmin:xmax+1, ymin:ymax+1] # needs 'FLUX ERROR' from hdu...
    aperture_crop = pld_pixel_mask[xmin:xmax+1, ymin:ymax+1]

    # first order PLD design matrix
    pld_flux = flux_crop[:, aperture_crop]
    f1 = np.reshape(pld_flux, (len(pld_flux), -1))
    X1 = f1 / np.nansum(pld_flux, axis=-1)[:, None]
    # No NaN pixels
    X1 = X1[:, np.isfinite(X1).all(axis=0)]

    # higher order PLD design matrices
    X_sections = [np.ones((len(flux_crop), 1)), X1]
    for i in range(2, pld_order+1):
        f2 = np.product(list(multichoose(X1.T, pld_order)), axis=1).T
        try:
            # We use an optional dependency for very fast PCA (fbpca).
            # If the import fails we will fall back on using the slower `np.linalg.svd`
            from fbpca import pca
            #print('f2: ',np.shape(f2), 'n_pca_terms: ',n_pca_terms)
            components, _, _ = pca(f2, n_pca_terms)
        except ImportError :
            log.error("PLD uses the `fbpca` package. You can pip install "
                      "with `pip install fbpca`. Using `np.linalg.svd` "
                      "instead.")
        except AssertionError as ae:
            print(ae)
            print('problem with fbpca, using np.linalg.svd instead')
            print('')
            components, _, _ = np.linalg.svd(f2)
        X_n = components[:, :n_pca_terms]
        X_sections.append(X_n)

    # Create the design matrix X by stacking X1 and higher order components, and
    # adding a column vector of 1s for numerical stability (see Luger et al.).
    # X has shape (n_components_first + n_components_higher_order + 1, n_cadences)
    X = np.concatenate(X_sections, axis=1)

    # set default transit mask
    if cadence_mask is None:
        cadence_mask = np.ones_like(time, dtype=bool)
    M = lambda x: x[cadence_mask]

    # mask transits in design matrix (if requested by user)
    MX = M(X)


    # compute the coefficients C on the basis vectors;
    # the PLD design matrix will be dotted with C to solve for the noise model.
    ivar = 1.0 / M(rawflux_err)**2 # inverse variance
    A = np.dot(MX.T, MX * ivar[:, None])
    B = np.dot(MX.T, M(rawflux) * ivar)

    # apply prior to design matrix weights for numerical stability
    A[np.diag_indices_from(A)] += 1e-8
    ###    
    try:
        C=np.linalg.solve(A,B) 
    except LinAlgError:
        print('np.linalg.solve gave Singular Matrix problem. Using np.linalg.lstsq')
        C=np.linalg.lstsq(A,B)[0]
    ###
    # compute detrended light curve
    pld_model = np.dot(X, C)
    pld_detrended = rawflux - (pld_model)+ np.nanmedian(rawflux) #as used in Old LK
    pld_error = rawflux_err
    
#     pld_detrended = rawflux/pld_model + np.nanmedian(rawflux)
#     pld_error = rawflux_err/pld_model
    
    nanmask = np.where(np.isfinite(pld_detrended)==True)[0] #make nanmask on last step (PLD/DET/SAP)
    time = time[nanmask]
    input_flux = rawflux[nanmask]
    error = rawflux_err[nanmask]
    pld_detrended = pld_detrended[nanmask]
    pld_error =pld_error[nanmask]
    pld_model = pld_model[nanmask]
    centx = centx[nanmask]
    centy = centy[nanmask]   
    
    
    print('PLD len check: T',len(time),' SAPF ',len(rawflux),' SAPE ',len(rawflux_err),' PLD M ',len(pld_model),' PLD F', len(pld_detrended),' PLD E ',len(pld_error),' centx ',len(centx),' centy ',len(centy) )
    PLD_LC = pd.DataFrame({'Time':time, 'SAP Flux':input_flux,'SAP Error':error,\
                       'PLD Flux':pld_detrended, 'PLD Error': pld_error,\
                       'PLD Model':pld_model, 'Centroid X Positions':centx,\
                      'Centroid Y Positions':centy})
    PLD_LC.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_PLD_LC.txt")
    return PLD_LC


def singleoutliers(data, stepsize=1):
    single=np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    single = np.array(list(filter(lambda x : len(x) <= 1, single))).flatten() 
    single = np.sort(np.array([np.int64(x) for x in single]))
    return single

def consecutive(data, stepsize=1):
    consec=np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    c=list(filter(lambda x : len(x) > 1, consec))
    newc=[]
    for x in range(len(c)):
        newc=np.append(newc,c[x])
    consec = np.sort(np.array([np.int64(x) for x in newc]))
    return consec


def outlier_removal_old(ID,Sector,input_LC, remove_outliers, window_size_in_days, Nsigma_low,Nsigma_high,method,savelcpath,verbose):    
    import numpy as np
    import pandas as pd
    from scipy import stats
    ###
    ###
    
    #needs to be flexible to allow different combos of PLD, DET and SAP    
    #read in input data
    time = np.array(input_LC['Time'])
    sap_flux = np.array(input_LC['SAP Flux'])
    sap_error= np.array(input_LC['SAP Error'])
    det_flux = np.array(input_LC['Detrended Flux'])
    det_error = np.array(input_LC['Detrended Error'])
    trend_lc =  np.array(input_LC['Fitted Trend'])
    pld_flux = np.array(input_LC['PLD Flux'])
    pld_error = np.array(input_LC['PLD Error'])
    pld_model = np.array(input_LC['PLD Model'])   
    centx = np.array(input_LC['Centroid X Positions'])   
    centy = np.array(input_LC['Centroid Y Positions'])       
    
    
    preclipLC_df = pd.DataFrame({"Time":time, "SAP Flux": sap_flux, "SAP Error":sap_error, "Detrended Flux":det_flux, "Detrended Error":det_error,"Fitted Trend":trend_lc,"PLD Flux":pld_flux,"PLD Error":pld_error, "PLD Model":pld_model,"Centroid X Positions":centx,"Centroid Y Positions":centy})
    preclipLC_df.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_PLD_LC_preclipped.txt",index=False)
    ###
    ###    
    if remove_outliers=='yes':
        #needs to be flexible to allow different combos of PLD, DET and SAP    
        # use last step in operation's flux (SAP, DET or PLD)
        flux = pld_flux
        error = pld_error
        ###
        if method=='global':
            #global noise level
#             flux_threshold_lo = np.nanmedian(flux)-Nsigma_low*np.nanstd(flux)
#             flux_threshold_hi = np.nanmedian(flux)+Nsigma_high*np.nanstd(flux)
            flux_threshold_lo = np.nanmedian(flux)-Nsigma_low*stats.median_absolute_deviation(flux)
            flux_threshold_hi = np.nanmedian(flux)+Nsigma_low*stats.median_absolute_deviation(flux)
        ###
        cad = np.nanmedian(np.diff(time))
        Npts = int(np.round(((window_size_in_days/cad))+1,1))
        size=len(time)
        window_size=Npts
        if window_size > size:
            window_size = size
        bins = int(size/Npts)+2
        good_ind_lo=[]
        bad_ind_lo=[]
        good_ind_hi=[]
        bad_ind_hi=[]
        for N in range(1,bins):     
#             print('Bin '+str(N))
            time_in_window = time[window_size*(N-1):N*window_size]
            flux_in_window = flux[window_size*(N-1):N*window_size]
            err_in_window = error[window_size*(N-1):N*window_size]
            if method=="local":
                #local noise
#                 flux_threshold_lo = np.nanmedian(flux_in_window)-Nsigma_low*np.nanstd(flux_in_window)
#                 flux_threshold_hi = np.nanmedian(flux_in_window)+Nsigma_low*np.nanstd(flux_in_window)
                flux_threshold_lo = np.nanmedian(flux_in_window)-Nsigma_low*stats.median_absolute_deviation(flux_in_window)
                flux_threshold_hi = np.nanmedian(flux_in_window)+Nsigma_low*stats.median_absolute_deviation(flux_in_window)
            ###
            #global noise
            ind_lo = np.where(flux_in_window < flux_threshold_lo)[0]
            ind_hi = np.where(flux_in_window > flux_threshold_hi)[0]          
            #print('noise levels')
            #print(ind_lo)
            #print(ind_hi)
            #print(' ')
            ###
            single_lo = singleoutliers(ind_lo)
            single_hi = singleoutliers(ind_hi)
            ###
            consec_lo = consecutive(ind_lo)
            consec_hi = consecutive(ind_hi)
            ###
            #map elements identified in window to full input array
            if len(consec_hi)>0:
                good_ind_hi_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[consec_hi],4),side='left')
                good_ind_hi=np.append(good_ind_hi,good_ind_hi_temp) 
            if len(single_hi)>0:
                bad_ind_hi_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[single_hi],4),side='left')
                bad_ind_hi=np.append(bad_ind_hi,bad_ind_hi_temp)
            ###
            if len(consec_lo)>0:
                good_ind_lo_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[consec_lo],4),side='left')
                good_ind_lo=np.append(good_ind_lo,good_ind_lo_temp)         
            if len(single_lo)>0:
                bad_ind_lo_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[single_lo],4),side='left')
                bad_ind_lo=np.append(bad_ind_lo,bad_ind_lo_temp)
            ###
            #convert matched elements to intergers
            bad_ind_lo=np.sort(np.array([np.int64(x) for x in bad_ind_lo]))
            bad_ind_hi=np.sort(np.array([np.int64(x) for x in bad_ind_hi]))
            good_ind_lo=np.sort(np.array([np.int64(x) for x in good_ind_lo]))
            good_ind_hi=np.sort(np.array([np.int64(x) for x in good_ind_hi]))    
            ###
            ###
            ###
            ### append both high/low results to final arrays
            bad_ind = np.sort(np.append(bad_ind_lo,bad_ind_hi).flatten())
            # good_ind = np.sort(np.append(good_ind_lo,good_ind_hi).flatten())
            ###
            ### MAAAAAYBE we don't want to keep positive outliers after all...
            good_ind = np.sort((good_ind_lo).flatten())
            temp_bad_ind = np.append(bad_ind_lo,bad_ind_hi)
            bad_ind = np.sort(np.append(temp_bad_ind,good_ind_hi).flatten())
            ###
            ###
            bad_ind=np.sort(np.array([np.int64(x) for x in bad_ind]))    
            good_ind=np.sort(np.array([np.int64(x) for x in good_ind]))    
            ###
            #convert matched elements to intergers
            ###
        ###
        #finally, remove bad points (keep good points) and make new time,flux, error arrays
        ###
        badtime = np.array([i for j, i in enumerate(time) if j in bad_ind])
        badsap_flux = np.array([i for j, i in enumerate(sap_flux) if j in bad_ind])
        baderror = np.array([i for j, i in enumerate(sap_error) if j in bad_ind])
        badflux_detrended = np.array([i for j, i in enumerate(det_flux) if j in bad_ind])
        baddet_error = np.array([i for j, i in enumerate(det_error) if j in bad_ind])
        badtrend_lc = np.array([i for j, i in enumerate(trend_lc) if j in bad_ind])
        badpld_detrended = np.array([i for j, i in enumerate(pld_flux) if j in bad_ind])
        badpld_model = np.array([i for j, i in enumerate(pld_model) if j in bad_ind])
        badpld_error = np.array([i for j, i in enumerate(pld_error) if j in bad_ind])
        badcentx= np.array([i for j, i in enumerate(centx) if j in bad_ind])
        badcenty= np.array([i for j, i in enumerate(centy) if j in bad_ind])        
        
        goodtime = np.array([i for j, i in enumerate(time) if j in good_ind])
        goodsap_flux = np.array([i for j, i in enumerate(sap_flux) if j in good_ind])
        gooderror = np.array([i for j, i in enumerate(sap_error) if j in good_ind])
        goodflux_detrended = np.array([i for j, i in enumerate(det_flux) if j in good_ind])
        gooddet_error = np.array([i for j, i in enumerate(det_error) if j in good_ind])
        goodtrend_lc = np.array([i for j, i in enumerate(trend_lc) if j in good_ind])
        goodpld_detrended = np.array([i for j, i in enumerate(pld_flux) if j in good_ind])
        goodpld_model = np.array([i for j, i in enumerate(pld_model) if j in good_ind])
        goodpld_error = np.array([i for j, i in enumerate(pld_error) if j in good_ind])
        goodcentx= np.array([i for j, i in enumerate(centx) if j in good_ind])
        goodcenty= np.array([i for j, i in enumerate(centy) if j in good_ind])
        
        
        
        time = np.array([i for j, i in enumerate(time) if j not in bad_ind])
        sap_flux = np.array([i for j, i in enumerate(sap_flux) if j not in bad_ind])
        sap_error = np.array([i for j, i in enumerate(sap_error) if j not in bad_ind])
        det_flux = np.array([i for j, i in enumerate(det_flux) if j not in bad_ind])
        det_error = np.array([i for j, i in enumerate(det_error) if j not in bad_ind])
        trend_lc = np.array([i for j, i in enumerate(trend_lc) if j not in bad_ind])
        pld_flux = np.array([i for j, i in enumerate(pld_flux) if j not in bad_ind])
        pld_error = np.array([i for j, i in enumerate(pld_error) if j not in bad_ind])
        pld_model = np.array([i for j, i in enumerate(pld_model) if j not in bad_ind])        
        centx = np.array([i for j, i in enumerate(centx) if j not in bad_ind])        
        centy = np.array([i for j, i in enumerate(centy) if j not in bad_ind])         
        ###
        ###
        ###
        import pandas as pd
        #saving flagged data points
        print('saving flagged outliers')
        bad_ind_DF = pd.DataFrame({"Time":badtime, "SAP Flux": badsap_flux, "SAP Error":baderror, "Detrended Flux":badflux_detrended, "Detrended Error":baddet_error,"Fitted Trend":badtrend_lc,"PLD Flux":badpld_detrended,"PLD Model":badpld_model,"PLD Error":badpld_error,"Centroid X Positions":badcentx,"Centroid Y Positions":badcenty})
        good_ind_DF = pd.DataFrame({"Time":goodtime, "SAP Flux": goodsap_flux, "SAP Error":gooderror, "Detrended Flux":goodflux_detrended, "Detrended Error":gooddet_error,"Fitted Trend":goodtrend_lc,"PLD Flux":goodpld_detrended,"PLD Model":goodpld_model,"PLD Error":goodpld_error,"Centroid X Positions":goodcentx,"Centroid Y Positions":goodcenty})
        bad_ind_DF.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bad_outliers.txt",index=False)
        good_ind_DF.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_good_outliers.txt",index=False)
        ###
        ###
        ###
        if verbose==True:
            print('after outlier removal: ',' T', len(time),' Det',len(det_flux), ' trend',len(trend_lc), 'PLD model',len(pld_model),' PLD F',len(pld_flux),' SAP E', len(sap_error))
    ###
    if remove_outliers=='no':
        time = np.ascontiguousarray(time, dtype=np.float64)
        sap_flux = np.ascontiguousarray(sap_flux, dtype=np.float64)
        sap_error = np.ascontiguousarray(sap_error, dtype=np.float64)
        det_flux = np.ascontiguousarray(det_flux, dtype=np.float64)
        det_error= np.ascontiguousarray(det_error, dtype=np.float64)
        trend_lc = np.ascontiguousarray(trend_lc, dtype=np.float64)                        
        pld_flux = np.ascontiguousarray(pld_flux, dtype=np.float64)        
        pld_error = np.ascontiguousarray(pld_error, dtype=np.float64)
        pld_model = np.ascontiguousarray(pld_model, dtype=np.float64)
        centx = np.ascontiguousarray(centx, dtype=np.float64)
        centy = np.ascontiguousarray(centy, dtype=np.float64)        
        if verbose==True:
            print('skipping outlier removal:',' T', len(time),' Det',len(det_flux), ' trend',len(trend_lc), 'PLD model',len(pld_model),' PLD F',len(pld_flux),' SAP E', len(sap_error))        
    ###
    ###
    ###                  
    nanmask = np.where(np.isfinite(pld_flux)==True)[0]
    time = time[nanmask]
    sap_flux = sap_flux[nanmask]
    sap_error=sap_error[nanmask]
    det_flux = det_flux[nanmask]
    det_error=det_error[nanmask]
    trend_lc=trend_lc[nanmask] 
    pld_flux = pld_flux[nanmask]    
    pld_error=pld_error[nanmask]
    pld_model=pld_model[nanmask]
    centx=centx[nanmask]
    centy=centy[nanmask]    
    
    
    #needs to be flexible to allow different combos of PLD, DET and SAP    
    LC_df = pd.DataFrame({"Time":time, "SAP Flux": sap_flux, "SAP Error":sap_error, "Detrended Flux":det_flux, "Detrended Error":det_error,"Fitted Trend":trend_lc,"PLD Flux":pld_flux,"PLD Error":pld_error,"PLD Model":pld_model,"Centroid X Positions":centx,"Centroid Y Positions":centy})
    ###
    LC_df.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_final_LC.txt",index=False)
    ###
    ###
    return LC_df, good_ind_DF, bad_ind_DF, preclipLC_df



def outlier_removal(ID,Sector,input_LC, remove_outliers, Nsigma_low,Nsigma_high,savelcpath,verbose,method='local',window_size_in_days=None):    
    import numpy as np
    import pandas as pd
    from scipy import stats
    ###
    ###
    
    #needs to be flexible to allow different combos of PLD, DET and SAP    
    #read in input data
    time = np.array(input_LC['Time'])
    sap_flux = np.array(input_LC['SAP Flux'])
    sap_error= np.array(input_LC['SAP Error'])
    det_flux = np.array(input_LC['Detrended Flux'])
    det_error = np.array(input_LC['Detrended Error'])
    trend_lc =  np.array(input_LC['Fitted Trend'])
    pld_flux = np.array(input_LC['PLD Flux'])
    pld_error = np.array(input_LC['PLD Error'])
    pld_model = np.array(input_LC['PLD Model'])   
    centx = np.array(input_LC['Centroid X Positions'])   
    centy = np.array(input_LC['Centroid Y Positions'])       
    
    
    preclipLC_df = pd.DataFrame({"Time":time, "SAP Flux": sap_flux, "SAP Error":sap_error, "Detrended Flux":det_flux, "Detrended Error":det_error,"Fitted Trend":trend_lc,"PLD Flux":pld_flux,"PLD Error":pld_error, "PLD Model":pld_model,"Centroid X Positions":centx,"Centroid Y Positions":centy})
    preclipLC_df.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_PLD_LC_preclipped.txt",index=False)
    ###
    ###    
    
    #defining window for removing outliers in local noise
    LCDur=(np.nanmax(time) - np.nanmin(time))
    maxP = LCDur/2 #longest period for 2 transits in a light curve (~14 days for TESS single sector LCs)
    R_planet_RE = 1
    
    #getting stellar parameters from TIC
    from transitleastsquares import catalog_info
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    
    
    if window_size_in_days==None:
        # we want to keep the longest transit for an Earth-like planet for a single sector of data
        # using stellar parameters to determine transit duration
        window_size_in_days = 3*Tdur(maxP, R_star,M_star, R_planet_RE)
    
    if remove_outliers=='yes':
        #needs to be flexible to allow different combos of PLD, DET and SAP    
        # use last step in operation's flux (SAP, DET or PLD)
        flux = pld_flux
        error = pld_error
        ###
        if method=='global':
            #global noise level
#             flux_threshold_lo = np.nanmedian(flux)-Nsigma_low*np.nanstd(flux)
#             flux_threshold_hi = np.nanmedian(flux)+Nsigma_high*np.nanstd(flux)
            flux_threshold_lo = np.nanmedian(flux)-Nsigma_low*stats.median_absolute_deviation(flux)
            flux_threshold_hi = np.nanmedian(flux)+Nsigma_low*stats.median_absolute_deviation(flux)
        ###
        cad = np.nanmedian(np.diff(time))
        Npts = int(np.round(((window_size_in_days/cad))+1,1))
        size=len(time)
        window_size=Npts
        if window_size > size:
            window_size = size
        bins = int(size/Npts)+2
        good_ind_lo=[]
        bad_ind_lo=[]
        good_ind_hi=[]
        bad_ind_hi=[]
        for N in range(1,bins):     
#             print('Bin '+str(N))
            time_in_window = time[window_size*(N-1):N*window_size]
            flux_in_window = flux[window_size*(N-1):N*window_size]
            err_in_window = error[window_size*(N-1):N*window_size]
            if method=="local":
                #local noise
#                 flux_threshold_lo = np.nanmedian(flux_in_window)-Nsigma_low*np.nanstd(flux_in_window)
#                 flux_threshold_hi = np.nanmedian(flux_in_window)+Nsigma_low*np.nanstd(flux_in_window)
                flux_threshold_lo = np.nanmedian(flux_in_window)-Nsigma_low*stats.median_absolute_deviation(flux_in_window)
                flux_threshold_hi = np.nanmedian(flux_in_window)+Nsigma_low*stats.median_absolute_deviation(flux_in_window)
            ###
            #global noise
            ind_lo = np.where(flux_in_window < flux_threshold_lo)[0]
            ind_hi = np.where(flux_in_window > flux_threshold_hi)[0]          
            #print('noise levels')
            #print(ind_lo)
            #print(ind_hi)
            #print(' ')
            ###
            single_lo = singleoutliers(ind_lo)
            single_hi = singleoutliers(ind_hi)
            ###
            consec_lo = consecutive(ind_lo)
            consec_hi = consecutive(ind_hi)
            ###
            #map elements identified in window to full input array
            if len(consec_hi)>0:
                good_ind_hi_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[consec_hi],4),side='left')
                good_ind_hi=np.append(good_ind_hi,good_ind_hi_temp) 
            if len(single_hi)>0:
                bad_ind_hi_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[single_hi],4),side='left')
                bad_ind_hi=np.append(bad_ind_hi,bad_ind_hi_temp)
            ###
            if len(consec_lo)>0:
                good_ind_lo_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[consec_lo],4),side='left')
                good_ind_lo=np.append(good_ind_lo,good_ind_lo_temp)         
            if len(single_lo)>0:
                bad_ind_lo_temp = np.searchsorted(np.around(time,4), np.around(time_in_window[single_lo],4),side='left')
                bad_ind_lo=np.append(bad_ind_lo,bad_ind_lo_temp)
            ###
            #convert matched elements to intergers
            bad_ind_lo=np.sort(np.array([np.int64(x) for x in bad_ind_lo]))
            bad_ind_hi=np.sort(np.array([np.int64(x) for x in bad_ind_hi]))
            good_ind_lo=np.sort(np.array([np.int64(x) for x in good_ind_lo]))
            good_ind_hi=np.sort(np.array([np.int64(x) for x in good_ind_hi]))    
            ###
            ###
            ###
            ### append both high/low results to final arrays
            bad_ind = np.sort(np.append(bad_ind_lo,bad_ind_hi).flatten())
            ###
            ### MAAAAAYBE we don't want to keep positive outliers after all...
            # good_ind = np.sort(np.append(good_ind_lo,good_ind_hi).flatten())
            ### MAAAAAYBE we don't want to keep positive outliers after all...
            good_ind = np.sort((good_ind_lo).flatten())
            temp_bad_ind = np.append(bad_ind_lo,bad_ind_hi)
            bad_ind = np.sort(np.append(temp_bad_ind,good_ind_hi).flatten())
            ###
            ###
            bad_ind=np.sort(np.array([np.int64(x) for x in bad_ind]))    
            good_ind=np.sort(np.array([np.int64(x) for x in good_ind]))    
            ###
            #convert matched elements to intergers
            ###
        ###
        #finally, remove bad points (keep good points) and make new time,flux, error arrays
        ###
        badtime = np.array([i for j, i in enumerate(time) if j in bad_ind])
        badsap_flux = np.array([i for j, i in enumerate(sap_flux) if j in bad_ind])
        baderror = np.array([i for j, i in enumerate(sap_error) if j in bad_ind])
        badflux_detrended = np.array([i for j, i in enumerate(det_flux) if j in bad_ind])
        baddet_error = np.array([i for j, i in enumerate(det_error) if j in bad_ind])
        badtrend_lc = np.array([i for j, i in enumerate(trend_lc) if j in bad_ind])
        badpld_detrended = np.array([i for j, i in enumerate(pld_flux) if j in bad_ind])
        badpld_model = np.array([i for j, i in enumerate(pld_model) if j in bad_ind])
        badpld_error = np.array([i for j, i in enumerate(pld_error) if j in bad_ind])
        badcentx= np.array([i for j, i in enumerate(centx) if j in bad_ind])
        badcenty= np.array([i for j, i in enumerate(centy) if j in bad_ind])        
        
        goodtime = np.array([i for j, i in enumerate(time) if j in good_ind])
        goodsap_flux = np.array([i for j, i in enumerate(sap_flux) if j in good_ind])
        gooderror = np.array([i for j, i in enumerate(sap_error) if j in good_ind])
        goodflux_detrended = np.array([i for j, i in enumerate(det_flux) if j in good_ind])
        gooddet_error = np.array([i for j, i in enumerate(det_error) if j in good_ind])
        goodtrend_lc = np.array([i for j, i in enumerate(trend_lc) if j in good_ind])
        goodpld_detrended = np.array([i for j, i in enumerate(pld_flux) if j in good_ind])
        goodpld_model = np.array([i for j, i in enumerate(pld_model) if j in good_ind])
        goodpld_error = np.array([i for j, i in enumerate(pld_error) if j in good_ind])
        goodcentx= np.array([i for j, i in enumerate(centx) if j in good_ind])
        goodcenty= np.array([i for j, i in enumerate(centy) if j in good_ind])
        
        
        
        time = np.array([i for j, i in enumerate(time) if j not in bad_ind])
        sap_flux = np.array([i for j, i in enumerate(sap_flux) if j not in bad_ind])
        sap_error = np.array([i for j, i in enumerate(sap_error) if j not in bad_ind])
        det_flux = np.array([i for j, i in enumerate(det_flux) if j not in bad_ind])
        det_error = np.array([i for j, i in enumerate(det_error) if j not in bad_ind])
        trend_lc = np.array([i for j, i in enumerate(trend_lc) if j not in bad_ind])
        pld_flux = np.array([i for j, i in enumerate(pld_flux) if j not in bad_ind])
        pld_error = np.array([i for j, i in enumerate(pld_error) if j not in bad_ind])
        pld_model = np.array([i for j, i in enumerate(pld_model) if j not in bad_ind])        
        centx = np.array([i for j, i in enumerate(centx) if j not in bad_ind])        
        centy = np.array([i for j, i in enumerate(centy) if j not in bad_ind])         
        ###
        ###
        ###
        import pandas as pd
        #saving flagged data points
        print('saving flagged outliers')
        bad_ind_DF = pd.DataFrame({"Time":badtime, "SAP Flux": badsap_flux, "SAP Error":baderror, "Detrended Flux":badflux_detrended, "Detrended Error":baddet_error,"Fitted Trend":badtrend_lc,"PLD Flux":badpld_detrended,"PLD Model":badpld_model,"PLD Error":badpld_error,"Centroid X Positions":badcentx,"Centroid Y Positions":badcenty})
        good_ind_DF = pd.DataFrame({"Time":goodtime, "SAP Flux": goodsap_flux, "SAP Error":gooderror, "Detrended Flux":goodflux_detrended, "Detrended Error":gooddet_error,"Fitted Trend":goodtrend_lc,"PLD Flux":goodpld_detrended,"PLD Model":goodpld_model,"PLD Error":goodpld_error,"Centroid X Positions":goodcentx,"Centroid Y Positions":goodcenty})
        bad_ind_DF.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bad_outliers.txt",index=False)
        good_ind_DF.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_good_outliers.txt",index=False)
        ###
        ###
        ###
        if verbose==True:
            print('after outlier removal: ',' T', len(time),' Det',len(det_flux), ' trend',len(trend_lc), 'PLD model',len(pld_model),' PLD F',len(pld_flux),' SAP E', len(sap_error))
    ###
    if remove_outliers=='no':
        time = np.ascontiguousarray(time, dtype=np.float64)
        sap_flux = np.ascontiguousarray(sap_flux, dtype=np.float64)
        sap_error = np.ascontiguousarray(sap_error, dtype=np.float64)
        det_flux = np.ascontiguousarray(det_flux, dtype=np.float64)
        det_error= np.ascontiguousarray(det_error, dtype=np.float64)
        trend_lc = np.ascontiguousarray(trend_lc, dtype=np.float64)                        
        pld_flux = np.ascontiguousarray(pld_flux, dtype=np.float64)        
        pld_error = np.ascontiguousarray(pld_error, dtype=np.float64)
        pld_model = np.ascontiguousarray(pld_model, dtype=np.float64)
        centx = np.ascontiguousarray(centx, dtype=np.float64)
        centy = np.ascontiguousarray(centy, dtype=np.float64)        
        if verbose==True:
            print('skipping outlier removal:',' T', len(time),' Det',len(det_flux), ' trend',len(trend_lc), 'PLD model',len(pld_model),' PLD F',len(pld_flux),' SAP E', len(sap_error))        
    ###
    ###
    ###                  
    nanmask = np.where(np.isfinite(pld_flux)==True)[0]
    time = time[nanmask]
    sap_flux = sap_flux[nanmask]
    sap_error=sap_error[nanmask]
    det_flux = det_flux[nanmask]
    det_error=det_error[nanmask]
    trend_lc=trend_lc[nanmask] 
    pld_flux = pld_flux[nanmask]    
    pld_error=pld_error[nanmask]
    pld_model=pld_model[nanmask]
    centx=centx[nanmask]
    centy=centy[nanmask]    
    
    
    #needs to be flexible to allow different combos of PLD, DET and SAP    
    LC_df = pd.DataFrame({"Time":time, "SAP Flux": sap_flux, "SAP Error":sap_error, "Detrended Flux":det_flux, "Detrended Error":det_error,"Fitted Trend":trend_lc,"PLD Flux":pld_flux,"PLD Error":pld_error,"PLD Model":pld_model,"Centroid X Positions":centx,"Centroid Y Positions":centy})
    ###
    LC_df.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_final_LC.txt",index=False)
    ###
    ###
    return LC_df, good_ind_DF, bad_ind_DF, preclipLC_df





def plot_it_all_up(ID,Sector,cutoutsize,cadence,Nsigma_low,Nsigma_high,hdu,median_image,pix_mask,bkg_mask, RAWLC_df, clippedRAWLC_df,LC_df, good_ind_DF, bad_ind_DF, preclipLC_df, magnitude_limit,dot_scale,path,downloadpath):
    import numpy as np
    #stuff for getting data from MAST
    import astropy
    from astroquery.mast import Catalogs
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    from astroquery.gaia import Gaia
    from astropy.wcs import WCS
    from astropy.io import fits
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Angle
    from astroquery.vizier import Vizier
    from matplotlib import pyplot as plt
    from transitleastsquares import catalog_info
    
    # changing cache directories
    Tesscut.cache_location=downloadpath
    Catalogs.cache_location=downloadpath
    Vizier.cache_location=downloadpath
    
    
    ### Plotting and Saving FFI and selected apertures
    ###
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist = Get_stellar_params(ID,downloadpath)
    ###
    wcs = WCS(hdu[2].header)
    
    if cadence=='short':
        x=hdu[1].header['1CRPX4']-1
        y=hdu[1].header['2CRPX4']-1
    if cadence=='long':
        x=hdu[1].header['1CRPX4']-1
        y=hdu[1].header['2CRPX4']-1
    CCD=hdu[0].header['CCD']
    Camera=hdu[0].header['Camera']
    reference_pixel=[x,y]
    centx,centy = centroid_quadratic(median_image, pix_mask,reference_pixel)
    ###
    rawtime=np.array(RAWLC_df['Time'].to_list())
    rawflux=np.array(RAWLC_df['SAP Flux'].to_list())
    rawflux_error=np.array(RAWLC_df['SAP Error'].to_list())
    ###    
    clippedrawtime=np.array(clippedRAWLC_df['Time'].to_list())
    clippedrawflux=np.array(clippedRAWLC_df['SAP Flux'].to_list())
    clippedrawflux_error=np.array(clippedRAWLC_df['SAP Error'].to_list())
    ###    
    ###
    # pre-outlier removed data (slightly better representation of steps)
    time=np.array(preclipLC_df['Time'].to_list())
    sap_flux=np.array(preclipLC_df['SAP Flux'].to_list())
    sap_error=np.array(preclipLC_df['SAP Error'].to_list())
    det_error=np.array(preclipLC_df['Detrended Error'].to_list())
    pld_error=np.array(preclipLC_df['PLD Error'].to_list())
    flux_detrended=np.array(preclipLC_df['Detrended Flux'].to_list())
    trend_lc=np.array(preclipLC_df['Fitted Trend'].to_list())
    pld_detrended=np.array(preclipLC_df['PLD Flux'].to_list())
    pld_model=np.array(preclipLC_df['PLD Model'].to_list())
    ###
    badtime=np.array(bad_ind_DF['Time'].to_list())
    badsapflux=np.array(bad_ind_DF['SAP Flux'].to_list())    
    baddetflux=np.array(bad_ind_DF['Detrended Flux'].to_list())    
    badpldflux=np.array(bad_ind_DF['PLD Flux'].to_list())    
    ###
    goodtime=np.array(good_ind_DF['Time'].to_list())
    goodsapflux=np.array(good_ind_DF['SAP Flux'].to_list())
    gooddetflux=np.array(good_ind_DF['Detrended Flux'].to_list())
    goodpldflux=np.array(good_ind_DF['PLD Flux'].to_list())
    ###
    #final data
    finaltime=np.array(LC_df['Time'].to_list())
    finalflux=np.array(LC_df['Detrended Flux'].to_list())
    finalerror=np.array(LC_df['Detrended Error'].to_list())
    #
    ###    
    ###
    print('raw SAP data T ', len(rawtime), ' F ',len(rawflux),' E ', len(rawflux_error))
    ###
    #fontsize
    fs=16
    ###
    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(521,projection = wcs)
    ax2 = fig.add_subplot(522,projection = wcs)
    ax3 = fig.add_subplot(512)
    ax4 = fig.add_subplot(513)
    ax5 = fig.add_subplot(514)
    ax6 = fig.add_subplot(515)
    ###
    ###
    axes = [ax1,ax2]
    ###
    ###
    ###
    if cadence=='long':
        savefigpath = path+'FFI_PLD_plots/'
        savelcpath = path+'FFI_PLD_LCs/'
    if cadence=='short':
        savefigpath = path+'TPF_PLD_plots/'
        savelcpath = path+'TPF_PLD_LCs/' 
    ###
    axes=[ax1,ax2]
    plot_cutouts(ID,Sector,cadence,hdu,pix_mask,bkg_mask,\
                 reference_pixel,fig,axes,savelcpath,downloadpath)
#     handles, labels = ax2.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     if len(by_label.values())>4:
#         ncols=2
#     else:
#         ncols=1
#     ax2.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.4,1.1),\
#                loc='upper right',fontsize=8,ncol=ncols,markerscale=1,\
#                labelspacing=1.,title="GAIA Mag")
    fig.tight_layout(pad=1)
    
    ###
    ###
    ###
    fig.suptitle(r"TIC "+str(ID)+" Sector "+str(Sector)+" Camera "+str(Camera)+" CCD "+str(CCD)+\
                 " $R_{star}$: "+str(np.round(R_star,3))+" $R_{\odot}$  $M_{star}$: "+str(np.round(M_star,3))+\
                 " $M_{\odot}$"+"\n Teff "+str(Teff)+" ; TESSmag "+str(Tmag)+'; Vmag '+str(Vmag),\
                 fontsize = fs,x=0.5,y=1.08)
    ###
    ###
    ###
    ###
    ax3.set_title("Simple Aperture Photometry (SAP) Light Curve and PLD Noise Model",fontsize = fs)
    ###
    ###    
    cdpp_sap = CDPP(clippedrawtime,clippedrawflux,clippedrawflux_error,'median','ppm',binsize=(1.0/24.0))
    cdpp_det = CDPP(time,flux_detrended,det_error,'median','ppm',binsize=(1.0/24.0)) 
    cdpp_pld = CDPP(time,pld_detrended,pld_error,'median','ppm',binsize=(1.0/24.0)) 
    cdpp_final = CDPP(finaltime,finalflux,finalerror,'median','ppm',binsize=(1.0/24.0)) 
    ###
    std_sap = 5*np.nanstd(clippedrawflux)
    ###
    ###
    ###
    ax3.set_ylim(np.nanmedian(clippedrawflux)-std_sap,np.nanmedian(clippedrawflux)+std_sap)
    ax3.plot(rawtime,rawflux,marker='.',color='grey',linestyle='none',\
             label='jitter/glare/Mdumps',zorder=-100)
    ax3.plot(clippedrawtime,clippedrawflux,marker='.',color='black',\
             linestyle='none',\
             label=r' SAP Flux'+'\n CDPP: '+str(np.round(cdpp_sap,2))+' $\sigma _{ppm}$ ''hr$^{-1/2}$',zorder=-100)
#     ax3.plot(time,trend_lc,color='orange',label='Fitted trend')
    ax3.plot(time,pld_model,'y-',label='PLD model')
    ###
    ###
#     ax4.set_title("Detrended Light Curve with Pixel Level Decorrelation (PLD) Model",fontsize = fs) 
    ax4.set_title("PLD Corrected Light Curve with Fitted Trend Line",fontsize = fs)    
#     ax4.plot(time,flux_detrended,"k.",label=r' Detrended Flux'+'\n CDPP: '+str(np.round(cdpp_det,2))+' ppm',zorder=-100)
    ax4.plot(time,pld_detrended, "k.",\
             label=r' PLD Flux'+'\n CDPP: '+str(np.round(cdpp_pld,2))+' $\sigma _{ppm}$ ''hr$^{-1/2}$',zorder=-100) 
#     ax4.plot(time,pld_model,'y-',label='PLD model')
    ax4.plot(time,trend_lc,color='orange',label='Fitted trend')
    ###
    ###
#     ax5.set_title("PLD Detrended Light Curve",fontsize = fs)
#     ax5.plot(time,pld_detrended, "k.",label=r' PLD Flux'+'\n CDPP: '+str(np.round(cdpp_pld,2))+' ppm',zorder=-100)
    ax5.set_title("PLD Corrected and Smoothed Light Curve",fontsize = fs)      
    ax5.plot(time,flux_detrended,"k.",\
             label=r' Detrended Flux'+'\n CDPP: '+str(np.round(cdpp_det,2))+' $\sigma _{ppm}$ ''hr$^{-1/2}$',zorder=-100)             
    ###
    ###
    ax6.set_title("PLD Corrected, Smoothed and Outlier Removed Light Curve",fontsize = fs)
    ax6.plot(finaltime,finalflux, "k.",label=r' Final Flux'+'\n CDPP: '+str(np.round(cdpp_final,2))+' $\sigma _{ppm}$ ''hr$^{-1/2}$',zorder=-100)
    ###
#     ax3.plot(badtime,badsapflux,'ro',zorder=2,label='bad outliers')
    ax5.plot(badtime,baddetflux,'r.',zorder=2,label='bad outliers')
    ax5.plot(goodtime,gooddetflux,'g.',zorder=2,label='good outliers')
    ###
    ###
    ###
    ###
    ax3.set_xlabel("Time (BTJD)",fontsize=fs)
#     ax3.set_ylabel("Normalized Relative flux ",fontsize=fs)
    ax3.legend(loc='best',ncol=2,fontsize=fs-2,fancybox=True,framealpha=1,markerscale=2)
    ax3.tick_params(axis='both', which='major', labelsize=fs)
    ax3.tick_params(axis='both', which='minor', labelsize=fs)
    ###
    ###
    ax4.set_xlabel("Time (BTJD)",fontsize=fs)
    ax4.set_ylabel("Normalized Relative flux ",fontsize=fs)
    ax4.legend(loc='best',ncol=2,fontsize=fs-2,fancybox=True,framealpha=1,markerscale=2)
    ax4.tick_params(axis='both', which='major', labelsize=fs)
    ax4.tick_params(axis='both', which='minor', labelsize=fs)
    ###
    ###
    ax5.set_xlabel("Time (BTJD)",fontsize=fs)
#     ax5.set_ylabel("Normalized Relative flux ",fontsize=fs)
    ax5.legend(loc='best',ncol=2,fontsize=fs-2,fancybox=True,framealpha=1,markerscale=2)
    ax5.tick_params(axis='both', which='major', labelsize=fs)
    ax5.tick_params(axis='both', which='minor', labelsize=fs)         
    ###
    ###
    ax6.set_xlabel("Time (BTJD)",fontsize=fs)
#     ax5.set_ylabel("Normalized Relative flux ",fontsize=fs)
    ax6.legend(loc='best',ncol=2,fontsize=fs-2,fancybox=True,framealpha=1,markerscale=2)
    ax6.tick_params(axis='both', which='major', labelsize=fs)
    ax6.tick_params(axis='both', which='minor', labelsize=fs)
    ###
    ###
    
    #ax3.legend(loc='lower right', bbox_to_anchor=(1.175,0.0))
    #ax4.legend(loc='lower right', bbox_to_anchor=(1.175,0.0))
    #ax5.legend(loc='lower right', bbox_to_anchor=(1.175,0.0))    
    #ax6.legend(loc='lower right', bbox_to_anchor=(1.175,0.0))
    
    #
    #ax3.legend(loc='upper right', bbox_to_anchor=(1.175,1.0), bbox_transform=ax3.transAxes)
    #ax4.legend(loc='upper right', bbox_to_anchor=(1.175,1.0), bbox_transform=ax4.transAxes)
    #ax5.legend(loc='upper right', bbox_to_anchor=(1.175,1.0), bbox_transform=ax5.transAxes)
    #ax6.legend(loc='upper right', bbox_to_anchor=(1.175,1.0), bbox_transform=ax6.transAxes)
    
    ax3.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax4.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax5.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ax6.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    ###
    ###    
    mdumps,t_0,t_1 = momentumdump_check(Sector)
    t_0=np.nanmin(time)
    Num_mdumps = int(np.round((np.nanmax(time) - np.nanmin(time))/mdumps,2))+1
    print('')  
    ###
    ###    
    
    if Sector==31:
        t_0end =  2157.45371
        t_1end = 2169.94398
        time_mdump1 = t_0+ (t_0end - t_0)/2
        time_mdump2 = t_1+ (t_1end - t_1)/2
        #
        ax3.axvline(x=time_mdump1,zorder=-2)
        ax4.axvline(x=time_mdump1,zorder=-2)
        ax5.axvline(x=time_mdump1,zorder=-2)
        ax6.axvline(x=time_mdump1,zorder=-2)
        #
        ax3.axvline(x=time_mdump2,zorder=-2)
        ax4.axvline(x=time_mdump2,zorder=-2)
        ax5.axvline(x=time_mdump2,zorder=-2)
        ax6.axvline(x=time_mdump2,zorder=-2)        
    else:
        for N in range(Num_mdumps):
            time_mdump1 = t_0+(N)*mdumps
            time_mdump2 = t_1+(N+0.5)*mdumps  
            if time_mdump1 < t_1:
                ax3.axvline(x=time_mdump1,zorder=-2)
                ax4.axvline(x=time_mdump1,zorder=-2)
                ax5.axvline(x=time_mdump1,zorder=-2)
                ax6.axvline(x=time_mdump1,zorder=-2)
            if time_mdump2 < np.nanmax(time):
                ax3.axvline(x=time_mdump2,zorder=-2)
                ax4.axvline(x=time_mdump2,zorder=-2)
                ax5.axvline(x=time_mdump2,zorder=-2)            
                ax6.axvline(x=time_mdump2,zorder=-2)            
    ###
    ###
    fig.tight_layout(pad=0.1,h_pad=0,w_pad=10)
    fig.savefig(savefigpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_LC_summary.png",bbox_inches='tight')
    #plt.show()
    plt.close()




###### putting it all together ##########
###### putting it all together ##########
###### putting it all together ##########


#work on making order of operations flexible
def full_pipeline(ID,cutoutsize,Sector,minimum_photon_counts,threshold,pld_order,n_pca_terms, Nsigma_low, Nsigma_high, remove_outliers, before_after_in_minutes, path, cadence, verbose, keep_FITS=True, keep_imagedata=True, window_size_in_days=None,use_SPOC_aperture='no'):    
    from transitleastsquares import catalog_info    
    import sys
    ###
    #first, check if target has known stellar radius and/or mass:
    from transitleastsquares import catalog_info    
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    if np.isfinite(R_star)==False or np.isfinite(M_star)==False:
        print('TIC '+str(ID)+' has no known Stellar Mass or Radius in TIC')
        return
    else:
        ###
        ###
        ###
        print('TIC '+str(ID)+' Sector '+str(Sector))
        #Step 0: Creating directories to save figures and data
        if verbose==True:
            print('Step 0: Making Directories')
            print(' ')
        path, savefigpath, savelcpath,downloadpath = Make_dirs(path,Sector,cadence)
        ###
        ###
        ###
        #Step 1: Obtaining HDU for FFI/TPF
        if verbose==True:
            print('Step 1: Obtaining HDU from FFI/TPF')
            print(' ')
        try:
            hdu,CCD,Camera,quality_mask,reference_pixel = gethdu(ID,Sector,cutoutsize,cadence,minimum_photon_counts,verbose,downloadpath)
            ###
            ###        
            print(' ')
            if hdu==None:
                #print('No Image data for TIC '+str(ID)+' in Sector '+ str(Sector)+'!!!')
                sys.exit('No Image data for TIC '+str(ID)+' in Sector '+ str(Sector)+'!!!') 
        except AttributeError as AE:
            print(AE)
            sys.exit('No Image data for TIC '+str(ID)+' in Sector '+ str(Sector)+'!!!') 
        ###
        ###
        ###
        ###
        if verbose==True:
            print('Step 2: Performing Background Subtraction and Simple Aperture Photometry')
            print(' ')
        try:
            bkg_mask, pix_mask ,flux, median_image, SAP_LC, flux_contamination_ratio = SAP(ID,Sector,cutoutsize,hdu,quality_mask,threshold,cadence,reference_pixel,verbose,savelcpath,use_SPOC_aperture='no')
            ###
            ###
            ###
        except TypeError as TE:
            print(TE)
            print('Unable to create aperture mask. Skipping this target...')
            return 
        ###
        ###
        ###
        if len(SAP_LC['SAP Error'])==0:
            print(' ')
            print('Uneven array lengths, FFI likely on edge of detector/partially shown')
            return
        ###
        ###
        if verbose==True:
            print('Step 3: Removing Momentum dumps and regions of high jitter / Earth-Moon glare')
            print(' ')
        mask_mdump, mdumps,t_0,t_1, flux, RAWLC_df, clippedRAWLC_df = Applying_Mdump_removal(ID,Sector,Camera,CCD,before_after_in_minutes,SAP_LC,flux,savelcpath,verbose)
        ###
        ###
        ### saving pixel and background masks and image fluxes
        saveNDarr(pix_mask,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_pix_mask")
        saveNDarr(bkg_mask,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bkg_mask")
        saveNDarr(flux,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_image_fluxes")    
        ###
        ### calculating centroid positions throughout images and resaving to file
        cxs,cys = check_centroids(ID,Sector,cutoutsize,cadence,reference_pixel,savelcpath)
        time = np.array(clippedRAWLC_df['Time'])
        sap_flux=np.array(clippedRAWLC_df['SAP Flux'])
        sap_error=np.array(clippedRAWLC_df['SAP Error'])
        bkg_flux=np.array(clippedRAWLC_df['Background Flux'])
        clippedRAWLC_df = pd.DataFrame({"Time":time, "SAP Flux": sap_flux, "SAP Error":sap_error,"Background Flux":bkg_flux, "Centroid X Positions":cxs,"Centroid Y Positions":cys})
        clippedRAWLC_df.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_RAW_LC_systematics_removed.txt",index=False)
        ###
        if verbose==True:
            print('Step 4: Performing Pixel Level Decorrelation modeling')
            print(' ')
        ###
        ## work on making this flexible to take either PLD or SAP
        input_LC = pd.DataFrame({'Time':np.array(clippedRAWLC_df['Time']),\
                         'Flux':np.array(clippedRAWLC_df['SAP Flux']),\
                        'Error':np.array(clippedRAWLC_df['SAP Error']),\
                                 "Centroid X Positions":np.array(clippedRAWLC_df["Centroid X Positions"]),\
                                "Centroid Y Positions":np.array(clippedRAWLC_df["Centroid Y Positions"])})

        PLD_LC = PLD_model(ID,Sector,flux,pix_mask,input_LC,savelcpath,pld_order=pld_order, n_pca_terms=n_pca_terms)
        ###
        ###
        ###
        if verbose==True:
            print('Step 5: Applying smoothing filter')
            print(' ')    
        print('len check for step 5:')
        print('PLD T',len(np.array(PLD_LC['Time'])),'PLD F',len(np.array(PLD_LC['PLD Flux'])),'PLD E',len(np.array(PLD_LC['PLD Error'])))
        ## work on making this flexible to take either DET or SAP
        input_LC2 = pd.DataFrame({'Time':np.array(PLD_LC['Time']),\
                                 'Flux':np.array(PLD_LC['PLD Flux']),\
                                'Error':np.array(PLD_LC['PLD Error']),\
                                  'Model':np.array(PLD_LC['PLD Model']),\
                                  'SAP Flux':np.array(input_LC['Flux']),\
                                  'SAP Error':np.array(input_LC['Error']),\
                                 "Centroid X Positions":np.array(PLD_LC["Centroid X Positions"]),\
                                "Centroid Y Positions":np.array(PLD_LC["Centroid Y Positions"])})
        Det_LC, nanmask = BWMC_auto(ID,Sector,input_LC2,savelcpath)           
        ###
        ### ensure PLD outputs and Detrended outputs have same length using nanmask output
        print('2nd len check for step 5: ')
    #     print('T: ',len(Det_LC['Time']),'SAP F: ',len(PLD_LC['SAP Flux']), 'SAP E: ',len(PLD_LC['SAP Error'])   ,\
    #           ' PLD F: ', len(PLD_LC['PLD Flux']),' PLD E: ', len(PLD_LC['PLD Error']),\
    #                     ' Det F: ',len(Det_LC['Detrended Flux']), 'Det E: ',len(Det_LC['Detrended Error']))    
    #     PLD_LC = pd.DataFrame({'Time':np.array(Det_LC['Time']),'SAP Flux':(PLD_LC['SAP Flux'])[nanmask],\
    #                            'SAP Error':(PLD_LC['SAP Error'])[nanmask],'PLD Flux':np.array(PLD_LC['PLD Flux'])[nanmask],\
    #                            'PLD Error':np.array(PLD_LC['PLD Error'])[nanmask],\
    #                           'PLD Model':np.array(PLD_LC['PLD Model'])[nanmask],\
    #                          'Centroid X Positions':np.array(PLD_LC['Centroid X Positions']),\
    #                           'Centroid Y Positions':np.array(PLD_LC['Centroid Y Positions'])})
        print('T: ',len(Det_LC['Time']),'SAP F: ',len(Det_LC['SAP Flux']), 'SAP E: ',len(Det_LC['SAP Error']),' PLD F: ', len(Det_LC['PLD Flux']),' PLD E: ', len(Det_LC['PLD Error']),' PLD M: ', len(Det_LC['PLD Model']),' Det F: ',len(Det_LC['Detrended Flux']), 'Det E: ',len(Det_LC['Detrended Error']),' Det M: ',len(Det_LC['Fitted Trend']))   
        ###
        ###
        ###
        if verbose==True:
            print('Step 6: Applying Outlier Removal (if set to "yes")')
            print(' ')
            ###
            #this needs to be the MOST flexible part to deal with combos of PLD, DET and SAP
        print('len check for step 6:')
        print('T: ',len(Det_LC['Time']),'SAP F: ',len(Det_LC['SAP Flux']), 'SAP E: ',len(Det_LC['SAP Error']),' PLD F: ', len(Det_LC['PLD Flux']),' PLD E: ', len(Det_LC['PLD Error']),' PLD M: ', len(Det_LC['PLD Model']),' Det F: ',len(Det_LC['Detrended Flux']), 'Det E: ',len(Det_LC['Detrended Error']),' Det M: ',len(Det_LC['Fitted Trend']))   
        input_LC3 = pd.DataFrame({'Time':np.array(Det_LC['Time']),\
                                  'SAP Flux':np.array(Det_LC['SAP Flux']),\
                                  'SAP Error':np.array(Det_LC['SAP Error']),\
                                 'Detrended Flux':np.array(Det_LC['Detrended Flux']),\
                                'Detrended Error':np.array(Det_LC['Detrended Error']),\
                                  'Fitted Trend':np.array(Det_LC['Fitted Trend']),\
                                 'PLD Flux':np.array(Det_LC['PLD Flux']),\
                                 'PLD Error':np.array(Det_LC['PLD Error']),\
                                 'PLD Model':np.array(Det_LC['PLD Model']),\
                                 "Centroid X Positions":np.array(Det_LC["Centroid X Positions"]),\
                                  "Centroid Y Positions":np.array(Det_LC["Centroid Y Positions"])})
        ###
        ###
        LC_df, good_ind_DF, bad_ind_DF, preclipLC_df = outlier_removal(ID,Sector,input_LC3, remove_outliers, Nsigma_low,Nsigma_high,savelcpath,verbose,window_size_in_days=window_size_in_days)       
        ###
        ###
        ###
        ###
        ###
        ###
        if verbose==True:
            print('Step 7: Plotting and Saving FFI and selected apertures')
            print(' ')
        plot_it_all_up(ID,Sector,cutoutsize,cadence,Nsigma_low,Nsigma_high,\
                       hdu,median_image,pix_mask,bkg_mask, RAWLC_df, \
                       clippedRAWLC_df, LC_df, good_ind_DF, bad_ind_DF, preclipLC_df, \
                       magnitude_limit=18,dot_scale=20,path=path,downloadpath=downloadpath)
        ###
        ###
        if keep_FITS==False:
            # deleting FITS files (no longer need them for light curve processing
            # can always download again)
            os.system("rm -r " + downloadpath) #delete cache path
        ###
        ###
        print('FINAL LENGTHS :', ' T', len(LC_df['Time']),' Det F',len(LC_df['Detrended Flux']), ' trend',len(LC_df['Fitted Trend']), 'PLD model',len(LC_df['PLD Model']),' PLD F',len(LC_df['PLD Flux']),' SAP E', len(LC_df['SAP Error']))

###### putting it all together ##########
###### putting it all together ##########
###### putting it all together ##########








##########################################
##########################################
##########################################
####### TRANSIT SEARCH FUNCTIONS #########
##########################################
##########################################
##########################################


def phasefold(T0,time,period,flux):
    phase=(time- T0 + 0.5*period) % period - 0.5*period        
    ind=np.argsort(phase, axis=0)
    return phase[ind],flux[ind]


def MoreThan_N_Transits(ID,time,flux,error,T_C_array,T_C,planet_model,planet_model_time,\
                        Period,xwidth,spacing,window_size,T_C_x_position,T_C_y_position,\
                        fontsize,markersize,axis,XLIM,TLS_Dur):
    ###
    #new
    spacing = 1.0-np.nanmin(planet_model) #sort of like transit depth
    spacing=0.5*spacing
#     print('spacing: ',spacing)
    if spacing<0.01:
        spacing=0.005
    xshift=1.1
    yshift=0.1
    XLIM=0-0.5*TLS_Dur#1.75*TLS_Dur
    ###
    ###
    for x in range(len(T_C_array)):
        Even=[]
        Evenmodel=[]
        Odd=[]
        Oddmodel=[]
        if x %2 ==0: #even
            ###
            cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
            cut_t  =  time[cut]
            cut_f  =  flux[cut]
            cut_fe = error[cut]
            cut2 = np.where( ((T_C_array[x]-window_size) < planet_model_time) & ((T_C_array[x]+window_size) > planet_model_time)  )[0]
            cut_model_f = planet_model[cut2]
            cut_model_t = planet_model_time[cut2]
            ###
            if len(cut_f)<1:
#                 print('cut window too small, switching to 1.5 hr window')
                window_size=1.5
                cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
                cut_t  =  time[cut]
                cut_f  =  flux[cut]
                cut_fe = error[cut]
                cut2 = np.where( ((T_C_array[x]-window_size) < planet_model_time) & ((T_C_array[x]+window_size) > planet_model_time)  )[0]
                cut_model_f = planet_model[cut2]
                cut_model_t = planet_model_time[cut2]
            ###
            phase,_ = phasefold(T_C_array[x],cut_t,Period,cut_f)
            Even=np.append(Even,phase)
            pf_model,_ = phasefold(T_C_array[x],cut_model_t,Period,cut_model_f)
            Evenmodel=np.append(Evenmodel,pf_model)
            xxxx=1
            axis.plot(24*Even,cut_f,color='dimgrey',marker='o',linestyle='none',markersize=markersize+1, rasterized=True)
            axis.plot(24*Evenmodel,cut_model_f,'r.-',markersize=markersize-1, rasterized=True)
        ###
        else: #odd
            cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
            cut_t  =  time[cut]
            cut_f  =  flux[cut]
            cut_fe = error[cut]
            cut2 = np.where( ((T_C_array[x]-window_size) < planet_model_time) & ((T_C_array[x]+window_size) > planet_model_time)  )[0]
            cut_model_f = planet_model[cut2]
            cut_model_t = planet_model_time[cut2]
            ###
            if len(cut_f)<1:
#                 print('cut window too small, switching to 1.5 hr window')
                window_size=1.5
                cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
                cut_t  =  time[cut]
                cut_f  =  flux[cut]
                cut_fe = error[cut]
                cut2 = np.where( ((T_C_array[x]-window_size) < planet_model_time) & ((T_C_array[x]+window_size) > planet_model_time)  )[0]
                cut_model_f = planet_model[cut2]
                cut_model_t = planet_model_time[cut2]
            ###
            phase,_ = phasefold(T_C_array[x],cut_t,Period,cut_f)
            Odd=np.append(Odd,phase)
            pf_model,_ = phasefold(T_C_array[x],cut_model_t,Period,cut_model_f)
            Oddmodel=np.append(Oddmodel,pf_model)
            xxxx=xxxx+2
            axis.plot(24*Odd,cut_f+spacing*(xxxx),color='lightblue',marker='o',linestyle='none',markersize=markersize+1, rasterized=True)
            axis.plot(24*Oddmodel,cut_model_f+spacing*(xxxx),'r.-',markersize=markersize-1, rasterized=True)
    ###
    axis.annotate("Odd", xy=( XLIM, np.nanmean(cut_model_f+spacing*((xxxx+0.5)))+T_C_y_position ), va='top',xycoords='data', fontsize=fontsize+4,weight="bold")
    axis.annotate("Even", xy=( XLIM, np.nanmean(cut_model_f)-spacing*(1)+T_C_y_position ), va='top',xycoords='data', fontsize=fontsize+4,weight="bold")
    ###
    ymax = np.nanmax(cut_f+spacing*4)#(xxxx+1)*1.5) 
    ymin = np.nanmin(cut_f-spacing*2) 
#     print(ymin,ymax)
    return ymin,ymax



def LessThan_N_Transits(ID,time,flux,error, T_C_array,T_C,planet_model,planet_model_time,\
                        Period,xwidth,spacing,window_size,T_C_x_position,T_C_y_position,\
                        fontsize,markersize,axis,XLIM,TLS_Dur):
    XLIM=0-TLS_Dur#float(10.0*TLS_Dur)#-0.5*TLS_Dur
    #print(XLIM)
    #alter text position and size:
    T_C_x_position=T_C_x_position+1.25
    #print(T_C_x_position)
    fontsize=fontsize-4
    shift=0.75
    ###
    spacing = 1.0-np.nanmin(planet_model) #sort of like transit depth
#     spacing=2.5*spacing
    spacing=0.5*spacing
    if spacing<0.01:
        spacing=0.005
    ###
    for x in range(len(T_C_array)):
        cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
        cut_t  =  time[cut]
        cut_f  =  flux[cut]
        cut_fe = error[cut]
        cut2 = np.where( ((T_C_array[x]-window_size) < planet_model_time) & ((T_C_array[x]+window_size) > planet_model_time)  )[0]
        cut_model_f = planet_model[cut2]
        cut_model_t = planet_model_time[cut2]
        if len(cut_f)<1:
            print('cut window too small, switching to 1.5 hr window')
            window_size=1.5
            cut = np.where(((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time))[0]
            cut_t  =  time[cut]
            cut_f  =  flux[cut]
            cut_fe = error[cut]
            cut2 = np.where( ((T_C_array[x]-window_size) < planet_model_time) & ((T_C_array[x]+window_size) > planet_model_time)  )[0]
            cut_model_f = planet_model[cut2]
            cut_model_t = planet_model_time[cut2]
        ###
        phase,cut_f = phasefold(T_C_array[x],cut_t,Period,cut_f)
        pf_model,cut_model_f = phasefold(T_C_array[x],cut_model_t,Period,cut_model_f)
        if x % 2 == 0: #alternate colors
            xxxx = x+1
            axis.plot(24*phase,cut_f+spacing*(xxxx),color='dimgrey',marker='o',linestyle='-',markersize=markersize+1, rasterized=True)
            axis.plot(24*pf_model,cut_model_f+spacing*(xxxx),'r.-',markersize=markersize-1, rasterized=True)            
            axis.annotate(str(np.round(T_C_array[x],3))+" BTJD", xy=( XLIM, np.nanmean(cut_model_f+spacing*(xxxx+shift)-T_C_y_position) ), va='top',xycoords='data', fontsize=fontsize,weight="bold")            
        else:
            xxxx = x+1
            axis.plot(24*phase,cut_f+spacing*(xxxx),color='lightblue',marker='o',linestyle='-',markersize=markersize+1,rasterized=True)
            axis.plot(24*pf_model,cut_model_f+spacing*(xxxx),'r.-',markersize=markersize-1, rasterized=True)       
            axis.annotate(str(np.round(T_C_array[x],3))+" BTJD", xy=( XLIM, np.nanmean(cut_model_f+spacing*(xxxx+shift)-T_C_y_position) ), va='top',xytext=(XLIM, np.nanmean(cut_model_f+spacing*(xxxx+shift)-T_C_y_position)),xycoords='data', fontsize=fontsize,weight="bold")
    ymax = np.nanmax(cut_model_f+spacing*(len(T_C_array)+1)) 
    ymin = np.nanmin(cut_model_f+spacing*(len(T_C_array)-5)) 
    return ymin,ymax


def phasematch_and_seperate_plot_TLS(ID,time,flux,error, T_C_array,T_C,planet_model,planet_model_time,
                                     Period,xwidth,spacing,window_size,
                                    T_C_x_position,T_C_y_position,fontsize,markersize,axis,TLS_Dur):
    ###
    from matplotlib.offsetbox import AnchoredText
    N=5 #number of transits
    if len(T_C_array) > N:
        ymin,ymax = MoreThan_N_Transits(ID,time,flux,error, T_C_array,T_C,planet_model,planet_model_time,
                                     Period,xwidth,spacing,window_size,
                                    T_C_x_position,T_C_y_position,fontsize,markersize,axis,0,TLS_Dur)
    if len(T_C_array) <= N:
        ymin,ymax = LessThan_N_Transits(ID,time,flux,error, T_C_array,T_C,planet_model,planet_model_time,
                                     Period,xwidth,spacing,window_size,
                                    T_C_x_position,T_C_y_position,fontsize,markersize,axis,0,TLS_Dur)
    return ymin,ymax

def Vet_with_EDIVetter(ID, Sector, TLS_OUTPUT, qld, SDE_threshold, N_transits):
    import EDIunplugged as EDI
    import pandas as pd
    params=EDI.parameters(TLS_OUTPUT,limbDark=[qld[0], qld[1]], impact=0, snrThreshold=SDE_threshold, minTransit=N_transits)
    params=EDI.Go(params,telescope='TESS')
    
    EDI_results = pd.DataFrame({'ID':ID,'Sector':Sector,'Flux Contamination':params.fluxContaminationFP, 'Too Many Transits Masked':params.TransMaskFP, 'Odd/Even Transit Variation':params.even_odd_transit_misfit,'Signal is not Unique':params.uniquenessFP,'Secondary Eclipse Found':params.SeFP,'Low Transit Phase Coverage':params.phaseCoverFP, 'Transit Duration Too Long':params.tdurFP, 'Signal is a False Positive':params.FalsePositive},index=[0])
    return EDI_results

def TLS_func(ID,Sector,cadence,time,flux,error,N_transits,minP,oversampling_factor,duration_grid_step,path,for_injections=False):
    #for reporting TLS model's planet radius
    R_earth = 6.378*10.0**8.0 #cm
    R_sun = 6.955*10.0**10.0 #cm
    ###
    ###
    # calculate CDPP (in ppm per sqrt hour) to save later
    cdpp = CDPP(time,flux,error,'median','ppm',binsize=(1.0/24.0))    
    time_span = np.nanmax(time) - np.nanmin(time)    
    #
    from transitleastsquares import catalog_info    
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    from transitleastsquares import transitleastsquares
    ###
    maxP= (max(time)-min(time))/N_transits #length of our light curve
    if cadence=='long':
        T0_fit_margin=0 #samples every data point
    else:
        T0_fit_margin = 0.001
    ###
    if np.isfinite(R_star)==True and np.isfinite(M_star)==True:
        ###
        tls = transitleastsquares(time,flux,error)
        if np.isnan(R_star_min)==True:
            print('R_star_min = NaN')
            print(' ')
            ###
            tls = transitleastsquares(time,flux,error)
            tls_power = tls.power(period_min=minP,period_max=maxP,n_transits_min=N_transits,\
                                  oversampling_factor=oversampling_factor,\
                                  duration_grid_step=duration_grid_step,\
                                  T0_fit_margin=T0_fit_margin,show_progress_bar=False)
        else:
            tls_power = tls.power(R_star_min=R_star-R_star_min, R_star_max=R_star+R_star_max,R_star=R_star,\
                                  M_star_min=M_star-M_star_min, M_star_max=M_star+M_star_max,M_star=M_star,\
                                  u=qld,period_min=minP,period_max=maxP,\
                                  n_transits_min=N_transits,\
                                  oversampling_factor=oversampling_factor,\
                                  duration_grid_step=duration_grid_step,\
                                  T0_fit_margin=T0_fit_margin,show_progress_bar=False)  
        ###        
        ###
        tls_power_periods = tls_power.periods
        ###
        #TLS results
        TLS_periods=tls_power.period
        TLS_periods_uncertainty=tls_power.period_uncertainty 
        TLS_odd_even = tls_power.odd_even_mismatch         
        TLS_FAP = tls_power.FAP 
        TLS_t0s=tls_power.T0
        TLS_depths=tls_power.depth
        TLS_Power=tls_power.power
        TLS_sde=tls_power.SDE #top peak
        TLS_Dur = tls_power.duration
        TLS_TCs = tls_power.transit_times
        TLS_SR  = tls_power.SR
        TLS_SDE = (TLS_Power-np.nanmedian(TLS_Power))/np.nanstd(TLS_Power)
        
        # check TLS_SDE, if nan mean = nan, this is a junk result, likely due to noisy lightcurve
        if len(np.where(np.isnan(TLS_SDE)==True)[0])==len(TLS_SDE):
            print('TLS unabe to converge to solution. Check light curve, likely high CDPP.')            
            return None, None, None , None
        ### Vet with EDI-Vetter
        EDI_results=Vet_with_EDIVetter(ID, Sector, tls_power, qld, SDE_threshold=6, N_transits=N_transits)
        
        TLS_depths_arr=tls_power.transit_depths #array of mean depths
        TLS_depths_arr_uncertainty=tls_power.transit_depths_uncertainties
        #making assumption that median of array of mean depth errors is representative of depth error
        TLS_depths_err = np.nanmedian(TLS_depths_arr)/len(TLS_depths_arr) #new,float
        
        TLS_rp_rs = tls_power.rp_rs #new, float
        TLS_snr = tls_power.snr #new, float
        TLS_transit_count= tls_power.transit_count #new, int
        TLS_distinct_transit_count = tls_power.distinct_transit_count #new, int        
        TLS_per_transit_count= tls_power.per_transit_count #new, array
        TLS_snr_per_transit  = tls_power.snr_per_transit #new, array
        TLS_snr_pink_per_transit = tls_power.snr_pink_per_transit #new. array
        ###
        #calculating TLS estimated planet radius and error
        R_p = np.sqrt(1.0-TLS_depths)*R_star*R_sun/R_earth
        R_star_err = R_star_min
        R_p_err = R_p * np.sqrt((R_star_err/R_star)**2 + (TLS_depths_err/TLS_depths)**2)
        ###
        #TLS models
        TLS_model_time = tls_power.model_lightcurve_time 
        TLS_model = tls_power.model_lightcurve_model 
        ###
        ###        
        ###        
        ###
        #saving results
        if for_injections==False:
            Path=path+'Sector_'+str(Sector)+'/'
        if for_injections==True:
            Path=path
        if cadence=='long':
            saveReportpath = Path+'FFI_TLS_Report/'
        if cadence=='short':            
            saveReportpath = Path+'TPF_TLS_Report/'
        ###
        if os.path.exists(saveReportpath)==True:
    #         print('folder exists, moving on...') #feel free to uncomment these out
            pass
        else:
    #         print('making directory') 
            os.makedirs(saveReportpath)
        ###
        import pandas as pd
        ###
        TLSdf = pd.DataFrame({"TLS Periods":tls_power_periods, "TLS Power": TLS_Power, "TLS SR":TLS_SR, "TLS SDE":TLS_SDE})
        TLSdf.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_TLS.txt",index=False)
        ###
        TLSmodeldf = pd.DataFrame({"Time":TLS_model_time, "Model": TLS_model})
        TLSmodeldf.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_TLS_model.txt",index=False)
        ###
        EDI_results.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_EDI_results.txt",index=[0])
        ###
        ###    
#         from NEMESIS_pipeline import Make_dirs, Get_stellar_params, gethdu, momentumdump_check
        savefigpath1 = Path+'FFI_PLD_plots/'
        savelcpath1 = Path+'FFI_PLD_LCs/'
        savefigpath2 = Path+'TPF_PLD_plots/'
        savelcpath2 = Path+'TPF_PLD_LCs/'    
        downloadpath = Path+'cache/'

        if cadence=='long':
            savefigpath=savefigpath1
            savelcpath=savelcpath1
            downloadpath=downloadpath

        if cadence=='short':        
            savefigpath=savefigpath2
            savelcpath=savelcpath2
            downloadpath=downloadpath
        ###
#         hdu,CCD,Camera,quality_mask,reference_pixel = gethdu(ID,Sector,cutoutsize=11,cadence=cadence,minimum_photon_counts=1,verbose=True,downloadpath=downloadpath)
#         if keep_FITS==False:
#             # deleting FITS files (no longer need them for light curve processing
#             # can always download again)
#             os.system("rm -r " + downloadpath) #delete cache path
        ###
        Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist = Get_stellar_params(ID,downloadpath)
        mdumps,t_0,t_1 = momentumdump_check(Sector)
        #calculate flux contamination of nearby stars within 3 TESS pixels
        flux_contamination_ratio = calc_flux_contamination(ID)
        ###        
        ###    
        ###    
        TLSbestfitdf = pd.DataFrame({"TLS Period [d]":TLS_periods, "TLS Period Error":TLS_periods_uncertainty,"TLS TC [BTJD]": TLS_t0s, "TLS depths [ppt]":(1-TLS_depths)*1000,"TLS depth Error":TLS_depths_err/1000,"TLS SDE":TLS_sde, "TLS SNR":TLS_snr,"TLS FAP":TLS_FAP,"TLS Dur [hrs]":TLS_Dur*24, "TLS Transit Count":TLS_transit_count,"TLS Distinct Transit Count":TLS_distinct_transit_count,"TLS Odd Even Mismatch":TLS_odd_even,"RP_RS":TLS_rp_rs,"Planet Radius [RE]":R_p,'Planet Radius Error':R_p_err,"CDPP [ppm/sqrt hr]": cdpp, 'Time Span [d]':time_span,'Stellar Radius [RS]':R_star, "Stellar Mass [MS]":M_star,"Teff [K]":Teff,"Flux Contamination Ratio":flux_contamination_ratio, "Vmag":Vmag,"TESSmag":Tmag,"rmag":rmag,"imag":imag,"zmag":zmag,"Jmag":Jmag,"Hmag":Hmag, "Kmag":Kmag,"Momentum Dump Rate [d]":mdumps,"RA":ra, "DEC":dec,"logg":logg,"rho [g/ccm]":rho,"dist [pc]":dist}, index=[0])
        ###        
        TLSbestfitdf.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_TLS_bestfit.txt",index=False)
        ###
        TLSTCsdf = pd.DataFrame({"TLS TCs [BTJD]":TLS_TCs,"TLS Depths":(1-TLS_depths_arr)*1000,"TLS Depths Error":TLS_depths_arr_uncertainty,"SNR Per Transit":TLS_snr_per_transit,"SNR Pink Per Transit":TLS_snr_pink_per_transit,})
        ###        
        TLSTCsdf.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_TLS_TCs.txt",index=False)
        ###
        return TLSdf, TLSmodeldf, TLSbestfitdf,TLSTCsdf
    else:
        print(" ")
        print("NaNs in mass or radius")
        print("Stellar Mass: ",M_star," Radius: ", R_star)
        print(" ")
        pass

#for determining BLS Period Error from HWHM of Peak in Power Spectrum
def peak(x, c):
    return np.exp(-np.power(x - c, 2) / 16.0)
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
def HWHM(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    hmx_left_right= [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]
    HWHM=0.5*(hmx_left_right[1] - hmx_left_right[0])
    return HWHM

def BLS_func(ID,Sector,cadence,time,flux,error,N_transits,minP,oversampling_factor,duration_grid_step,path,for_injections=False):
    from astropy.timeseries import BoxLeastSquares
    #for reporting BLS model's planet radius
    R_earth = 6.378*10.0**8.0 #cm
    R_sun = 6.955*10.0**10.0 #cm
    ###
    ###
    # calculate CDPP (in ppm per sqrt hour) to save later
    cdpp = CDPP(time,flux,error,'median','ppm',binsize=(1.0/24.0))    
    time_span = np.nanmax(time) - np.nanmin(time)
    #    
    from transitleastsquares import period_grid,duration_grid, catalog_info
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)

    # First lets make the grid
    LCduration = np.nanmax(time) - np.nanmin(time) #duration of light curve

    # lets pick min/max of orbital period grid
    maxP = LCduration/N_transits #orbital periods for grid

    periods = period_grid(R_star=R_star, M_star=M_star, time_span=LCduration, period_min=minP, period_max=maxP,oversampling_factor=oversampling_factor)
    durations= duration_grid(periods,shortest=None,log_step=duration_grid_step) 
    #shortest is unused in source code definition (why is it there?)    
    ###
    if np.isfinite(R_star)==True and np.isfinite(M_star)==True:

        #start BLS search
        bls = BoxLeastSquares(time, flux, error) 
        bls_power = bls.power(periods, durations)

        BLS_SDE = (bls_power.power - np.nanmean(bls_power.power))/np.nanstd(bls_power.power)
        
        #BLS results
        index = np.argmax(BLS_SDE) #finds strongest peak in BLS power spectrum
        BLS_Period=bls_power.period[index]
        try:
            BLS_Period_err = HWHM(bls_power.period[::-1],BLS_SDE)
        except (IndexError, ValueError) as E:
            BLS_Period_err = np.nan        
        BLS_T0=bls_power.transit_time[index]
        BLS_Depth=bls_power.depth[index]
        BLS_Depth_err=bls_power.depth_err[index]
        BLS_Dur = bls_power.duration[index]
        BLS_sde = BLS_SDE[index]
        
        #count transit times based on T0 and P
        from transitleastsquares.stats import all_transit_times
        BLS_transit_times = all_transit_times(BLS_T0, time, BLS_Period)
        BLS_transit_count = len(BLS_transit_times)
        
        R_p = np.sqrt(BLS_Depth)*R_star*R_sun/R_earth
        R_star_err = R_star_min
        R_p_err=R_p * np.sqrt((R_star_err/R_star)**2 + (BLS_Depth_err/BLS_Depth)**2)
        BLS_model=bls.model(time, BLS_Period, BLS_Dur, BLS_T0)                
        
        #saving results
        if for_injections==False:
            Path=path+'Sector_'+str(Sector)+'/'
        if for_injections==True:
            Path=path
        if cadence=='long':
            saveReportpath = Path+'FFI_BLS_Report/'
        if cadence=='short':            
            saveReportpath = Path+'TPF_BLS_Report/'
        ###
        if os.path.exists(saveReportpath)==True:
    #         print('folder exists, moving on...') #feel free to uncomment these out
            pass
        else:
    #         print('making directory') 
            os.makedirs(saveReportpath)
        ###
        import pandas as pd
        
        BLSdf = pd.DataFrame({"BLS Periods":bls_power.period, "BLS Power": bls_power.power, "BLS SDE":BLS_SDE})
        
        #check BLS Power Spectrum for oddities (like infinite peaks/depths)
        if np.nanmin(bls_power.power)==-np.inf or np.nanmax(bls_power.power)==np.inf:
            print('')
            print('PROBLEM with BLS: infinite values in Power. Check Light Curve for weird effects.')
            return None,None,None
        
        BLSmodeldf = pd.DataFrame({"Time":time , "Model": BLS_model})
        
#         from NEMESIS_pipeline import Make_dirs, Get_stellar_params, gethdu, momentumdump_check
        savefigpath1 = Path+'FFI_PLD_plots/'
        savelcpath1 = Path+'FFI_PLD_LCs/'
        savefigpath2 = Path+'TPF_PLD_plots/'
        savelcpath2 = Path+'TPF_PLD_LCs/'    
        downloadpath = Path+'cache/'

        if cadence=='long':
            savefigpath=savefigpath1
            savelcpath=savelcpath1
            downloadpath=downloadpath

        if cadence=='short':        
            savefigpath=savefigpath2
            savelcpath=savelcpath2
            downloadpath=downloadpath

        Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist = Get_stellar_params(ID,downloadpath)
        mdumps,t_0,t_1 = momentumdump_check(Sector)
        #calculate flux contamination of nearby stars within 3 TESS pixels
        flux_contamination_ratio = calc_flux_contamination(ID)
        ###        
        ###    
        ### 
        stats = bls.compute_stats(BLS_Period,BLS_Dur,BLS_T0)
        BLS_Depth_odd,BLS_Depth_odd_err = stats['depth_odd'][0],stats['depth_odd'][1]
        BLS_Depth_even,BLS_Depth_even_err = stats['depth_even'][0],stats['depth_even'][1]
        
        odd_even_difference = abs(BLS_Depth_odd - BLS_Depth_even)
        odd_even_std_sum = BLS_Depth_odd_err + BLS_Depth_even_err
        odd_even_mismatch = odd_even_difference / odd_even_std_sum
                
        BLSbestfitdf = pd.DataFrame({"BLS Period [d]":BLS_Period, "BLS Period Error":BLS_Period_err,"BLS TC [BTJD]": BLS_T0, "BLS depth [ppt]":(BLS_Depth)*1000,"BLS depth Error":BLS_Depth_err,"BLS SDE":BLS_sde, "BLS FAP":np.nan,"BLS Dur [hrs]":BLS_Dur*24, "BLS Odd Even Mismatch":odd_even_mismatch,"BLS Transit Count":BLS_transit_count, "Planet Radius [RE]":R_p,"Planet Radius Error":R_p_err,"CDPP [ppm/sqrt hr]": cdpp, 'Time Span [d]':time_span, 'Stellar Radius [RS]':R_star, "Stellar Mass [MS]":M_star,"Teff [K]":Teff, "Flux Contamination Ratio":flux_contamination_ratio,"Vmag":Vmag,"TESSmag":Tmag,"rmag":rmag,"imag":imag,"zmag":zmag,"Jmag":Jmag,"Hmag":Hmag, "Kmag":Kmag,"Momentum Dump Rate [d]":mdumps,"RA":ra, "DEC":dec,"logg":logg,"rho [g/ccm]":rho,"dist [pc]":dist}, index=[0])
        
        BLSdf.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_BLS.txt",index=False)
        BLSmodeldf.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_BLS_model.txt",index=False)
        BLSbestfitdf.to_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_BLS_bestfit.txt",index=False)
        
        return BLSdf, BLSmodeldf, BLSbestfitdf
    
    else:
        print(" ")
        print("NaNs in mass or radius")
        print("Stellar Mass: ",M_star," Radius: ", R_star)
        print(" ")
        pass
    
    
def TransitSearch(method, ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path,for_injections=False):    
    time = np.array(input_LC['Time'])
    flux = np.array(input_LC['Flux'])
    error= np.array(input_LC['Error'])
    ###
    if method=='TLS':
        TLSdf, TLSmodeldf, TLSbestfitdf,TLSTCsdf = TLS_func(ID,Sector,cadence,time,flux,error,N_transits,minP,oversampling_factor,duration_grid_step,path,for_injections)
        if isinstance(TLSdf, type(None)):
            return None,None,None
        else:
            PowerSpectrum_df = TLSdf
            TransitModel_df = TLSmodeldf
            TransitParams_df = TLSbestfitdf
        ###
    if method=='BLS':        
        BLSdf, BLSmodeldf, BLSbestfitdf = BLS_func(ID,Sector,cadence,time,flux,error,N_transits,minP,oversampling_factor,duration_grid_step,path,for_injections)
        
        if isinstance(BLSdf, type(None)):
            return None,None,None
        else:
            PowerSpectrum_df = BLSdf
            TransitModel_df = BLSmodeldf
            TransitParams_df = BLSbestfitdf
        
    return PowerSpectrum_df, TransitModel_df, TransitParams_df






def Transit_plot(ID,Sector,cadence,method,input_LC, PowerSpectrum_df,TransitModel_df, TransitParams_df, path, for_injections=False):
    import os
    if for_injections==False:
        Path=path+'Sector_'+str(Sector)+'/'
    if for_injections==True:
        Path=path
    
    if cadence=='long':
        if method=='BLS':
            saveReportpath = Path+'FFI_BLS_Report/'
        if method=='TLS':
            saveReportpath = Path+'FFI_TLS_Report/'
    if cadence=='short':            
        if method=='BLS':
            saveReportpath = Path+'TPF_BLS_Report/'
        if method=='TLS':
            saveReportpath = Path+'TPF_TLS_Report/'
    ###
    if os.path.exists(saveReportpath)==True:
#         print('folder exists, moving on...') #feel free to uncomment these out
        pass
    else:
#         print('making directory') 
        os.makedirs(saveReportpath)
    
    #getting stellar parameters from TIC
    from transitleastsquares import catalog_info
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    ###    
    fs = 9
    spacing = 0.02
    #for reporting TLS model's planet radius
    R_earth = 6.378*10.0**8.0 #cm
    R_sun = 6.955*10.0**10.0 #cm   
    ###
    time = np.array(input_LC['Time'])
    flux = np.array(input_LC['Flux'])
    error = np.array(input_LC['Error']) 
    
    #recalculate window size used for smoothing
    LCDur=(np.nanmax(time) - np.nanmin(time))
    maxP = LCDur/2
    R_planet_RE = 1
    
    # we want to keep the longest transit for an Earth-like planet for a single sector of data
    # using stellar parameters to determine transit duration
    window_size = 3*Tdur(maxP, R_star,M_star, R_planet_RE)
    
    ###
    if method=='BLS':
        P = TransitParams_df['BLS Period [d]'].item()
        T0= TransitParams_df['BLS TC [BTJD]'].item()
        Dur= TransitParams_df['BLS Dur [hrs]'].item()
        R_p= TransitParams_df['Planet Radius [RE]'].item() 
        
        Periods = np.array(PowerSpectrum_df['BLS Periods'])
        Power = np.array(PowerSpectrum_df['BLS SDE'])
        
        ModelT=np.array(TransitModel_df['Time']) 
        ModelF=np.array(TransitModel_df['Model']) 
        label1='BLS Period: '+str( np.round(P,4) )+" days; Transit Duration: "+str(np.round(Dur,4))+" hours"
        label2='BLS Model'
        
        savefile=saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_BLS.png"
    if method=='TLS':
        P = TransitParams_df['TLS Period [d]'].item()
        T0= TransitParams_df['TLS TC [BTJD]'].item()
        Dur= TransitParams_df['TLS Dur [hrs]'].item()
        R_p= TransitParams_df['Planet Radius [RE]'].item() 
        
        Periods = np.array(PowerSpectrum_df['TLS Periods'])
        Power = np.array(PowerSpectrum_df['TLS SDE'])
        
        ModelT=np.array(TransitModel_df['Time']) 
        ModelF=np.array(TransitModel_df['Model']) 
        label1='TLS Period: '+str( np.round(P,4) )+" days; Transit Duration: "+str(np.round(Dur,4))+" hours"
        label2='TLS Model'
        savefile=saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_TLS.png"
    ###
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(left=0.65, right=1.25, wspace=0.25,hspace=0.5)
    ###
    ###
    fig = plt.figure(figsize=(10,6))        
    ax1 = fig.add_subplot(gs1[0:1, 0:2])
    ###
    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.axvline(x=P,color='r',label=label1) #rounding period to 4 decimal places 
    ax1.axvline(x=0.5*P,color='r',linestyle='--')
    ax1.axvline(x=2.0*P,color='r',linestyle='--') 
    ###    
    mdumps,t_0,t_1 = momentumdump_check(Sector)
    ax1.axvline(x=mdumps,color='grey',linestyle='--',label='momentum dump rate (days): '+str(mdumps)) 
    ###    
    ax1.plot(Periods, Power, rasterized=True)
    ax1.set_title("TIC "+str(ID)+" "+"Sector "+str(Sector))#" Camera "+Camera+" CCD "+CCD)
    ax1.set_xlabel("Period (days)")
    ax1.set_xticks(np.arange(1.0, 15.0, 1.0))
    ax1.set_xlim(np.nanmin(Periods)-0.5, np.nanmax(Periods)+0.5)
    ###
    ax1.set_ylabel("SDE")
    ax1.set_ylim(np.nanmin(Power)-0.5, np.nanmax(Power)+0.5)
    ax1.legend(loc='best',fancybox=True,framealpha=0.5)
    ###
    ###    
    ax2 = fig.add_subplot(gs1[1:, 0:1])
    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.plot(time, flux ,c='red',markersize=4,marker='.',linestyle='none',zorder=1,label='detrended: windowsize: '+str(np.round(window_size*24,2))+' hrs')#+", binsize: "+str(bin_size)) 
    ax2.plot(ModelT,ModelF,'b.',label=label2,markersize=2.5)
    ax2.legend(loc='upper center',ncol=3,fontsize=fs)
    ax2.set_ylim(np.nanmin(flux)-3*np.nanstd(flux), np.nanmax(flux)+3*np.nanstd(flux))
    ###    
    #plotting momentum dumps
    t_0=np.nanmin(time)
    Num_mdumps = int(np.round((np.nanmax(time) - np.nanmin(time))/mdumps,2))+1
    print('')  
    ###
    ###    
    for N in range(Num_mdumps):
        time_mdump1 = t_0+(N)*mdumps
        time_mdump2 = t_1+(N+0.5)*mdumps    
        if time_mdump1 < t_1:
            ax2.axvline(x=time_mdump1,zorder=-2)
        if time_mdump2 < np.nanmax(time):
            ax2.axvline(x=time_mdump2,zorder=-2)
    ###    
    ax2.set_title("Star Radius: "+str(np.round(R_star,3))+r" $R_{\odot}$   Star Mass: "+str(np.round(M_star,3))+r" $M_{\odot}$")
    ax2.set_xlabel("Time ( JD)")
    ax2.set_ylabel("Normalized Flux") 
    ###
    ###
    pf_model,ff_model = phasefold(ModelT,T0,P,ModelF)
    pf,ff = phasefold(time,T0,P,flux)
    ###
    ###
    ax3 = fig.add_subplot(gs1[1:, 1:2])
    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    ax3.plot(24*pf,ff,c='red',markersize=6,marker='.',linestyle='none',zorder=1,label='detrended: windowsize: '+str(window_size*24)+' hrs')
    ax3.plot(24*pf_model,ff_model,'b.-',label=label2,markersize=5)
    ax3.set_xlabel("Orbital Phase (Hours)")
    ax3.set_ylabel("Normalized Flux") 
    ax3.set_title("Planet Radius: "+str(np.round(R_p,3))+" in Earth radii")
    ax3.set_ylim(np.nanmin(ModelF)-25*np.nanstd(ModelF), np.nanmax(ModelF)+25*np.nanstd(ModelF))
    # print('Dur check:', Dur)
    # print('Dur step:', int(3*Dur)/4)
    if Dur < 1:
        ax3.set_xticks(np.arange(-2,2+1,1))
    else:
        ax3.set_xticks(np.arange(int(-3*Dur),int(3*Dur)+int(3*Dur)/4,int(3*Dur)/4))
     # hours
    ax3.ticklabel_format(useOffset=False)
    ###
    if Dur> 8.0:
        ax3.set_xlim(-3*Dur,3*Dur)
    else:
        ax3.set_xlim(-5,5)
    ###
    if Dur< 1.0:
        ax3.set_xlim(-3*Dur,3*Dur)
    ###
    gs1.tight_layout(fig)
    ###
    plt.savefig(savefile)
#     plt.show()
    plt.close()
    
    
# functions needed to produce 1 page TLS Validation Reports
# functions needed to produce 1 page TLS Validation Reports
# functions needed to produce 1 page TLS Validation Reports
# functions needed to produce 1 page TLS Validation Reports
# functions needed to produce 1 page TLS Validation Reports



def plot_odd_even_transits(LC_df, TLSbestfit_df, TLSTCs_df, TLSmodel_df, ax,fig):
    ax.set_title('All Odd / Even Events')
    
    markersize=5
    fontsize=12
    #T_C_x_position and T_C_y_position control where the text appears for time stamps of "transit events"
    T_C_x_position = -0.55
    T_C_y_position =0.002
    
    time = np.array(LC_df['Time'])
    flux = np.array(LC_df['Detrended Flux'])
    error = np.array(LC_df['Detrended Error'])
        
    P = TLSbestfit_df['TLS Period [d]'].item()
    T0 = TLSbestfit_df['TLS TC [BTJD]'].item()
    Dur= TLSbestfit_df['TLS Dur [hrs]'].item()
    Depth = (TLSbestfit_df['TLS depths [ppt]'].item())/1000 #in ppo now
    
    spacing = 4* (Depth)
        
    XLIM=1.5*Dur
    
    T_C_array = np.array(TLSTCs_df['TLS TCs [BTJD]'])
    Depths_array = np.array(TLSTCs_df['TLS Depths'])
    Depths_err_array = np.array(TLSTCs_df['TLS Depths Error'])
    
    Modeltime = np.array(TLSmodel_df['Time'])
    Modelflux = np.array(TLSmodel_df['Model'])
    pf_model,ff_model = phasefold(T0,Modeltime,P,Modelflux)
    
    # cutting a 1 days worth of data around individual odd/even transits
    # and appending them into odd/even arrays for comparison
    
    
    window_size = 1 #day
    EvenDepths=[]
    OddDepths=[]
    Even=[]
    Evenflux=[]
    Odd=[]
    Oddflux=[]
    for x in range(len(T_C_array)):
        if x %2 ==0: #even
            EvenDepths=np.append(EvenDepths,Depths_array[x])
            cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
            cut_t  =  time[cut]
            cut_f  =  flux[cut]
            cut_fe = error[cut]
            if len(cut_f)<1: #in case window size is too small to cut data around
                window_size=1.5
                cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
                cut_t  =  time[cut]
                cut_f  =  flux[cut]
                cut_fe = error[cut]            
            ###
            phasefolded,foldedflux = phasefold(T_C_array[x],cut_t,P,cut_f)
            Even=np.append(Even,phasefolded)
            Evenflux=np.append(Evenflux,foldedflux)
            
            
        else: #odd
            OddDepths=np.append(OddDepths,Depths_array[x])
            cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
            cut_t  =  time[cut]
            cut_f  =  flux[cut]
            cut_fe = error[cut]
            if len(cut_f)<1: #in case window size is too small to cut data around
                window_size=1.5
                cut = np.where( ((T_C_array[x]-window_size) < time) & ((T_C_array[x]+window_size) > time)  )[0]
                cut_t  =  time[cut]
                cut_f  =  flux[cut]
                cut_fe = error[cut]
            phasefolded,foldedflux = phasefold(T_C_array[x],cut_t,P,cut_f)
            Odd=np.append(Odd,phasefolded)
            Oddflux=np.append(Oddflux,foldedflux)
            
    ax.plot(24*Odd,np.array(Oddflux)+spacing,color='lightblue',marker='.',linestyle='none',markersize=markersize+1, rasterized=True,label='Odd')            
    ax.plot(24*Even,np.array(Evenflux),color='dimgrey',marker='.',linestyle='none',markersize=markersize+1, rasterized=True,label='Even')
    ax.plot(24*pf_model,ff_model,'r.-',linewidth=1,markersize=2,label='TLS Model')
    ax.plot(24*pf_model,ff_model+spacing,'r.-',linewidth=1,markersize=2)
    ###
    ymax = np.nanmax(np.nanmean(ff_model)+2*spacing)
    ymin = np.nanmin(np.nanmean(ff_model)-spacing) 

    
    ax.set_xlim(-XLIM,XLIM)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel('Phase [Hours since '+str(np.round(T0,3))+' [BTJD]')
    ax.set_ylabel('Normalized Flux + Offset')
    
    
    #get odd/even metrics from TLS
    odd_even_mismatch = (TLSbestfit_df['TLS Odd Even Mismatch'].item()) #in standard deviations
    
    tx=0.085
    ty=0.915
    ax.text(tx,ty,'N Transits: '+str(len(T_C_array))+' O/E mismatch '+str(np.round(odd_even_mismatch,3))+r' ${\sigma}$', transform=fig.transFigure, size=fontsize-2)
    
    ax.axhline(y=1-np.nanmean(EvenDepths)/1000,color='green',linestyle='-')
    ax.axhline(y=1+spacing-np.nanmean(OddDepths)/1000,color='green',linestyle='-',label='Odd/Even Mismatch')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
#     ax.legend(by_label.values(), by_label.keys(),ncol=2,fontsize=fs-1)

    ax.annotate("Odd", xy=( -1, np.nanmean(ff_model)+1.5*spacing ), va='top',xycoords='data', fontsize=fontsize+4,weight="bold")
    ax.annotate("Even", xy=(-1, np.nanmean(ff_model)-0.5*spacing ), va='top',xycoords='data', fontsize=fontsize+4,weight="bold")

def plot_power_spectra(TLS_df,TLSbestfit_df, ax):
    #power spectra
    TLS_periods= TLS_df['TLS Periods']
    TLS_Power = TLS_df['TLS Power']
    
    #best fit params
    P = TLSbestfit_df['TLS Period [d]'].item()
    RP = TLSbestfit_df['Planet Radius [RE]'].item()
    Depth = TLSbestfit_df['TLS depths [ppt]'].item()
    mdumps=TLSbestfit_df['Momentum Dump Rate [d]'].item()
    
    ax.axvline(x=P,color='r')
    if 0.5*P> np.nanmin(TLS_periods):
        ax.axvline(x=0.5*P,color='r',linestyle='--')
    ###
    if 2.0*P < np.nanmax(TLS_periods):
        ax.axvline(x=2.0*P,color='r',linestyle='--')
        
    ax.plot(TLS_periods,TLS_Power, color='black', rasterized=True)

    ax.axvline(x=mdumps,color='grey',linestyle='--')
    ax.set_xlabel('Period [days]')
    ax.set_ylabel('TLS Power')
    ax.set_xticks(np.arange(np.nanmin(TLS_periods), np.nanmax(TLS_periods)+1, 1))
    if np.nanmax(TLS_Power)> 12:
        ax.set_yticks(np.arange(int(np.nanmin(TLS_Power)), int(np.nanmax(TLS_Power)+5), 5))
    if (np.nanmax(TLS_Power)>= 7) & (np.nanmax(TLS_Power)< 12):
        ax.set_yticks(np.arange(int(np.nanmin(TLS_Power)), int(np.nanmax(TLS_Power)+2), 2))        
    if np.nanmax(TLS_Power)< 7:
        ax.set_yticks(np.arange(int(np.nanmin(TLS_Power)), int(np.nanmax(TLS_Power)+1), 1))
    ax.set_title('TLS Power Spectrum: '+'Period '+str(np.round(P,3))+' d'+' Depth '+str(np.round(Depth,3))+' ppt'+' Planet Radius: '+str(np.round(RP,3))+' RE')    

    
def fullphasefold(time,T0,period,flux,offset):
    phase= (time - T0 + offset*period) / (period) - np.floor((time - T0 + offset*period) / period)
    ind=np.argsort(phase, axis=0)
    return phase[ind],flux[ind]
    
def plot_phasefold_LCs(ID,Sector,LC_df,TLS_df,TLSbestfit_df,TLSTCs_df,TLSmodel_df, axa,axb,axc,axd,axe):
    #fontsize
    fs=12
    
    #plots LC, PFLC, 0.5*P PFLC, 2*P PFLC and full PFLC    
    time=np.array(LC_df['Time'])
    flux=np.array(LC_df['Detrended Flux'])
    error=np.array(LC_df['Detrended Error'])
    sap_flux=np.array(LC_df['SAP Flux'])
    sap_error=np.array(LC_df['SAP Error'])
    
    
    modeltime = np.array(TLSmodel_df['Time'])
    modelflux = np.array(TLSmodel_df['Model'])
    
    T0 = TLSbestfit_df['TLS TC [BTJD]'].item()
    P = TLSbestfit_df['TLS Period [d]'].item()
    Dur=TLSbestfit_df['TLS Dur [hrs]'].item()
    Depth=TLSbestfit_df['TLS depths [ppt]'].item()/1000
    
    XLIM=3.5*Dur
    YLIM=2*Depth
    
    T_C_array = np.array(TLSTCs_df['TLS TCs [BTJD]'])
    
    #calculate full phase 0 to 1 + an offset to shift midtransit from 0 to offset
    offset=0.25
    fullphase, fullphaseflux = fullphasefold(time,T0,P,flux,offset)    
    
    #calculate phase in hours since T0 for 0.5x, 1x, 2x Period
    phasefolda, phasefoldfluxa = phasefold(time,T0,P,flux)
    phasefoldb, phasefoldfluxb = phasefold(time,T0,P*0.5,flux)
    phasefoldc, phasefoldfluxc = phasefold(time,T0,P*2.0,flux)
    
    #do same for transit models
    fullphase_model, fullphaseflux_model = fullphasefold(modeltime,T0,P,modelflux,offset)
    phasefold_modela, phasefoldflux_modela = phasefold(modeltime,T0,P,modelflux)
    phasefold_modelb, phasefoldflux_modelb = phasefold(modeltime,T0,0.5*P,modelflux)
    phasefold_modelc, phasefoldflux_modelc = phasefold(modeltime,T0,2.0*P,modelflux)

    #power spectra limits for PFLCs
    TLSPmin,TLSPmax = np.nanmin(np.array(TLS_df['TLS Periods'])) , np.nanmax(np.array(TLS_df['TLS Periods']))
    
    # plot LC
    cdpp_sap = CDPP(time,sap_flux,sap_error,'median','ppm',binsize=(1.0/24.0))
    cdpp_det = CDPP(time,flux,error,'median','ppm',binsize=(1.0/24.0))
    
    axa.set_title(r'Light Curve CDPPs: SAP CDPP = '+str(np.round(cdpp_sap,1))+' $\sigma _{ppm}$ ''hr$^{-1/2}$, Detrended CDPP ='+str(np.round(cdpp_det,1))+' $\sigma _{ppm}$ ''hr$^{-1/2}$') 
    
    mdumps,t_0,t_1 = momentumdump_check(Sector)
    t_0=np.nanmin(time) #sometimes data near beginning gets chopped based on TESS DRNs
    if Sector==31:
        t_0end =  2157.45371
        t_1end = 2169.94398
        time_mdump1 = t_0+ (t_0end - t_0)/2
        time_mdump2 = t_1+ (t_1end - t_1)/2
        axa.axvline(x=time_mdump1,zorder=-2)
        axa.axvline(x=time_mdump2,zorder=-2)        
    else:
        Num_mdumps = int(np.round((np.nanmax(time)-np.nanmin(time))/mdumps,2))+1
        for N in range(Num_mdumps):
            time_mdump1 = t_0+(N)*mdumps
            time_mdump2 = t_1+(N+0.5)*mdumps  
            if time_mdump1 < t_1:
                axa.axvline(x=time_mdump1,zorder=-2)          
            if time_mdump2 < np.nanmax(time):
                axa.axvline(x=time_mdump2,zorder=-2)
            
    axa.plot(time,flux,'k.',markersize=3,zorder=1, rasterized=True)
    axa.plot(np.array(TLSmodel_df['Time'].to_list()),np.array(TLSmodel_df['Model'].to_list())\
             ,'r.',markersize=1, rasterized=True)
    for x in range(len(T_C_array)):
        ### plotting 3 slightly overlapping to make it more obvious in tiny subplot window
        axa.plot(T_C_array[x], 1.0+1.5*Depth, marker=r'$\downarrow$',color='cyan', rasterized=True)
        axa.plot(T_C_array[x], 1.0+1.6*Depth, marker=r'$\downarrow$',color='cyan', rasterized=True)
        axa.plot(T_C_array[x], 1.0+1.7*Depth, marker=r'$\downarrow$',color='cyan', rasterized=True)        
        ###
#     tx=0.39
#     ty=0.8
#     axa.text(tx,ty,'Momentum Dump Rate: '+str(mdumps)+' days', transform=fig.transFigure, size=fs-2)
    axa.set_xlabel('Time [BTJD]')
    axa.set_ylabel('Norm. Flux')
    
    
    # plot PFLC
    axb.set_title('Phase Folded Light Curve',fontsize=fs-1)  
    axb.plot(24*phasefolda, phasefoldfluxa,'k.',markersize=3,zorder=0, rasterized=True)
    axb.plot(24*phasefold_modela, phasefoldflux_modela,'r.-',markersize=2,zorder=1, rasterized=True)
    axb.set_xlabel(r'Phase [Hours since '+str(np.round(T0,3))+' [BTJD]')
    axb.set_ylabel('Norm. Flux')
    
    # plot full PFLC
    axc.set_title("Full Phase Folded Light Curve",fontsize = fs)
    axc.plot(fullphase, fullphaseflux,'k.',markersize=3,zorder=0, rasterized=True)    
    axc.plot(fullphase_model, fullphaseflux_model,'r.-',markersize=2,zorder=1, rasterized=True) 
    axc.set_xlabel('Phase + 0.25')
    axc.set_ylabel('Norm. Flux')
    
    # plot PFLC with 0.5x P
    axd.set_title('0.5x Period = '+(str(np.round(0.5*P,3)))+' days')
    axd.plot(24*phasefoldb, phasefoldfluxb,'k.',markersize=3,zorder=0, rasterized=True)
    #models never looks good at 1/2x 
    # axd.plot(24*phasefold_modelb, phasefoldflux_modelb,'r.-',markersize=2,zorder=1, rasterized=True)
    #models never looks good at 1/2x 
    axd.set_xlabel(r'Phase [Hours since '+str(np.round(T0,3))+' [BTJD]')
    axd.set_ylabel('Norm. Flux')
    
    # plot PFLC with 2x P
    axe.set_title('2x Period = '+(str(np.round(2*P,3)))+' days')
    axe.plot(24*phasefoldc, phasefoldfluxc,'k.',markersize=3,zorder=0, rasterized=True)
    axe.plot(24*phasefold_modelc, phasefoldflux_modelc,'r.-',markersize=2,zorder=1, rasterized=True)
    axe.set_xlabel(r'Phase [Hours since '+str(np.round(T0,3))+' [BTJD]')
    axe.set_ylabel('Norm. Flux')
    
    
    axc.set_xticks(np.arange(0.0, 1+0.25, 0.25))    
    if XLIM < 8:
        axb.set_xticks(np.arange(int(-XLIM), int(XLIM)+1, 1.0))        
        axd.set_xticks(np.arange(int(-XLIM), int(XLIM)+1, 1.0))
        axe.set_xticks(np.arange(int(-XLIM), int(XLIM)+1, 1.0))
    if XLIM > 8:
        axb.set_xticks(np.arange(int(-XLIM), int(XLIM)+2, 2.0))
        axd.set_xticks(np.arange(int(-XLIM), int(XLIM)+2, 2.0))
        axe.set_xticks(np.arange(int(-XLIM), int(XLIM)+2, 2.0))
        
        
    axb.set_xlim(-XLIM,XLIM)
    axc.set_xlim(-0.01,1.01)
    axd.set_xlim(-XLIM,XLIM)
    axe.set_xlim(-XLIM,XLIM)
    
    axa.set_ylim(1-YLIM,1+YLIM)
    axb.set_ylim(1-YLIM,1+YLIM)
    axc.set_ylim(1-YLIM,1+YLIM)
    axd.set_ylim(1-YLIM,1+YLIM)
    axe.set_ylim(1-YLIM,1+YLIM)
    #turn off exponential notiation in axes
    axa.ticklabel_format(useOffset=False)
    axb.ticklabel_format(useOffset=False)
    axc.ticklabel_format(useOffset=False)
    axd.ticklabel_format(useOffset=False)
    axe.ticklabel_format(useOffset=False)


        
def Get_FFI(ID,Sector,cadence,path,use_SPOC_aperture='no',for_injections=False):
    #Step 0: Creating directories to save figures and data
    import pandas as pd   
    verbose=False
    if for_injections==False:
        Path=path+'Sector_'+str(Sector)+'/'
    if for_injections==True:
        Path=path
    if cadence=='long':
        saveReportpath = Path+'FFI_TLS_Report/'
        savelcpath= Path+'FFI_PLD_LCs/'
        downloadpath = Path+'cache/'
    if cadence=='short':            
        saveReportpath = Path+'TPF_TLS_Report/'
        savelcpath= Path+'TPF_PLD_LCs/'
        downloadpath = Path+'cache/'
    try:
        bkg_mask = readNDarr(savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bkg_mask")
        pix_mask = readNDarr(savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_pix_mask")
        images = readNDarr(savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_image_fluxes")
        median_image = np.nanmedian(images, axis=0)
        
        try:
            hdu,CCD,Camera,quality_mask,reference_pixel = gethdu(ID,Sector,cutoutsize=11,cadence=cadence,\
                                                                 minimum_photon_counts=1,verbose=True,\
                                                                 downloadpath=downloadpath)
        except TypeError as TE:
            print(TE)
            import time as clock
            os.system('rm -r ~/.astropy/cache/download/py3/lock') #clear any locks that might be in cache
            clock.sleep(10) #wait 10 seconds and try again
            hdu,CCD,Camera,quality_mask,reference_pixel = gethdu(ID,Sector,cutoutsize=11,cadence=cadence,\
                                                                 minimum_photon_counts=1,verbose=True,\
                                                                 downloadpath=downloadpath)        
    except FileNotFoundError as FNFE:
        print('')
        print(FNFE)
        print('recreating cutouts, aperture and background masks with default settings')
        print(' ')
        #Step 1: Download FFI Cutout from MAST
        # sometimes MAST/Astropy has issues, if it fails try again
        # if it got to this point, the FFI definitely exists!
        try:
            hdu,CCD,Camera,quality_mask,reference_pixel = gethdu(ID,Sector,cutoutsize=11,cadence=cadence,\
                                                                 minimum_photon_counts=1,verbose=True,\
                                                                 downloadpath=downloadpath)
        except TypeError as TE:
            print(TE)
            import time as clock
            os.system('rm -r ~/.astropy/cache/download/py3/lock') #clear any locks that might be in cache
            clock.sleep(10) #wait 10 seconds and try again
            hdu,CCD,Camera,quality_mask,reference_pixel = gethdu(ID,Sector,cutoutsize=11,cadence=cadence,\
                                                                 minimum_photon_counts=1,verbose=True,\
                                                                 downloadpath=downloadpath)
            print('')            
        #step 2: get aperture and background masks
        bkg_mask, pix_mask ,flux, median_image, SAP_LC = SAP(ID=ID,Sector=Sector,cutoutsize=11,hdu=hdu,\
                                                             quality_mask=quality_mask,threshold=7.5,cadence=cadence,\
                                                             reference_pixel=reference_pixel,verbose=False,\
                                                             savelcpath=savelcpath,use_SPOC_aperture='no')   
        #resave pkl data
        saveNDarr(pix_mask,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_pix_mask")
        saveNDarr(bkg_mask,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bkg_mask")
        saveNDarr(flux,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_image_fluxes")    
    ###
    #Step 3: Get information on target star and apply some basic selection cuts
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)   
    ###
    ###
    #Get more stellar params
    ###
    Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist = Get_stellar_params(ID,downloadpath)
    ###
    CCD=hdu[0].header['CCD']
    Camera=hdu[0].header['Camera']    
    wcs = WCS(hdu[2].header)
    return median_image, hdu, wcs, pix_mask, bkg_mask, Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist,CCD,Camera        
        
        
def plot_image(ID,Sector,cadence,path,ax_placement,fig,fs,for_injections=False):
    if for_injections==False:
        Path=path+'Sector_'+str(Sector)+'/'
    if for_injections==True:
        Path=path
    if cadence=='long':
        saveReportpath = Path+'FFI_TLS_Report/'
        savelcpath= Path+'FFI_PLD_LCs/'
        downloadpath = Path+'cache/'
    if cadence=='short':            
        saveReportpath = Path+'TPF_TLS_Report/'
        savelcpath= Path+'TPF_PLD_LCs/'
        downloadpath = Path+'cache/'
        
    #get image data and stellar params
    median_image, hdu,wcs, pix_mask, bkg_mask,Vmag,Tmag,Gmag,rmag,imag,zmag,Jmag,Hmag,Kmag,Teff,ra,dec,logg,rho,dist,CCD,Camera = Get_FFI(ID,Sector,cadence,path,for_injections=for_injections)    
    
    ax = fig.add_subplot(ax_placement,projection=wcs)
    if cadence=='short':
        x=hdu[1].header['1CRPX4']-1
        y=hdu[1].header['2CRPX4']-1
        ax.set_title("TPF Cutout",fontsize = fs)
    if cadence=='long':
        x=hdu[1].header['1CRPX4']
        y=hdu[1].header['2CRPX4']
        ax.set_title("FFI Cutout",fontsize = fs)
    reference_pixel=[x,y]    
    axes=[ax]
    plot_cutouts(ID,Sector,cadence,hdu,pix_mask,bkg_mask,reference_pixel,fig,axes,savelcpath,downloadpath,do_colorbar='no')
    
    
# Let's work on getting DSS images along with our FFI cutouts
def getDSS(ID,cutoutsize,downloadpath):

    # astroquery
    from astroquery.mast import Tesscut
    from astroquery.mast import Catalogs

    # Astropy
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.visualization import simple_norm

    from reproject import reproject_interp
    import socket 
    import urllib
    import time as clock
    import requests
    
    from astroplan import FixedTarget
#     from astroplan.plots import plot_finder_image
    from astroquery.skyview import SkyView
    from astroplan import download_IERS_A
    
    # changing cache directories
    Tesscut.cache_location=downloadpath
    Catalogs.cache_location=downloadpath
    SkyView.cache_location=downloadpath
    
    starName="TIC "+str(ID)
    degrees = 21/3600 #21 arcsec to degrees
    radSearch = degrees # angular radius in degrees
    catalogData = Catalogs.query_object(starName, radius = radSearch, catalog = "TIC")

    #checking to see if target is correct in catalog and not nearby stars
    for x in range(len(catalogData['ID'])):
        if int(catalogData['ID'][x])==ID:

            ra = catalogData[x]['ra']
            dec = catalogData[x]['dec']
            Tmag = catalogData[x]['Tmag']
            Teff = catalogData[x]['Teff']
            Vmag = catalogData[x]['Vmag'] 
            coord = SkyCoord(ra, dec, unit = "deg")
    try:
        download_IERS_A()
    except (socket.timeout,FileNotFoundError,RuntimeError) as STO:
        print('')
        print('DSS Request Timeout(?)')
        print(STO)
        print('trying again')
        clock.sleep(10)
        #clear any potential locks in cache
        os.system('rm -r ~/.astropy/cache/download/py3/lock') 
        try:
            download_IERS_A()
        except (socket.timeout,FileNotFoundError,RuntimeError) as STO:
            print('DSS Request Timeout Again(?)')
            print('...oh well?')
            #clear any potential locks in cache
            os.system('rm -r ~/.astropy/cache/download/py3/lock') 
            pass
    except RuntimeError as RE:
        print(RE)
        print('')
        #os.system('rm -r ~/.astropy/cache/download/py3/lock') 
        clock.sleep(10)
        try: 
            download_IERS_A()
        except socket.timeout as STO:
            print('DSS Request Timeout Again')
            print('...oh well?')
            pass
    survey = 'DSS2 Red'
    target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    sizepix=(cutoutsize)*21*u.arcsec
    fov_radius = (cutoutsize)*21*u.arcsec
    grid=False
    
    try:
        target = FixedTarget(coord=target_coord, name="Survey = {}".format(survey))
    except: # if DSS2 Red is not available, download the DSS field of view image instead
        survey = 'DSS'
        target = FixedTarget(coord=target_coord, name="Survey = {}".format(survey))

    coord = target if not hasattr(target, 'coord') else target.coord
    position = coord.icrs
    coordinates = 'icrs'
    target_name = None if isinstance(target, SkyCoord) else target.name

    dss_pixel_scale=1.7
    tess_pixel_scale=21
    
    fficutout=cutoutsize*tess_pixel_scale
    
    #translate to dss pixel scale
    npixels_for_dss=int(fficutout*dss_pixel_scale) 
    try:
        hdu = SkyView.get_images(position=position, coordinates=coordinates,
                                 survey=survey, radius=fov_radius, pixels=npixels_for_dss, grid=grid)[0][0]
        wcs = WCS(hdu.header)
        return hdu,hdu.data,wcs
    except (urllib.error.HTTPError,RuntimeError) as URLE:
        print(' ')
        print(URLE)
        print('problem with getting DSS image')
        print(' ')
        return None,None,None
    except (requests.exceptions.ReadTimeout, astroquery.exceptions.TimeoutError,socket.timeout) as STO:
        print('')
        print('DSS Request Timeout')
        print(STO)
        print('trying again')
        clock.sleep(60)
        try:
            hdu = SkyView.get_images(position=position, coordinates=coordinates,
                                     survey=survey, radius=fov_radius, pixels=npixels_for_dss, grid=grid)[0][0]
            wcs = WCS(hdu.header)
            return hdu,hdu.data,wcs
        except urllib.error.HTTPError as URLE:
            print(' ')
            print(URLE)
            print('problem with getting DSS image')
            print(' ')
            return None,None,None
        except (requests.exceptions.ReadTimeout, astroquery.exceptions.TimeoutError,socket.timeout):
            return None,None,None

def plot_centroids_in_phase(LC_df,TLSbestfit_df,ax):
    fs=12
    ax.set_title("Centroid Motion",fontsize = fs)
    
    centx= np.array(LC_df['Centroid X Positions'])
    centy =np.array(LC_df['Centroid Y Positions'])
    time = np.array(LC_df['Time'])

    centx-=np.median(centx)
    centy-=np.median(centy)
    
    sigmalimx = 5*np.nanstd(centx)
    sigmalimy = 5*np.nanstd(centy)
    

    T0 =TLSbestfit_df['TLS TC [BTJD]'].item()
    P =TLSbestfit_df['TLS Period [d]'].item()
    Dur=TLSbestfit_df['TLS Dur [hrs]'].item()
    XLIM=3.5*Dur
    
    pfx,xx = phasefold(T0, time,P,centx)
    pfy,yy = phasefold(T0, time,P,centy)
    
    axb = ax.twinx() #make 2nd y axis for Y centroid positions
    
    ax.plot(24*pfx,xx,'k.')
    axb.plot(24*pfy,yy,'r.')
    ax.set_xlim(-XLIM,XLIM)
    ax.set_ylabel('Delta X')
    axb.set_ylabel('Delta Y',rotation=270)
    axb.yaxis.label.set_color('red')
    axb.tick_params(axis='y', colors='red')
    ax.set_xlabel('Phase [Hours since '+str(np.round(T0,3))+' [BTJD]')
    ax.axhline(y=sigmalimx,color='green',linestyle='--')
    ax.axhline(y=-sigmalimx,color='green',linestyle='--')
    ax.axhline(y=sigmalimy,color='cyan',linestyle='--')
    ax.axhline(y=-sigmalimy,color='cyan',linestyle='--')
    
def plot_dss_orientation(ax,ID,downloadpath,cutoutsize=11,do_DSS_plot=True):
    fs=12
    ax.set_title("DSS",fontsize = fs)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    if do_DSS_plot==True:
        dsshdu,dss,wcs=getDSS(ID,cutoutsize,downloadpath)
    if do_DSS_plot==False:
        #for transit injections don't do this
        print('NOT doing getDSS for injections (takes too long...)')
        #dsshdu,dss,wcs=getDSS(ID,cutoutsize,downloadpath)
        #for transit injections don't do this
        dsshdu,dss,wcs = None, None, None
    
    if isinstance(dss, type(None)):
        print('no DSS image, see above output')
        ax.text(0.5, 0.5, 'No DSS Image', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    else:    
        ax.imshow(dss,cmap=plt.cm.Greys)
        ###
        northisup=True
        eastisright=False #default?
        eastisright=True #force it to be true!

        lwr = 2.5
        cr = 'firebrick'
        arrowkwargs = {'width':0.5, 'headwidth':4, 'shrink':0.05, 'color':cr, 'alpha':0.5}

        if northisup==True:
            if eastisright==True:
                ax.invert_xaxis()
                ax.invert_yaxis() #<---is this right?
                shift=0.025
                ax.annotate('', xy=(0.01+shift, 0.25), xytext=(0.01+shift, 0.05), xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrowkwargs)
                ax.annotate('N', xy=(0.01+shift, 0.255), xycoords="axes fraction", color=cr)
                ax.annotate('', xy=(0.2+shift, 0.060), xytext=(0.+shift, 0.060), xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrowkwargs)
                ax.annotate('E', xy=(0.22+shift, 0.022), xycoords="axes fraction", color=cr)

            else:
                ax.invert_yaxis()
                ax.annotate('', xy=(0.95, 0.25), xytext=(0.95, 0.05), xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrowkwargs)
                ax.annotate('N', xy=(0.94, 0.255), xycoords="axes fraction", color=cr)
                ax.annotate('', xy=(0.76, 0.060), xytext=(0.96, 0.060), xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrowkwargs)
                ax.annotate('E', xy=(0.73, 0.022), xycoords="axes fraction", color=cr)

            
            

def plot_text(ID,Sector,TLSbestfit_df,TLSTCs_df,EDI_results,fig):
    fs=12
    star_header='Stellar Parameters for TIC '+str(ID) 
    TLStxt_header='TLS Results'
    startxt_array=['Stellar Mass [MS]','Stellar Radius [RS]','Teff [K]', 'Vmag', 'TESSmag', 'Jmag', 'Hmag', 'Kmag','dist [pc]','logg','RA', 'rho [g/ccm]','DEC']
    TLStxt_array=['TLS Period [d]', 'TLS TC [BTJD]', 'TLS Dur [hrs]','TLS depths [ppt]','Planet Radius [RE]','TLS SDE', 'TLS Odd Even Mismatch','TLS FAP']
    
    
    vertspacing=0.01
    horispacing=1.1
    ###
    fontx=0.6275
    fonty=0.325
    #star stuff
    N=fonty-0.025
    fig.text(fontx+0.01,N,star_header, transform=fig.transFigure, size=fs+4)
    N=5
    FS=fs-2
    for x in range(0,len(startxt_array)):
        if x==(len(startxt_array)-1):
            fig.text(fontx+horispacing*0.175,fonty-(N+x)*vertspacing,"Sector : "+str(Sector), transform=fig.transFigure, size=FS)
        #left
        if x % 2 == 0:
            if startxt_array[x]=='rho [g/ccm]':
                text=r'${\rho}$ [g/cm$^{3}$]'
                fig.text(fontx,fonty-(N+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[startxt_array[x]].item(),3)),  size=FS)
            else:
                fig.text(fontx,fonty-(N+x)*vertspacing,startxt_array[x]+" : "+str(np.round(TLSbestfit_df[startxt_array[x]].item(),3)),  size=FS)
        else: #right
            if startxt_array[x]=='rho [g/ccm]':
                text=r'${\rho}$ [g/cm$^{3}$]'
                fig.text(fontx+horispacing*0.175,fonty-(N-1+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[startxt_array[x]].item(),3)),  size=FS)
            else:
                fig.text(fontx+horispacing*0.175,fonty-(N-1+x)*vertspacing,startxt_array[x]+" : "+str(np.round(TLSbestfit_df[startxt_array[x]].item(),3)),  size=FS)
    ###
    #TLS stuff
    N=N+len(startxt_array)+2
    fig.text(fontx+horispacing*0.175/2,fonty-N*vertspacing,TLStxt_header, transform=fig.transFigure, size=fs+4)
    N=N+3
    for x in range(0,len(TLStxt_array)):
        if x % 2 == 0: #left
            if TLStxt_array[x]=='TLS FAP':
                text = str(TLSbestfit_df['TLS FAP'].item())#"
                text = "{:.2e}".format(TLSbestfit_df['TLS FAP'].item())
                fig.text(fontx,fonty-(N+x)*vertspacing,str(TLStxt_array[x])+" : "+text, transform=fig.transFigure, size=FS,color='black')
            elif TLStxt_array[x]=='TLS Odd Even Mismatch':
                text='Odd/Even Mismatch'
                if TLSbestfit_df[TLStxt_array[x]].item() > 5:
                    fig.text(fontx,fonty-(N+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3))+r' ${\sigma}$', transform=fig.transFigure, size=FS,color='red')
                else:
                    fig.text(fontx,fonty-(N+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3))+r' ${\sigma}$', transform=fig.transFigure, size=FS)
            elif TLStxt_array[x]=='TLS depths [ppt]':
                text='TLS Depth [ppt]'
                fig.text(fontx,fonty-(N+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3)), transform=fig.transFigure, size=FS)                
            else:
                fig.text(fontx,fonty-(N+x)*vertspacing,TLStxt_array[x]+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3)), transform=fig.transFigure, size=FS)
        else: #right
            if TLStxt_array[x]=='TLS FAP':
                text = str(TLSbestfit_df['TLS FAP'].item())#"{:.2e}".format(TLSbestfit_df['TLS FAP'].item())
                text = "{:.2e}".format(TLSbestfit_df['TLS FAP'].item())                
                fig.text(fontx+horispacing*0.175,fonty-(N-1+x)*vertspacing,str(TLStxt_array[x])+" : "+text, transform=fig.transFigure, size=FS,color='black')
            elif TLStxt_array[x]=='TLS Odd Even Mismatch':
                text='Odd/Even Mismatch'
                if TLSbestfit_df[TLStxt_array[x]].item() > 5:
                    fig.text(fontx+horispacing*0.175,fonty-(N-1+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3))+r' ${\sigma}$', transform=fig.transFigure, size=FS,color='red')
                else:
                    fig.text(fontx+horispacing*0.175,fonty-(N-1+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3))+r' ${\sigma}$', transform=fig.transFigure, size=FS)
            elif TLStxt_array[x]=='TLS depths [ppt]':
                text='TLS Depth [ppt]'
                fig.text(fontx+horispacing*0.175,fonty-(N-1+x)*vertspacing,text+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3)), transform=fig.transFigure, size=FS)                                
            else:
                fig.text(fontx+horispacing*0.175,fonty-(N-1+x)*vertspacing,TLStxt_array[x]+" : "+str(np.round(TLSbestfit_df[TLStxt_array[x]].item(),3)), transform=fig.transFigure, size=FS)
                
    #place text at bottom (only whitespace left...)
    edispace=0.1225
    edix=0.02
    ediy=0.98
    edicolor='red'
    #finding where EDI Vetter produced False Positive flags
    EDI_cols=list(EDI_results.columns.values)[2:] #ignoring ID,Sector columns
    for FP in range(len(EDI_cols)):
        if EDI_results[EDI_cols[FP]].item()==False:
            fig.text(edix+FP*edispace,ediy,EDI_cols[FP],fontsize=fs-5,color='black')    
        if EDI_results[EDI_cols[FP]].item()==True:
            fig.text(edix+FP*edispace,ediy,EDI_cols[FP],fontsize=fs-5,color=edicolor)      


def TLS_Report(ID,Sector,cadence,path,keep_FITS=False,keep_imagedata=True, for_injections=False):
    ###
    # making a 1 page validation report summarizing the overall transit search    
    ###
    if for_injections==False:
        Path=path+'Sector_'+str(Sector)+'/'
    if for_injections==True:
        Path=path
    #
    if cadence=='long':
        saveReportpath = Path+'FFI_TLS_Report/'
        savelcpath= Path+'FFI_PLD_LCs/'
        downloadpath = Path+'cache/'
    if cadence=='short':            
        saveReportpath = Path+'TPF_TLS_Report/'
        savelcpath= Path+'TPF_PLD_LCs/'
        downloadpath = Path+'cache/'
    ###        
    #creating directory if it already doesn't exist
    if os.path.exists(saveReportpath)==True:
        pass
    else:
        os.makedirs(saveReportpath)
    ###
    #in case it was deleted by a previous run:
    if os.path.exists(downloadpath)==True:
        pass
    else:
        os.makedirs(downloadpath)
    ###
    ###
    #files needed to compile results
    # light curves and centroids
    LC_df = pd.read_csv(savelcpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_final_LC.txt')
    ###
    # TLS results
    TLS_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS.txt')
    TLSmodel_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS_model.txt')
    TLSbestfit_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS_bestfit.txt')
    TLSTCs_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS_TCs.txt')
    TLStxt_array=list(TLSbestfit_df.columns.values)
    ###
    # EDI-Vetter Results for False Positive Flags
    EDI_results = pd.read_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_EDI_results.txt",index_col=0)
    ###
    #reading in a lot of dataframes can take up RAM, this can help clear it with gc.collect()
    import gc 
    #reading in a lot of dataframes can take up RAM, this can help clear it with gc.collect()
    ###
    import time as clock
    start=clock.time()
    ###
    #fontsize
    fs=12
    ###
    fig = plt.figure(figsize=(12,10))
    gs1 = gridspec.GridSpec(6, 3)#4,2
    #3x3 grid, (left,middle,right) helps makes sense of placement. gs1[height, width]
    left=0
    middle=1
    right=2
    ###    
    #Odd/Even plot
    ax0 = fig.add_subplot(gs1[0:2,left])
    plot_odd_even_transits(LC_df, TLSbestfit_df, TLSTCs_df, TLSmodel_df, ax0,fig)
    ###    
    #Lightcurve
    ax1 = fig.add_subplot(gs1[0, middle:])
    ###    
    #Power Spectrum
    ax2 = fig.add_subplot(gs1[1, middle:])
    plot_power_spectra(TLS_df,TLSbestfit_df, ax2)
    ###
    #Phasefolded Light Curves
    ax3 = fig.add_subplot(gs1[2, middle])
    ax4 = fig.add_subplot(gs1[2, right])
    ax5 = fig.add_subplot(gs1[3, middle])
    ax6 = fig.add_subplot(gs1[3, right])
    plot_phasefold_LCs(ID,Sector,LC_df,TLS_df,TLSbestfit_df,TLSTCs_df,TLSmodel_df, ax1,ax3,ax4,ax5,ax6)
    ###    
    # TESS Image Cutout
    ax7_placement=gs1[2:4,left]
    plot_image(ID,Sector,cadence,path,ax7_placement,fig,fs,for_injections=for_injections)
    ###    
    #Centroid Motion plot
    ax8 = fig.add_subplot(gs1[4:6,left])
    plot_centroids_in_phase(LC_df,TLSbestfit_df,ax8)
    ###
    # DSS Image
    ax9 = fig.add_subplot(gs1[4:6, middle])
    plot_dss_orientation(ax9,ID,downloadpath,cutoutsize=11)
    ###    
    #plot text from stellar and planet parameters
    plot_text(ID,Sector,TLSbestfit_df,TLSTCs_df,EDI_results,fig)
    ###
    gs1.update(wspace=0.0, hspace=0.0)
    gs1.tight_layout(fig)
    fig.savefig(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLSReport.png',bbox_inches='tight')
#     fig.show()
    plt.close()
    ###
    end=clock.time()

    runtime=end-start

    # clear garbage collection in RAM
    gc.collect()

    if runtime > 60:
        print('report runtime: '+str(runtime/60)+' minutes')
    if runtime < 60:
        print('report runtime: '+str(runtime)+' seconds')    
        
    ###
    ### Last Step: Clear image data (FITS and PKL files) to save space
    ### (optional)
    ###            
    #delete image data(aperture/background masks, cutout images)
    if keep_imagedata==False:
        pixmask_filename="TIC_"+str(ID)+"_Sector_"+str(Sector)+"_pix_mask"
        bkgmask_filename="TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bkg_mask"
        image_filename="TIC_"+str(ID)+"_Sector_"+str(Sector)+"_image_fluxes"                
        os.system("rm " + savelcpath+pixmask_filename+'.pkl') 
        os.system("rm " + savelcpath+bkgmask_filename+'.pkl') 
        os.system("rm " + savelcpath+image_filename+'.pkl')    
    if keep_FITS==False:
            # deleting FITS files (no longer need them for light curve processing
            # can always download again)
            os.system("rm -r " + downloadpath) #delete cache path                    
            
def Transit_Pipeline(threshold, ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path,keep_FITS=True,keep_imagedata=True,for_injections=False):
    
    ### first do BLS
    print('')
    print('doing BLS search')
    PowerSpectrum_df, TransitModel_df, TransitParams_df = TransitSearch('BLS',ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path,for_injections)
    ###
    if isinstance(PowerSpectrum_df, type(None)):
        print('problem with infinite/nan values inPower Spectrum')
        return None, None, None, None, None, None
    #make BLS Plot
    Transit_plot(ID,Sector,cadence,'BLS',input_LC, PowerSpectrum_df,TransitModel_df, TransitParams_df, path, for_injections)
    ###
    ### check if SDE is above threshold
    if (np.nanmax(np.array(PowerSpectrum_df['BLS SDE']))) > threshold:
        print('')
        print('BLS max peak > threshold!')
        print('')
        PowerSpectrum_df2, TransitModel_df2, TransitParams_df2 = TransitSearch('TLS',ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path, for_injections)
        if isinstance(PowerSpectrum_df2, type(None)):
            print('problem with infinite/nan values inPower Spectrum')
            return None, None, None, None, None, None        
        else:
            ###
            #make TLS Plot
            Transit_plot(ID,Sector,cadence,'TLS',input_LC, PowerSpectrum_df2,TransitModel_df2, TransitParams_df2, path, for_injections)
            ###
            # make TLS Report
            TLS_Report(ID,Sector,cadence,path,keep_FITS=keep_FITS,keep_imagedata=keep_imagedata)
        
        
        return PowerSpectrum_df, TransitModel_df, TransitParams_df, PowerSpectrum_df2, TransitModel_df2, TransitParams_df2
    else:
        print('BLS signal < threshold')
        return PowerSpectrum_df, TransitModel_df, TransitParams_df, None,None,None
    
    
##########################################
##########################################
####### TRANSIT SEARCHING FUNCTIONS ######
##########################################
##########################################




##########################################
##########################################
####### TRANSIT INJECTION FUNCTIONS ######
##########################################
##########################################

def Make_dirs_injection(path,Sector,cadence,Period,R_planet_RE):
    import os
    #Step 0: Creating directories to save figures and data
    path=path+'Sector_'+str(Sector)+'/'
    path = path+'Period_'+str(np.round(Period,2))+'_RP_'+str(np.round(R_planet_RE,2))+'/'
    
    savefigpath1 = path+'FFI_PLD_plots/'
    savelcpath1 = path+'FFI_PLD_LCs/'
    savefigpath2 = path+'TPF_PLD_plots/'
    savelcpath2 = path+'TPF_PLD_LCs/'    
    downloadpath = path+'cache/'
    ###
    if cadence=='long':
        savefigpath=savefigpath1
        savelcpath=savelcpath1
        downloadpath=downloadpath
        if os.path.exists(savefigpath1)==True:
            pass
        else: 
            os.makedirs(savefigpath1)
        if os.path.exists(savelcpath1)==True:
            pass
        else:
            os.makedirs(savelcpath1) 
        if os.path.exists(downloadpath)==True:
            pass
        else: 
            os.makedirs(downloadpath)                        
    if cadence=='short':        
        savefigpath=savefigpath2
        savelcpath=savelcpath2
        downloadpath=downloadpath
        if os.path.exists(savefigpath2)==True:
            pass
        else: 
            os.makedirs(savefigpath2)
        if os.path.exists(savelcpath2)==True:
            pass
        else:
            os.makedirs(savelcpath2)
        if os.path.exists(downloadpath)==True:
            pass
        else: 
            os.makedirs(downloadpath)            
    ###  
    return path, savefigpath, savelcpath, downloadpath


def transit_injection(input_LC, Period, T0, R_planet_RE, ID, Sector):
    import pandas as pd    
    
    T=np.array(input_LC['Time'])
    F=np.array(input_LC['SAP Flux'])
    E=np.array(input_LC['SAP Error'])
    #calculate the cadence (exposure time) in TESS data
    texp=np.nanmedian(np.diff(T))
    print('cad',texp)
    #looking up Stellar parameters from the TIC on MAST
    qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    qld_a=qld[0]
    qld_b=qld[1]

    #physical constants
    RS = 6.955*10.0**10.0 #cm, solar radius
    MS = 1.989*10.0**33.0 #grams, solar mass
    RE = 6.378*10.0**8.0 #cm, earth radius
    
    # arbitrarily choosing for transits to start at 20th data point
    # for 30 minute cadence, this is about 10 hrs after observations start
    
    time_start = T0 #remember Python uses zero indexing (0=first point) 
        
    SMA, SMA_cm = SMA_AU_from_Period_to_stellar(Period,R_star,M_star)
    
    
    #for transit injection
    import batman

    # Using Batman to inject transits into background subtracted and normalized Flux
    # Note: At this point, Flux is not yet detrended (Simple Aperture Photometry)
    ma = batman.TransitParams()
    ma.t0 = time_start  # time of inferior conjunction; first transit is X days after start
    ma.per = Period  # orbital period
    ma.rp = (R_planet_RE*RE)/(R_star*RS) #in units of stellar radii
    ma.a = SMA  # semi-major axis (in units of stellar radii)
    
    # the following 3 parameters are assumed for a perfect,
    # across the star's face, transit
    ma.inc = 90  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)

    ma.u = [qld_a, qld_b]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
#     print(' ')
#     print('injected params:')
#     print(ma.rp,'planet radius in stellar radii')
#     print(R_planet_RE,'planet radius in Earth radii')
#     print(ma.a,'SMA in stellar radii')
# #     print(SMA,'SMA in AU')
#     print(ma.t0,'transit starting time (TESS JD)')
#     print(ma.per, 'orbital period (days)')
#     print('')
    
    t = np.linspace(np.min(T),np.max(T),len(T))
    
#     m = batman.TransitModel(ma, t,supersample_factor = 7, exp_time=texp)  # initializes model
    m = batman.TransitModel(ma, T, supersample_factor = 7, exp_time=texp)  # initializes model    
    synthetic_signal = m.light_curve(ma)  # calculates light curve

    injectedflux = F+synthetic_signal-1 #adding 1 to make baseline = 1

    inj_LC = pd.DataFrame({'Time':T,'SAP Flux':injectedflux,'SAP Error':E, 'Injected Model':synthetic_signal})
    
    inj_params= pd.DataFrame({'Period':ma.per,'T0':ma.t0,'Planet Radius [RS]':ma.rp,'Injected Radius':R_planet_RE,'SMA [RS]':ma.a},index=[0])
    
    return inj_LC, inj_params



def full_pipeline_injection(ID,cutoutsize,Sector,minimum_photon_counts,threshold,pld_order,n_pca_terms, Nsigma_low, Nsigma_high, remove_outliers, before_after_in_minutes, path, cadence, verbose, Period, R_planet_RE,T0, keep_FITS=True, keep_imagedata=True, window_size_in_days=None,use_SPOC_aperture='no'):    
    from transitleastsquares import catalog_info    
    import sys
    ###
    #first, check if target has known stellar radius and/or mass:
    from transitleastsquares import catalog_info    
    try:
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    except (requests.exceptions.ConnectionError,requests.exceptions.HTTPError) as E:
        clock.sleep(5) #pause 5 seconds then try again
        qld, M_star, M_star_min, M_star_max, R_star, R_star_min, R_star_max = catalog_info(TIC_ID=ID)
    if np.isfinite(R_star)==False or np.isfinite(M_star)==False:
        print('TIC '+str(ID)+' has no known Stellar Mass or Radius in TIC')
        return
    else:
        ###
        ###
        ###
        print('TIC '+str(ID)+' Sector '+str(Sector))
        #Step 0: Creating directories to save figures and data
        if verbose==True:
            print('Step 0: Making Directories')
            print(' ')
        path, savefigpath, savelcpath,downloadpath = Make_dirs_injection(path,Sector,cadence,Period,R_planet_RE)
        ###
        ###
        ###
        #Step 1: Obtaining HDU for FFI/TPF
        if verbose==True:
            print('Step 1: Obtaining HDU from FFI/TPF')
            print(' ')
        try:
            hdu,CCD,Camera,quality_mask,reference_pixel = gethdu(ID,Sector,cutoutsize,cadence,minimum_photon_counts,verbose,downloadpath)
            ###
            ###        
            print(' ')
            if hdu==None:
                #print('No Image data for TIC '+str(ID)+' in Sector '+ str(Sector)+'!!!')
                sys.exit('No Image data for TIC '+str(ID)+' in Sector '+ str(Sector)+'!!!') 
        except AttributeError as AE:
            print(AE)
            sys.exit('No Image data for TIC '+str(ID)+' in Sector '+ str(Sector)+'!!!') 
        ###
        ###
        ###
        ###
        if verbose==True:
            print('Step 2: Performing Background Subtraction and Simple Aperture Photometry')
            print(' ')
        try:
            bkg_mask, pix_mask ,flux, median_image, SAP_LC, flux_contamination_ratio = SAP(ID,Sector,cutoutsize,hdu,quality_mask,threshold,cadence,reference_pixel,verbose,savelcpath,use_SPOC_aperture='no')
            ###
            ###
            ###
        except TypeError as TE:
            print(TE)
            print('Unable to create aperture mask. Skipping this target...')
            return 
        ###
        ###
        ###
        if len(SAP_LC['SAP Error'])==0:
            print(' ')
            print('Uneven array lengths, FFI likely on edge of detector/partially shown')
            return
        ###
        ### Transit Injection begins
        mdumps,t_0,t_1 = momentumdump_check(Sector)
        if T0==None:
            #random pt between beginning of sector and end of 1st orbit
            T0 = np.random.uniform(low=t_0,high=t_1) 
        else:            
            T0=t_0+1 #1 day after start of sector, fixed first transit time
        input_LC = SAP_LC
        inj_LC, inj_params = transit_injection(input_LC,Period, T0, R_planet_RE, ID, Sector)
        #save raw injected LC
        inj_LC.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_TransitInjected_RAWLC.txt",index=False)
        #save params
        inj_params.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_TransitInjected_param.txt",\
                          index=False)        
        SAP_LC['SAP Flux'] = inj_LC['SAP Flux']
        SAP_LC['SAP Error']= inj_LC['SAP Error']        
        print('Injecting Transit: P= '+str(Period)+'d; RP= '+str(R_planet_RE)+' T0: '+str(T0))
        ### Transit Injection finished
        ###
        if verbose==True:
            print('Step 3: Removing Momentum dumps and regions of high jitter / Earth-Moon glare')
            print(' ')
        mask_mdump, mdumps,t_0,t_1, flux, RAWLC_df, clippedRAWLC_df = Applying_Mdump_removal(ID,Sector,Camera,CCD,before_after_in_minutes,SAP_LC,flux,savelcpath,verbose)
        ###
        ###
        ### saving pixel and background masks and image fluxes
        saveNDarr(pix_mask,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_pix_mask")
        saveNDarr(bkg_mask,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bkg_mask")
        saveNDarr(flux,savelcpath,"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_image_fluxes")    
        ###
        ### calculating centroid positions throughout images and resaving to file
        cxs,cys = check_centroids(ID,Sector,cutoutsize,cadence,reference_pixel,savelcpath)
        time = np.array(clippedRAWLC_df['Time'])
        sap_flux=np.array(clippedRAWLC_df['SAP Flux'])
        sap_error=np.array(clippedRAWLC_df['SAP Error'])
        bkg_flux=np.array(clippedRAWLC_df['Background Flux'])
        clippedRAWLC_df = pd.DataFrame({"Time":time, "SAP Flux": sap_flux, "SAP Error":sap_error,"Background Flux":bkg_flux, "Centroid X Positions":cxs,"Centroid Y Positions":cys})
        clippedRAWLC_df.to_csv(savelcpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_RAW_LC_systematics_removed.txt",index=False)
        ###
        if verbose==True:
            print('Step 4: Performing Pixel Level Decorrelation modeling')
            print(' ')
        ###
        ## work on making this flexible to take either PLD or SAP
        input_LC = pd.DataFrame({'Time':np.array(clippedRAWLC_df['Time']),\
                         'Flux':np.array(clippedRAWLC_df['SAP Flux']),\
                        'Error':np.array(clippedRAWLC_df['SAP Error']),\
                                 "Centroid X Positions":np.array(clippedRAWLC_df["Centroid X Positions"]),\
                                "Centroid Y Positions":np.array(clippedRAWLC_df["Centroid Y Positions"])})

        PLD_LC = PLD_model(ID,Sector,flux,pix_mask,input_LC,savelcpath,pld_order=pld_order, n_pca_terms=n_pca_terms)
        ###
        ###
        ###
        if verbose==True:
            print('Step 5: Applying smoothing filter')
            print(' ')    
        print('len check for step 5:')
        print('PLD T',len(np.array(PLD_LC['Time'])),'PLD F',len(np.array(PLD_LC['PLD Flux'])),'PLD E',len(np.array(PLD_LC['PLD Error'])))
        ## work on making this flexible to take either DET or SAP
        input_LC2 = pd.DataFrame({'Time':np.array(PLD_LC['Time']),\
                                 'Flux':np.array(PLD_LC['PLD Flux']),\
                                'Error':np.array(PLD_LC['PLD Error']),\
                                  'Model':np.array(PLD_LC['PLD Model']),\
                                  'SAP Flux':np.array(input_LC['Flux']),\
                                  'SAP Error':np.array(input_LC['Error']),\
                                 "Centroid X Positions":np.array(PLD_LC["Centroid X Positions"]),\
                                "Centroid Y Positions":np.array(PLD_LC["Centroid Y Positions"])})
        Det_LC, nanmask = BWMC_auto(ID,Sector,input_LC2,savelcpath)           
        ###
        ### ensure PLD outputs and Detrended outputs have same length using nanmask output
        print('2nd len check for step 5: ')
    #     print('T: ',len(Det_LC['Time']),'SAP F: ',len(PLD_LC['SAP Flux']), 'SAP E: ',len(PLD_LC['SAP Error'])   ,\
    #           ' PLD F: ', len(PLD_LC['PLD Flux']),' PLD E: ', len(PLD_LC['PLD Error']),\
    #                     ' Det F: ',len(Det_LC['Detrended Flux']), 'Det E: ',len(Det_LC['Detrended Error']))    
    #     PLD_LC = pd.DataFrame({'Time':np.array(Det_LC['Time']),'SAP Flux':(PLD_LC['SAP Flux'])[nanmask],\
    #                            'SAP Error':(PLD_LC['SAP Error'])[nanmask],'PLD Flux':np.array(PLD_LC['PLD Flux'])[nanmask],\
    #                            'PLD Error':np.array(PLD_LC['PLD Error'])[nanmask],\
    #                           'PLD Model':np.array(PLD_LC['PLD Model'])[nanmask],\
    #                          'Centroid X Positions':np.array(PLD_LC['Centroid X Positions']),\
    #                           'Centroid Y Positions':np.array(PLD_LC['Centroid Y Positions'])})
        print('T: ',len(Det_LC['Time']),'SAP F: ',len(Det_LC['SAP Flux']), 'SAP E: ',len(Det_LC['SAP Error']),' PLD F: ', len(Det_LC['PLD Flux']),' PLD E: ', len(Det_LC['PLD Error']),' PLD M: ', len(Det_LC['PLD Model']),' Det F: ',len(Det_LC['Detrended Flux']), 'Det E: ',len(Det_LC['Detrended Error']),' Det M: ',len(Det_LC['Fitted Trend']))   
        ###
        ###
        ###
        if verbose==True:
            print('Step 6: Applying Outlier Removal (if set to "yes")')
            print(' ')
            ###
            #this needs to be the MOST flexible part to deal with combos of PLD, DET and SAP
        print('len check for step 6:')
        print('T: ',len(Det_LC['Time']),'SAP F: ',len(Det_LC['SAP Flux']), 'SAP E: ',len(Det_LC['SAP Error']),' PLD F: ', len(Det_LC['PLD Flux']),' PLD E: ', len(Det_LC['PLD Error']),' PLD M: ', len(Det_LC['PLD Model']),' Det F: ',len(Det_LC['Detrended Flux']), 'Det E: ',len(Det_LC['Detrended Error']),' Det M: ',len(Det_LC['Fitted Trend']))   
        input_LC3 = pd.DataFrame({'Time':np.array(Det_LC['Time']),\
                                  'SAP Flux':np.array(Det_LC['SAP Flux']),\
                                  'SAP Error':np.array(Det_LC['SAP Error']),\
                                 'Detrended Flux':np.array(Det_LC['Detrended Flux']),\
                                'Detrended Error':np.array(Det_LC['Detrended Error']),\
                                  'Fitted Trend':np.array(Det_LC['Fitted Trend']),\
                                 'PLD Flux':np.array(Det_LC['PLD Flux']),\
                                 'PLD Error':np.array(Det_LC['PLD Error']),\
                                 'PLD Model':np.array(Det_LC['PLD Model']),\
                                 "Centroid X Positions":np.array(Det_LC["Centroid X Positions"]),\
                                  "Centroid Y Positions":np.array(Det_LC["Centroid Y Positions"])})
        ###
        ###
        LC_df, good_ind_DF, bad_ind_DF, preclipLC_df = outlier_removal(ID,Sector,input_LC3, remove_outliers, Nsigma_low,Nsigma_high,savelcpath,verbose,window_size_in_days=window_size_in_days)       
        ###
        ###
        ###
        ###
        ###
        ###
        if verbose==True:
            print('Step 7: Plotting and Saving FFI and selected apertures')
            print(' ')
        plot_it_all_up(ID,Sector,cutoutsize,cadence,Nsigma_low,Nsigma_high,\
                       hdu,median_image,pix_mask,bkg_mask, RAWLC_df, \
                       clippedRAWLC_df, LC_df, good_ind_DF, bad_ind_DF, preclipLC_df, \
                       magnitude_limit=18,dot_scale=20,path=path,downloadpath=downloadpath)
        ###
        ###
        if keep_FITS==False:
            # deleting FITS files (no longer need them for light curve processing
            # can always download again)
            os.system("rm -r " + downloadpath) #delete cache path
        ###
        ###
        print('FINAL LENGTHS :', ' T', len(LC_df['Time']),' Det F',len(LC_df['Detrended Flux']), ' trend',len(LC_df['Fitted Trend']), 'PLD model',len(LC_df['PLD Model']),' PLD F',len(LC_df['PLD Flux']),' SAP E', len(LC_df['SAP Error']))

        


def TLS_Report_injection(ID,Sector,cadence,path,keep_FITS=False,keep_imagedata=True, for_injections=True):
    ### ONLY DIFFERENCE IS TURNING OFF DSS PLOT
    ###
    # making a 1 page validation report summarizing the overall transit search    
    ###
    if for_injections==False:
        Path=path+'Sector_'+str(Sector)+'/'
    if for_injections==True:
        Path=path
    if cadence=='long':
        saveReportpath = Path+'FFI_TLS_Report/'
        savelcpath= Path+'FFI_PLD_LCs/'
        downloadpath = Path+'cache/'
    if cadence=='short':            
        saveReportpath = Path+'TPF_TLS_Report/'
        savelcpath= Path+'TPF_PLD_LCs/'
        downloadpath = Path+'cache/'
    ###        
    #creating directory if it already doesn't exist
    if os.path.exists(saveReportpath)==True:
        pass
    else:
        os.makedirs(saveReportpath)
    ###
    #in case it was deleted by a previous run:
    if os.path.exists(downloadpath)==True:
        pass
    else:
        os.makedirs(downloadpath)
    ###
    ###
    #files needed to compile results
    # light curves and centroids
    LC_df = pd.read_csv(savelcpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_final_LC.txt')
    ###
    # TLS results
    TLS_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS.txt')
    TLSmodel_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS_model.txt')
    TLSbestfit_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS_bestfit.txt')
    TLSTCs_df = pd.read_csv(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLS_TCs.txt')
    TLStxt_array=list(TLSbestfit_df.columns.values)
    ###
    # EDI-Vetter Results for False Positive Flags
    EDI_results = pd.read_csv(saveReportpath+"TIC_"+str(ID)+"_Sector_"+str(Sector)+"_EDI_results.txt",index_col=0)
    ###
    #reading in a lot of dataframes can take up RAM, this can help clear it with gc.collect()
    import gc 
    #reading in a lot of dataframes can take up RAM, this can help clear it with gc.collect()
    ###
    import time as clock
    start=clock.time()
    ###
    #fontsize
    fs=12
    ###
    fig = plt.figure(figsize=(12,10))
    gs1 = gridspec.GridSpec(6, 3)#4,2
    #3x3 grid, (left,middle,right) helps makes sense of placement. gs1[height, width]
    left=0
    middle=1
    right=2
    ###    
    #Odd/Even plot
    ax0 = fig.add_subplot(gs1[0:2,left])
    plot_odd_even_transits(LC_df, TLSbestfit_df, TLSTCs_df, TLSmodel_df, ax0,fig)
    ###    
    #Lightcurve
    ax1 = fig.add_subplot(gs1[0, middle:])
    ###    
    #Power Spectrum
    ax2 = fig.add_subplot(gs1[1, middle:])
    plot_power_spectra(TLS_df,TLSbestfit_df, ax2)
    ###
    #Phasefolded Light Curves
    ax3 = fig.add_subplot(gs1[2, middle])
    ax4 = fig.add_subplot(gs1[2, right])
    ax5 = fig.add_subplot(gs1[3, middle])
    ax6 = fig.add_subplot(gs1[3, right])
    plot_phasefold_LCs(ID,Sector,LC_df,TLS_df,TLSbestfit_df,TLSTCs_df,TLSmodel_df, ax1,ax3,ax4,ax5,ax6)
    ###    
    # TESS Image Cutout
    ax7_placement=gs1[2:4,left]
    plot_image(ID,Sector,cadence,path,ax7_placement,fig,fs,for_injections=for_injections)
    ###    
    #Centroid Motion plot
    ax8 = fig.add_subplot(gs1[4:6,left])
    plot_centroids_in_phase(LC_df,TLSbestfit_df,ax8)
    ###
    # DSS Image
    ax9 = fig.add_subplot(gs1[4:6, middle])
    plot_dss_orientation(ax9,ID,downloadpath,cutoutsize=11,do_DSS_plot=False) #<---THE ONLY DIFFERENCE
    ###    
    #plot text from stellar and planet parameters
    plot_text(ID,Sector,TLSbestfit_df,TLSTCs_df,EDI_results,fig)
    ###
    gs1.update(wspace=0.0, hspace=0.0)
    gs1.tight_layout(fig)
    fig.savefig(saveReportpath+'TIC_'+str(ID)+'_Sector_'+str(Sector)+'_TLSReport.png',bbox_inches='tight')
#     fig.show()
    plt.close()
    ###
    end=clock.time()

    runtime=end-start

    # clear garbage collection in RAM
    gc.collect()

    if runtime > 60:
        print('report runtime: '+str(runtime/60)+' minutes')
    if runtime < 60:
        print('report runtime: '+str(runtime)+' seconds')    
        
    ###
    ### Last Step: Clear image data (FITS and PKL files) to save space
    ### (optional)
    ###            
    #delete image data(aperture/background masks, cutout images)
    if keep_imagedata==False:
        pixmask_filename="TIC_"+str(ID)+"_Sector_"+str(Sector)+"_pix_mask"
        bkgmask_filename="TIC_"+str(ID)+"_Sector_"+str(Sector)+"_bkg_mask"
        image_filename="TIC_"+str(ID)+"_Sector_"+str(Sector)+"_image_fluxes"                
        os.system("rm " + savelcpath+pixmask_filename+'.pkl') 
        os.system("rm " + savelcpath+bkgmask_filename+'.pkl') 
        os.system("rm " + savelcpath+image_filename+'.pkl')    
    if keep_FITS==False:
            # deleting FITS files (no longer need them for light curve processing
            # can always download again)
            os.system("rm -r " + downloadpath) #delete cache path
            
            
def Transit_Pipeline_injection(threshold, ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path,keep_FITS=True,keep_imagedata=True, for_injections=True):
    
    ### first do BLS
    print('')
    print('doing BLS search')
    PowerSpectrum_df, TransitModel_df, TransitParams_df = TransitSearch('BLS',ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path, for_injections)
    ###
    if isinstance(PowerSpectrum_df, type(None)):
        print('problem with infinite/nan values inPower Spectrum')
        return None, None, None, None, None, None
    #make BLS Plot
    Transit_plot(ID,Sector,cadence,'BLS',input_LC, PowerSpectrum_df,TransitModel_df, TransitParams_df, path, for_injections)
    ###
    ### check if SDE is above threshold
    if (np.nanmax(np.array(PowerSpectrum_df['BLS SDE']))) > threshold:
        print('')
        print('BLS max peak > threshold!')
        print('')
        PowerSpectrum_df2, TransitModel_df2, TransitParams_df2 = TransitSearch('TLS',ID,Sector,cadence,input_LC,N_transits,minP,oversampling_factor,duration_grid_step,path, for_injections)
        if isinstance(PowerSpectrum_df2, type(None)):
            print('problem with infinite/nan values inPower Spectrum')
            return None, None, None, None, None, None        
        else:
            ###
            #make TLS Plot
            Transit_plot(ID,Sector,cadence,'TLS',input_LC, PowerSpectrum_df2,TransitModel_df2, TransitParams_df2, path, for_injections)
            ###
            # make TLS Report
            TLS_Report_injection(ID,Sector,cadence,path,keep_FITS=keep_FITS,keep_imagedata=keep_imagedata, for_injections=for_injections)
        
        
        return PowerSpectrum_df, TransitModel_df, TransitParams_df, PowerSpectrum_df2, TransitModel_df2, TransitParams_df2
    else:
        print('BLS signal < threshold')
        return PowerSpectrum_df, TransitModel_df, TransitParams_df, None,None,None            

##########################################
##########################################
####### TRANSIT INJECTION FUNCTIONS ######
##########################################
##########################################





#########################################################
#########################################################
################### Change Log ##########################
#########################################################
#########################################################

# ## Change log for V13 (Now called NEMESIS V1.1) : April 22nd, 2021 (updating after lightkurve 2.0 came out)
# ## - Updated gethdu to better handle 504 Gate TimeOuts from MAST. 
# ##   Now uses a while loop until it downloads properly.

# ## Change log for V12 (Now called NEMESIS V1.0) : January 8th, 2021 (After receiving Referee's report from AJ)
# ## - Added flux contamination (deblending) correction function to pipeline: 
# ##   Will query TIC for nearby stars within 63 arcseconds and find flux contam ratio
# ##   Then it will subtract ratio from normalized SAP and then renormalized before 
# ##   being passed on to rest of pipeline.
# ## - Added aesthetic changes to clean up light curve summary figures
# ##   (used to be "*_PLD.png", is now "*_LC_summary.png"), TLS validation 
# ##   reports, and _________
# ## - Added Binning function based on inverse variance weighting
# ##   (Also used in AstroImageJ, courtesty of Dr. Karen Collins).
# ##   This isn't used anywhere but is very useful! 
# ## - Added injection pipeline which used to be a separate script.


# ## Change log for V11: August 17th, 2020
# ## - Added BLS functionality, same inputs and outputs as TLS (except FAP will be NaN)
# ## - Added Transit_Pipeline that does BLS first and if BLS peak > SDE threshold, 
# ##   it will then run TLS. Only TLS gets full reports. May add BLS reports in future...
# ## - After some testing, decided to remove positive consecutive outliers. 
# ##   May affect transit searches.
# ## - Added a function to center FFI cutouts and rewrite WCS headers based on 
# ##   centered pixel coordinates.
# ## - Added function to plot either TESS or GAIA sources for nearby stars.
# ## - Condensed FFI/TPF plotting instances to use same command: plot_cutouts
# ## - Went back to original pipeline order: 1)Get Images, 2) SAP, 3) PLD, 
# ##   4) Smoothing, 5) Outlier Removal, 6) save light curves and plots


# ## Change log for V10: June 30th,2020
# ## - Revisted outlier removal and added option for a sliding outlier remover
# ##   that keeps consecutive data points (like transits) which requires the 
# ##   inputs for method ("sliding"), window, number of standard deviations
# ##   and whether to use global noise of full light curve or just noise within
# ##   sliding window ("global", "local")

# ## Change log for V9: June 4th, 2020
# ## - Rewritten and condensed functions for easier understanding of each 
# ##   step in pipeline within the "PLD_FFI" function. 
# ## - Modified outlier removal to include data points below/above a user
# ##   defined threshold that are consecutive (like transits/flares). 

# ## Change log for V8: May 12, 2020 
# ## - Added features to plot other TESS stars in FFI based on TIC RA/DEC, 
# ##   and now shows sky background
# ## - Changed order of operations in light curve extraction:
# ## - Steps are now, 1) get FFI 2) make pixel mask 3) do background subtraction
# ##   4) Remove momentum dumps 5) Detrend 6) PLD 7) save lightcurves 8) plot
# ## - Tweaked momentumdump removal to better clip out bad regions of data
# ## - momentumdump_removal now returns a boolean mask instead of time,flux,error
# ##   arrays
# ## - Light curves before removal of systematics/detrending are now saved as 
# ##   "TIC_IDNum_Sector_SectorNum_RAW_LC.txt" in PLD_LCs directory
# ## - Provided user option to turn on/off outlier removal with "yes"/"no" input
# ## - Tweaked outlier removal to use a sliding sigma clipper using a 3 hr window
# ##   and removing data points within 3 std above and 7 std below the median flux 
# ## - Provided user option to select window of data to remove around momentum dumps
# ##   "before_after_in_minutes"
# ## - Added "smooth_window" as input for PLD so user can change window size for
# ##   smoothing the flux in units of hours. 
# ## - Now instead of having Jupyter Notebooks with definitions, I am now using
# ##   separate Python scripts to import custom functions. I should look into 
# ##   using Classes for definitions...



# ## Change log for V7: 
# ## Added capability to use more than 2nd order PLD. Can do any order user desires.
# ## Added sigma_clip for outlier removal (works better). Uses higher sigma for points 
# ## below than above to avoid truncating potential transit depths.

# ## Change log for V6: 
# ## - Can now be used for either 30 minute FFI file or 2 minute TPF file structures. 
# ##   PLD_FFI function now requires cadence input of "long" or "short", 
# ##   similar to lightkurve.
# ## - To obtain TPFs, we still use lightkurve's search_targetpixelfile function. 
# ##   It's just easier than coming up with a query from scratch. 
# ## - Will create FFI and TPF directories for light curve files and plots depending
# ##   on user selected cadence mode.

# ## Change log for V5: 
# ## -Modified aperture selection for target stars. Before, only the brightest pixels in the FFI cutout were used. 
# ##  This meant that neighboring stars that were brighter or similarly bright as the target star were being included 
# ##  in the pixel mask, creating a blended light curve and adding additional steps to manual vetting.
# ## -New aperture selection requires the RA and DEC of the target star. This is done with the "RaandDec_to_XandY" 
# ##  function which converts RA and DEC to pixel X,Y coordinates. With the X,Y pixel coordinates, the new aperture 
# ##  mask function titled "thresholdmask" uses a sigma threshold level to determine the closest and brightest pixels 
# ##  to the target star.

# ## Change log for V4:
# ## - Modified outlier removal to make Nsigma a variable instead of 7 (still set to 7, may change in future versions)
# ## - Fixed calculated errors. Previous versions had HUGE errors (+/- 3 flux units). Currently uniform errors based 
# ##   on PLD detrended flux values ~ +/- 0.005 (more reasonable).
# ## - Working Directory path is now an input to PLD_FFI so pathnames aren't hardcoded. Just run this script from 
# ##   where ever you want to place files.
# ## - Added more rigorous ID matching when using Astroquery to avoid accidentally matching to nearby bright stars. 
# ##   Also added a 21 arcsec radial cone search (TESS' pixel scale) instead of the 30 arcsec angular size used before.
# ## - Edited background pixel mask to ignore the brightest pixel(s) in the FFI cutout, variable name: "
# ##   except_these_pixels"
# ## - Updated PLD portion to accept situations where there are more than one solution for solving a*W=b. 
# ##   If more than one solution, Numpy's least squares solver will be used and first solution is selected.


# ## Change log for V3:
# ## - modified order of systematic removal, detrending and outlier removal 
# ## - Outlier removal now does sweeps looking applying an iterative 5-sigma cut and smoothing procedure followed 
# ##   by a final cut which uses 7 times the rms of the detrended light curve.
# ## - Modified background subtraction to ignore 4 brightest pixels in image cutout instead of before where we 
# ##   selected 100 brightest pixels. For small cutout sizes (<10), this would select all pixels as the background.

# 
# ## Change log for V2:
# ## - tweaked plotting of FFI image to combine with PLD/SAP light curve comparison 
# ## - Added TLS (with momentum dump markers)
# ## - Added Plotting script to show each TLS modeled event vertically separated
# 
# ## Change log for V1:
# ## - Introduced Pixel Level Decorrelation (explanation, directory creation, SAP, PLD, plotting, saving 
# ##   light curves/images to directories)
