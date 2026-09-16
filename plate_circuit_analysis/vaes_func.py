"""
CODE WITH FUNCTIONS RELATED TO NEW BOOTSTRAP COMPARISON APPROACHES

Bram Vaes

1 October 2021
"""

# ------------------------
# Import packages
import pmagpy.pmag as pmag
import pmagpy.pmagplotlib as pmagplotlib
import pmagpy.ipmag as ipmag
import pmagpy.contribution_builder as cb
from pmagpy import convert_2_magic as convert
import matplotlib as mat
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np        
import pandas as pd        
import os
from past.utils import old_div
import time
import boot_func as boot_func
from numpy import random
from numpy import genfromtxt
import csv
from statistics import mean
from statistics import median
from statistics import stdev
from random import uniform

# ------------------------

def pseudo_N(DIs, random_seed=None, Nref=None):
    """
    Draw a bootstrap sample of directions returning as many bootstrapped samples
    as in the input directions
    Parameters
    ----------
    DIs : nested list of dec, inc lists (known as a di_block)
    random_seed : set random seed for reproducible number generation (default is None)
    Returns
    -------
    Bootstrap_directions : nested list of dec, inc lists that have been
    bootstrapped resampled
    """
    #print(Nref)
    
    if random_seed != None:
        np.random.seed(random_seed)
    if Nref == None:
        sample_size = len(DIs)
    else:
        sample_size = Nref
    #print(Nref,sample_size)
    Inds = np.random.randint(len(DIs), size=sample_size)
    D = np.array(DIs)
    return D[Inds]

def di_boot(DIs, nb=500, Nref=None):
    """
     returns bootstrap means  for Directional data
     Parameters
     _________________
     DIs : nested list of Dec,Inc pairs
     nb : number of bootstrap pseudosamples
     Returns
    -------
     BDIs:   nested list of bootstrapped mean Dec,Inc pairs
    """
#
# now do bootstrap to collect BDIs  bootstrap means
#
    BDIs = []  # number of bootstraps, list of bootstrap directions
#

    for k in range(nb):  # repeat nb times
        #        if k%50==0:print k,' out of ',nb
        if Nref==None:
            pDIs = pseudo_N(DIs)
        else:
            pDIs = pseudo_N(DIs,Nref=Nref)  # get a pseudosample
        # plt.figure(num=1,figsize=(8,8))
        # ipmag.plot_net(1)   # plot equal area projection
        # ipmag.plot_di(di_block=VGPs,color='k',edge='k',marker='o',markersize=80)
        # ipmag.plot_di(di_block=pDIs,color='red',edge='k',marker='D',markersize=50)
        # plt.show()
        bfpars = pmag.fisher_mean(pDIs)  # get bootstrap mean bootstrap sample
        BDIs.append([bfpars['dec'], bfpars['inc']])
    return BDIs

def par_boot(DIs,kappas,nns,nb=500,Nref=None):
    """
     returns bootstrap means for Directional data,
     calculated using a parametric bootstrap (Tauxe, 2010)

     Parameters
     _________________
     DIs : nested list of Dec,Inc pairs
     nb : number of bootstrap pseudosamples
     Returns
    -------
     BDIs:   nested list of bootstrapped mean Dec,Inc pairs
    """
    #
# now do bootstrap to collect BDIs  bootstrap means
#
    BDIs = []  # number of bootstraps, list of bootstrap directions
    inds = []
#
    for k in range(nb):  # repeat nb times
        sample_size=len(DIs) # determine size of pseudosample
        pDIs=[]
        for i in range(sample_size): 
            ind = np.random.randint(sample_size, size=1) # create random index
            j = ind[0]

            pseudo_dirs = ipmag.fishrot(k=kappas[j],n=nns[j],dec=DIs[j][0],inc=DIs[j][1])
            pseudo_site = ipmag.fisher_mean(di_block=pseudo_dirs)
            pDI = [pseudo_site['dec'],pseudo_site['inc']]
            
            #print(j,DIs[j],kappas[j],nns[j])
            #print(pDI,pseudo_site['k'],pseudo_site['n'])

            pDIs.append(pDI)
            inds.append(j)
    
        bfpars = pmag.fisher_mean(pDIs)  # get bootstrap mean bootstrap sample
        BDIs.append([bfpars['dec'], bfpars['inc']])

        if k % 10 == 0:
            print(k)

    #print(len(inds),sample_size)

    # plt.figure(num=1,figsize=(8,8),dpi=100)
    # plt.hist(inds,bins=25,range=(0,2500),rwidth=0.9,density=False,color='steelblue')
    # plt.show()
    return BDIs

def common_mean_bootstrap(Data1, Data2, nb=500, para_boot=None, Nref=None, dir_df=None, save=False, save_folder='.', fmt='svg', figsize=(7, 2.3), x_tick_bins=4):
    """
    Conducts a bootstrap test (Tauxe, 2010) for a common mean on two declination,
    inclination data sets. Plots are generated of the cumulative distributions
    of the Cartesian coordinates of the means of the pseudo-samples (one for x,
    one for y and one for z). If the 95 percent confidence bounds for each
    component overlap, the two directions are not significantly different.
    Parameters
    ----------
    Data1 : a nested list of directional data [dec,inc] (a di_block)
    Data2 : a nested list of directional data [dec,inc] (a di_block)
            if Data2 is length of 1, treat as single direction
    NumSims : number of bootstrap samples (default is 1000)
    save : optional save of plots (default is False)
    save_folder : path to directory where plots should be saved
    fmt : format of figures to be saved (default is 'svg')
    figsize : optionally adjust figure size (default is (7, 2.3))
    x_tick_bins : because they occasionally overlap depending on the data, this
        argument allows you adjust number of tick marks on the x axis of graphs
        (default is 4)
    Returns
    -------
    three plots : cumulative distributions of the X, Y, Z of bootstrapped means
    Examples
    --------
    Develop two populations of directions using ``ipmag.fishrot``. Use the
    function to determine if they share a common mean (through visual inspection
    of resulting plots).
    >>> directions_A = ipmag.fishrot(k=20, n=30, dec=40, inc=60)
    >>> directions_B = ipmag.fishrot(k=35, n=25, dec=42, inc=57)
    >>> ipmag.common_mean_bootstrap(directions_A, directions_B)
    """
    counter = 0
    BDI1 = di_boot(Data1,nb=nb,Nref=Nref)
    cart1 = pmag.dir2cart(BDI1).transpose()
    X1, Y1, Z1 = cart1[0], cart1[1], cart1[2]
    if np.array(Data2).shape[0] > 2:
        if para_boot==None:
            BDI2 = di_boot(Data2,nb=nb)
        else:
            print('Initialize parametric bootstrap..')
            if dir_df==None:
                BDI2 = par_boot(Data2,nb=nb) # NB: additional input is required here
            else:
                BDI2 = pmag.dir_df_boot(dir_df=dir_df,nb=nb,par=para_boot)
            print('Succes!')
        cart2 = pmag.dir2cart(BDI2).transpose()
        X2, Y2, Z2 = cart2[0], cart2[1], cart2[2]
    else:
        cart = pmag.dir2cart(Data2).transpose()

    fignum = 1
    fig = plt.figure(figsize=figsize)
    #fig = plt.subplot(1, 3, 1)
    plt.subplot(1, 3, 1)

    minimum = int(0.025 * len(X1))
    maximum = int(0.975 * len(X1))

    X1, y = pmagplotlib.plot_cdf(fignum, X1, "X component", 'r', "")
    bounds1 = [X1[minimum], X1[maximum]]
    pmagplotlib.plot_vs(fignum, bounds1, 'r', '-')
    if np.array(Data2).shape[0] > 2:
        X2, y = pmagplotlib.plot_cdf(fignum, X2, "X component", 'b', "")
        bounds2 = [X2[minimum], X2[maximum]]
        pmagplotlib.plot_vs(fignum, bounds2, 'b', '--')
    else:
        pmagplotlib.plot_vs(fignum, [cart[0]], 'k', '--')
    plt.ylim(0, 1)
    plt.locator_params(nbins=x_tick_bins)

    plt.subplot(1, 3, 2)

    Y1, y = pmagplotlib.plot_cdf(fignum, Y1, "Y component", 'r', "")
    bounds1 = [Y1[minimum], Y1[maximum]]
    pmagplotlib.plot_vs(fignum, bounds1, 'r', '-')
    if np.array(Data2).shape[0] > 2:
        Y2, y = pmagplotlib.plot_cdf(fignum, Y2, "Y component", 'b', "")
        bounds2 = [Y2[minimum], Y2[maximum]]
        pmagplotlib.plot_vs(fignum, bounds2, 'b', '--')
    else:
        pmagplotlib.plot_vs(fignum, [cart[1]], 'k', '--')
    plt.ylim(0, 1)

    plt.subplot(1, 3, 3)

    Z1, y = pmagplotlib.plot_cdf(fignum, Z1, "Z component", 'r', "")
    bounds1 = [Z1[minimum], Z1[maximum]]
    pmagplotlib.plot_vs(fignum, bounds1, 'r', '-')

    if np.array(Data2).shape[0] > 2:
        Z2, y = pmagplotlib.plot_cdf(fignum, Z2, "Z component", 'b', "")
        bounds2 = [Z2[minimum], Z2[maximum]]
        pmagplotlib.plot_vs(fignum, bounds2, 'b', '--')
    else:
        pmagplotlib.plot_vs(fignum, [cart[2]], 'k', '--')
    plt.ylim(0, 1)
    plt.locator_params(nbins=x_tick_bins)

    plt.tight_layout()
    if save == True:
        plt.savefig(os.path.join(
            save_folder, 'common_mean_bootstrap') + '.' + fmt)
    plt.show()  

    return BDI1,BDI2    

def max_GCD_test(ref_data, test_data, nb=500, fig=False, boot_test=False):
    """
    Conducts a maximum GCD test

    Options:
    - To compute a bootstrapped 95% confidence circle: set boot_test to True
    - To plot the result: set fig to True
    """
    N_ref = len(ref_data) # store number of VGPs in reference dataset
    N_test = len(test_data) # store number of VGPs in test dataset
    ppoles,ppoles_A95=[],[]
    
    # GENERATE nb PSEUDOPOLES FROM REFERENCE DATASET
    for i in range(nb):
        # SELECT RANDOM VGPS FROM DATASET
        Inds = np.random.randint(N_ref, size=N_test)
        #print(Inds)
        D = np.array(ref_data)
        sample = D[Inds]

        # COMPUTE PSEUDOPOLES
        polepars = pmag.fisher_mean(sample)
        ppoles.append([polepars['dec'], polepars['inc']])
        ppoles_A95.append(polepars['alpha95'])
        #print(synth_poles)

    # COMPUTE REFERENCE MEAN
    VGP_mean = pmag.fisher_mean(ref_data) # compute mean of VGPs
    ref_mean = [VGP_mean['dec'],VGP_mean['inc']] # store reference mean

    # COMPUTE ANGULAR DISTANCE OF PSEUDOPOLES TO REFERENCE MEAN
    D_ppoles=[]
    for j in range(nb):
        ang_distance=pmag.angle(ppoles[j],ref_mean)
        D_ppoles.append(ang_distance[0])

    # COMPUTE STATISTICS
    D_ppoles_mean=mean(D_ppoles)
    D_ppoles_median=median(D_ppoles)
    
    D_ppoles.sort()
    ind_95perc=int(0.95*nb)
    B95 = D_ppoles[ind_95perc]
    print('B95=',B95)

    A95_median=median(ppoles_A95)
    print('Median A95=',A95_median)

    # COMPUTE TEST POLE
    test_mean = pmag.fisher_mean(test_data) # compute mean of VGPs
    test_pole = [test_mean['dec'],test_mean['inc']] # store reference mean
    GCD = pmag.angle(test_pole,ref_mean) 

    # BOOTSTRAP TEST DATA
    if boot_test==True:
        boot_test_data = di_boot(test_data,nb)
        boot_test_mean = pmag.fisher_mean(boot_test_data)

        D_tpoles=[]
        for j in range(nb):
            angle=pmag.angle(boot_test_data[j],[boot_test_mean['dec'],boot_test_mean['inc']])
            D_tpoles.append(angle[0])

        D_tpoles.sort()
        ind_test_95perc=int(0.95*nb)
        B95_test=D_tpoles[ind_test_95perc]

    # PLOT RESULTS
    if fig==True:
        fignum = 1
        #plt.figure(num=fignum,figsize=(8,8))
        ipmag.plot_net(fignum)   # plot equal area projection
        #ipmag.plot_di(di_block=ref_data,color='k',edge='k',marker='o',markersize=6,label='Reference VGPs (N=%3d)' % N_ref)
        ipmag.plot_di(di_block=ppoles,color='k',edge='k',marker='o',markersize=6,label='Pseudopoles (N=%3d)' % nb)
        boot_func.plot_di_mean(dec=ref_mean[0],inc=ref_mean[1],a95=B95,color='k',edge='k',marker='*',markersize=450,label='Reference')
        if boot_test==True:
            ipmag.plot_di(di_block=boot_test_data,color='cyan',edge='k',marker='o',markersize=15,label='Study VGPs (N=%3d)' % N_test)
            boot_func.plot_di_mean(dec=boot_test_mean['dec'],inc=boot_test_mean['inc'],a95=B95_test,color='blue',edge='k',marker='*',markersize=450)
        ipmag.plot_di(di_block=test_data,color='red',edge='k',marker='o',markersize=40,label='Study VGPs (N=%3d)' % N_test)
        boot_func.plot_di_mean(dec=test_pole[0],inc=test_pole[1],a95=test_mean['alpha95'],color='red',edge='k',marker='*',markersize=450,label='Study pole')
        plt.legend(loc=1, fontsize = 16) 
        plt.show()    

        fignum = 2
        plt.figure(num=fignum,figsize=(12,8),dpi=100)
        plt.axvline(B95,color='red',linestyle='--',label='B95')
        plt.axvline(GCD,color='blue',linestyle='--',label='Angular distance of paleopole')
        plt.errorbar(D_ppoles,range(nb),yerr=None,xerr=None,color='k',markeredgecolor='k',linewidth=1,linestyle='-',markersize=0)
        plt.xlabel('Angular distance to reference ($^\circ$)',fontsize=18,labelpad=10)
        plt.xlim([0,D_ppoles[nb-1]])
        plt.ylim([0,nb+100])
        plt.ylabel('Cumulative number of pseudopoles',fontsize=18,labelpad=10)
        plt.legend(loc=4, fontsize = 12) 
        plt.show()
    else:
        print('No plot')

    if boot_test==True:
        return B95, B95_test, A95_median
    else:
        return B95, A95_median

def par_GCD_test(ref_poles, ref_kappas, ref_ns, test_data, N_pp=None, nb=500, fig=False, boot_test=False, filename=False):
    """
    Conducts a maximum GCD test using a reference dataset of parametrically sampled VGPs
    """
    N_poles = len(ref_poles) # store number of VGPs in reference dataset
    if len(test_data)>1:
        N_test = len(test_data) # store number of VGPs in test dataset
    else:
        N_test = N_pp
    
    # GENERATE nb PSEUDOPOLES FROM REFERENCE DATASET
    if filename==False:
        ppoles = [boot_func.gen_pseudopoles(ref_poles, ref_kappas, ref_ns, N_poles, N_test) for i in range(nb)]
    else:
        sim_VGPs = genfromtxt(filename, delimiter=',')
        ppoles = [boot_func.gen_pseudopoles_file(sim_VGPs,N_test) for i in range (nb)]

    # COMPUTE REFERENCE MEAN
    VGP_mean = pmag.fisher_mean(ppoles) # compute mean of VGPs
    ref_mean = [VGP_mean['dec'],VGP_mean['inc']] # store reference mean

    # COMPUTE ANGULAR DISTANCE OF PSEUDOPOLES TO REFERENCE MEAN
    D_ppoles=[]
    for j in range(nb):
        ang_distance=pmag.angle(ppoles[j],ref_mean)
        D_ppoles.append(ang_distance[0])

    # COMPUTE STATISTICS
    D_ppoles.sort()
    ind_95perc=int(0.95*nb)
    B95 = D_ppoles[ind_95perc]

    # COMPUTE TEST POLE
    if len(test_data)>1:
        test_mean = pmag.fisher_mean(test_data) # compute mean of VGPs
        test_pole = [test_mean['dec'],test_mean['inc']] # store reference mean
    else:
        test_pole = test_data

    # BOOTSTRAP TEST DATA
    if boot_test==True:
        boot_test_data = di_boot(test_data,nb)
        boot_test_mean = pmag.fisher_mean(boot_test_data)

        D_tpoles=[]
        for j in range(nb):
            angle=pmag.angle(boot_test_data[j],[boot_test_mean['dec'],boot_test_mean['inc']])
            D_tpoles.append(angle[0])

        D_tpoles.sort()
        ind_test_95perc=int(0.95*nb)
        B95_test=D_tpoles[ind_test_95perc]

    # PLOT RESULTS
    if fig==True:
        fignum = 1
        #plt.figure(num=fignum,figsize=(8,8))
        ipmag.plot_net(fignum)   # plot equal area projection
        #ipmag.plot_di(di_block=ref_data,color='k',edge='k',marker='o',markersize=6,label='Reference VGPs (N=%3d)' % N_ref)
        ipmag.plot_di(di_block=ppoles,color='k',edge='k',marker='o',markersize=6,label='Pseudopoles (N=%3d)' % nb)
        boot_func.plot_di_mean(dec=ref_mean[0],inc=ref_mean[1],a95=B95,color='k',edge='k',marker='*',markersize=450,label='Reference')
        if boot_test==True:
            ipmag.plot_di(di_block=boot_test_data,color='cyan',edge='k',marker='o',markersize=15,label='Study VGPs (N=%3d)' % N_test)
            boot_func.plot_di_mean(dec=boot_test_mean['dec'],inc=boot_test_mean['inc'],a95=B95_test,color='blue',edge='k',marker='*',markersize=450)
        if len(test_data)>1:
            ipmag.plot_di(di_block=test_data,color='red',edge='k',marker='o',markersize=40,label='Study VGPs (N=%3d)' % N_test)
        boot_func.plot_di_mean(dec=test_pole[0],inc=test_pole[1],a95=test_mean['alpha95'],color='red',edge='k',marker='*',markersize=450,label='Study pole')
        plt.legend(loc=1, fontsize = 16) 
        plt.show()    
    else:
        print('No plot')

    if boot_test==True:
        return B95, B95_test
    else:
        return B95

#----------------------------------
# DEFINE FUNCTIONS FOR TOTAL RECONSTRUCTION POLE CALCULATION

def sph2cart(lat,lon,r):
    """
    Converts an Euler pole into cartesian coordinates
    Euler pole should be given as (latitude, longitude, rotation angle)
    """
    x=r*np.cos(lat)*np.cos(lon)
    y=r*np.cos(lat)*np.sin(lon)
    z=r*np.sin(lat)
    return x, y, z

def rot2mat(pole):
    """
    Converts an Euler pole into a (3x3) rotation matrix
    Euler pole should be given as an array of length 3 (latitude, longitude, rotation angle)
    """ 
    gr=np.pi/180.
    E=sph2cart(pole[0]*gr,pole[1]*gr,1)   # converts pole to cartesian coordinates
    omega=pole[2]*gr                      # converts angle to radians
    R=np.zeros((3,3))                     # creates 3x3 matrix
    R[0][0]=E[0]*E[0]*(1-np.cos(omega)) + np.cos(omega)
    R[0][1]=E[0]*E[1]*(1-np.cos(omega)) - E[2]*np.sin(omega)
    R[0][2]=E[0]*E[2]*(1-np.cos(omega)) + E[1]*np.sin(omega)
    R[1][0]=E[1]*E[0]*(1-np.cos(omega)) + E[2]*np.sin(omega)
    R[1][1]=E[1]*E[1]*(1-np.cos(omega)) + np.cos(omega)
    R[1][2]=E[1]*E[2]*(1-np.cos(omega)) - E[0]*np.sin(omega)
    R[2][0]=E[2]*E[0]*(1-np.cos(omega)) - E[1]*np.sin(omega)
    R[2][1]=E[2]*E[1]*(1-np.cos(omega)) + E[0]*np.sin(omega)
    R[2][2]=E[2]*E[2]*(1-np.cos(omega)) + np.cos(omega)
    return R

def mat2rot(m):
    """
    Converts a (3x3) rotation matrix into an Euler pole
    Euler pole should be given as an array of length 3 (latitude, longitude, rotation angle)
    """
    gr=np.pi/180.
    lon = np.arctan2( m[0][2] - m[2][0] , m[2][1] - m[1][2] )
    term = np.sqrt( (m[2][1]-m[1][2])**2 + (m[0][2]-m[2][0])**2 + (m[1][0]-m[0][1])**2 )
    if term==0:
        pole=np.zeros(3)
    else:
        lat = np.arcsin( (m[1][0]-m[0][1]) / term )
        ang = np.arctan2( term , (m[0][0]+m[1][1]+m[2][2]-1) )
        pole = [lat/gr,lon/gr,ang/gr]
    return pole

def addpoles(p1,p2):
    """
    Adds two Euler poles
    Euler poles should be given as an array of length 3 (latitude, longitude, rotation angle)
    """
    A=rot2mat(p1)
    B=rot2mat(p2)
    C=np.dot(B,A)
    if np.array_equal(C,np.identity(3)) is True:
        result=np.zeros(3)
    else:
        result=mat2rot(C)
    return result

def stagepole(p_old,p_young):
    """
    Computes forward (total) stage pole
    For example, computes stage pole of Eurasia vs North America from 40 to 30 Ma
    Output is an Euler pole given as (latitude, longitude, rotation angle)
    Note that the rotation angle represents the total rotation during the stage
    """
    if np.array_equal(p_old,p_young) is True:
        stage_p=np.zeros(3)
    else:
        p_t2=p_old.copy()
        p_t2[2]=-1*p_t2[2]
        stage_p=addpoles(p_t2,p_young)
    return stage_p

def interpole(trp,sp,t2,t1,t):
    """
    Computes the total reconstruction pole at a desired age (t)
    Requires a total reconstruction pole at an older time (t2) and a stagepole describing the motion between t2 and t1
    Here, t1 is end of the motion stage described by the stage pole, such that t1 < t < t2
    Output is an Euler pole given as (latitude, longitude, rotation angle)
    """
    stagepole=sp.copy()
    delta=(t2-t)/(t2-t1)
    stagepole[2]=stagepole[2]*delta
    interp_pole=addpoles(trp,stagepole)
    return interp_pole

def rot2frame(ID,t,rot_data):
    """
    Computes the total reconstruction pole of a specific plate (ID) relative to the reference frame (ID=0)
    at a chosen age (t)

    Specify rotation file as rot_file, e.g., 'master.rot'    
    """
    
    # STORE DATA IN ARRAYS
    plate_IDs=rot_data[:,0]     # stores plate IDs in array
    ages=rot_data[:,1]          # stores ages of rotation poles in array
    file_length=len(plate_IDs)  # computes length of rotation file
    
    trp=np.zeros(3)
    while (ID != 0):
        for j in range(file_length):              # loops through rotation file
            if (plate_IDs[j]==ID):
                if (ages[j]>=t):                  # finds age which is greater than t
                    pole_old=rot_data[j,2:5]          # defines pole at t > t_interpolation
                    pole_young=rot_data[j-1,2:5]      # defines pole at t < t_interpolation
                    pole_stage=stagepole(pole_old,pole_young)   # computes stage pole
                    pole_at_t=interpole(pole_old,pole_stage,ages[j],ages[j-1],t)    # computes pole at t_interpolation
                    trp=addpoles(trp,pole_at_t)   # computes new total pole
                    #print(t,ID,rot_data[j,5],pole_old,pole_young,pole_at_t,trp) # PRINT CHAIN OF ROTATIONS
                    ID=rot_data[j,5]                  # updates plate ID              
    return trp

def tr_pole(plate_ID,fixed_plate_ID,rot_data,t_max,multi_t=False,dt=0):
    """
    Computes the total reconstruction pole of a specific plate (ID) relative to a chosen fixed plate
    at selected times (t)

    Input:
    plate_ID: selected plate
    fixed_plate_ID: selected fixed plate
    rot_file: rotation text file, e.g., 'master.rot'
    t: reads (maximum) age of requested rotation pole(s) (in Myr)
    multi_t: True if poles should be computed for multiple ages
    dt: time interval between rotation poles (in Myr)  
    """

    if multi_t==False:
        t=np.array([t_max])
    else:
        nt_float=t_max/dt+1
        nt=int(t_max/dt+1)
        if (nt_float-nt > 0.0001):
            print('Error: Define correct time interval')
        time=0
        t=np.zeros(nt)
        while (time != t_max):      # creates array of ages for which poles are computed
            for i in range(nt):
                t[i]=time
                time=time+dt
                if (time==t_max):   # if t_max is reached, store final age and break loop
                    t[nt-1]=t_max
                    break

    # COMPUTE TOTAL RECONSTRUCTION POLES AT SELCTED TIMES
    tr_poles = []
    for i in range(len(t)):         # loops over selected times
        if (t[i]==0):           
            print(0,',',0,',',0,',',t[i])    # prints pole at t=0 Ma
            tr_poles.append([0,0,0])
        else:
            plate_pole=rot2frame(plate_ID,t[i],rot_data)                 # computes pole of selected plate rel. to reference frame
            fixed_plate_pole=rot2frame(fixed_plate_ID,t[i],rot_data)     # computes pole of fixed plate rel. to reference frame
            fixed_plate_pole[2]=-1*fixed_plate_pole[2]          # changes sign of rotation angle
            POLE=addpoles(plate_pole,fixed_plate_pole)          # computes total reconstruction pole
            tr_poles.append(POLE)

            #---------------------------------
            # PRINT TOTAL RECONSTRUCTION POLE
            #print(t[i],',',np.round(POLE[0],3),',',np.round(POLE[1],3),',',np.round(POLE[2],3)) # print .rot file order (age,lat,lon,ang)

    # OUTPUT
    if multi_t==False:
        return POLE     # returns total reconstruction pole (lat,lon,ang)
    else:
        return tr_poles # returns nested list of tr poles
    
def tr_pole_2(plate_ID,fixed_plate_ID,rot_data,t_max,multi_t=False,dt=0,multi_lists=False):
    """
    Computes the total reconstruction pole of a specific plate (ID) relative to a chosen fixed plate
    at selected times (t)

    Input:
    plate_ID: selected plate
    fixed_plate_ID: selected fixed plate
    rot_file: rotation text file, e.g., 'master.rot'
    t_max: reads (maximum) age of requested rotation pole(s) (in Myr)
    multi_t: True if poles should be computed for multiple ages
    dt: time interval between rotation poles (in Myr)  
    """
    if multi_t==False:
        t=np.array([t_max])
    else:
        nt_float=t_max/dt+1
        nt=int(t_max/dt+1)
        if (nt_float-nt > 0.0001):
            print('Error: Define correct time interval')
            print(nt_float,nt,t_max)
        time=0
        t=np.zeros(nt)
        while (time != t_max):      # creates array of ages for which poles are computed
            for i in range(nt):
                t[i]=time
                time=time+dt
                if (time==t_max):   # if t_max is reached, store final age and break loop
                    t[nt-1]=t_max
                    break

    # COMPUTE TOTAL RECONSTRUCTION POLES AT SELCTED TIMES

    if multi_lists == True:
        plate_IDs,ages,EP_lats,EP_lons,EP_angs=[],[],[],[],[]

    tr_poles = []
    for i in range(len(t)):         # loops over selected times
        if (t[i]==0):           
            #print(0,',',0,',',0,',',t[i])    # prints pole at t=0 Ma
            tr_poles.append([0,0,0])
            if multi_lists==True:
                plate_IDs.append(plate_ID)
                ages.append(0)
                EP_lats.append(0)
                EP_lons.append(0)
                EP_angs.append(0)
        else:
            plate_pole=rot2frame(plate_ID,t[i],rot_data)                 # computes pole of selected plate rel. to reference frame
            fixed_plate_pole=rot2frame(fixed_plate_ID,t[i],rot_data)     # computes pole of fixed plate rel. to reference frame
            fixed_plate_pole[2]=-1*fixed_plate_pole[2]          # changes sign of rotation angle
            POLE=addpoles(plate_pole,fixed_plate_pole)          # computes total reconstruction pole
            tr_poles.append(POLE)

            if multi_lists==True:
                plate_IDs.append(plate_ID)
                ages.append(t[i])
                EP_lats.append(POLE[0])
                EP_lons.append(POLE[1])
                EP_angs.append(POLE[2])

            #---------------------------------
            # PRINT TOTAL RECONSTRUCTION POLE
            #print(plate_ID,t[i],plate_pole,fixed_plate_pole)
            #print(t[i],',',np.round(POLE[0],3),',',np.round(POLE[1],3),',',np.round(POLE[2],3)) # print .rot file order (age,lat,lon,ang)

    # OUTPUT
    if multi_t==False:
        return POLE     # returns total reconstruction pole (lat,lon,ang)
    elif multi_lists==True:
        return plate_IDs,ages,EP_lats,EP_lons,EP_angs
    else:
        return tr_poles # returns nested list of tr poles

def pt_rot(EP, Lats, Lons):
    """
    Rotates points on a globe by an Euler pole rotation using method of
    Cox and Hart 1986, box 7-3.
    Parameters
    ----------
    EP : Euler pole list [lat,lon,angle] specifying the location of the pole;
    the angle is for a counterclockwise rotation about the pole
    Lats : list of latitudes of points to be rotated
    Lons : list of longitudes of points to be rotated
    Returns
    _________
    RLats : list of rotated latitudes
    RLons : list of rotated longitudes
    """
# gets user input of Rotation pole lat,long, omega for plate and converts
# to radians
    E = pmag.dir2cart([EP[1], EP[0], 1.])  # EP is pole lat,lon omega
    omega = np.radians(EP[2])  # convert to radians
    RLats, RLons = [], []
    for k in range(len(Lats)):
        if Lats[k] <= 90.:  # peel off delimiters
            # converts to rotation pole to cartesian coordinates
            A = pmag.dir2cart([Lons[k], Lats[k], 1.])
# defines cartesian coordinates of the pole A
            R = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
            R[0][0] = E[0] * E[0] * (1 - np.cos(omega)) + np.cos(omega)
            R[0][1] = E[0] * E[1] * (1 - np.cos(omega)) - E[2] * np.sin(omega)
            R[0][2] = E[0] * E[2] * (1 - np.cos(omega)) + E[1] * np.sin(omega)
            R[1][0] = E[1] * E[0] * (1 - np.cos(omega)) + E[2] * np.sin(omega)
            R[1][1] = E[1] * E[1] * (1 - np.cos(omega)) + np.cos(omega)
            R[1][2] = E[1] * E[2] * (1 - np.cos(omega)) - E[0] * np.sin(omega)
            R[2][0] = E[2] * E[0] * (1 - np.cos(omega)) - E[1] * np.sin(omega)
            R[2][1] = E[2] * E[1] * (1 - np.cos(omega)) + E[0] * np.sin(omega)
            R[2][2] = E[2] * E[2] * (1 - np.cos(omega)) + np.cos(omega)
# sets up rotation matrix
            Ap = [0, 0, 0]
            for i in range(3):
                for j in range(3):
                    Ap[i] += R[i][j] * A[j]
# does the rotation
            Prot = pmag.cart2dir(Ap)
            RLats.append(Prot[1])
            RLons.append(Prot[0])
        else:  # preserve delimiters
            RLats.append(Lats[k])
            RLons.append(Lons[k])

    if len(Lats)==1:
        return RLons[0],RLats[0]
    else:
        return RLons, RLats

#----------------------------------

def fish_VGPs(K=20, n=100, lon=0, lat=90):
    """
    Generates Fisher distributed unit vectors from a specified distribution
    using the pmag.py fshdev and dodirot functions.
    Parameters
    ----------
    k : kappa precision parameter (default is 20)
    n : number of vectors to determine (default is 100)
    lon : mean longitude of distribution (default is 0)
    lat : mean latitude of distribution (default is 90)
    di_block : this function returns a nested list of [lon,lat] as the default
    if di_block = False it will return a list of lon and a list of lat
    Returns
    ---------
    di_block : a nested list of [lon,lat] (default)
    lon,lat : a list of lon and a list of lat (if di_block = False)
    """

    k = np.array(K)
    R1 = random.random(size=n)
    R2 = random.random(size=n)
    L = np.exp(-2 * k)
    a = R1 * (1 - L) + L
    fac = np.sqrt(-np.log(a)/(2 * k))
    inc = 90. - np.degrees(2 * np.arcsin(fac))
    dec = np.degrees(2 * np.pi * R2)

    DipDir, Dip = np.ones(n, dtype=float).transpose(
    )*(lon-180.), np.ones(n, dtype=float).transpose()*(90.-lat)
    data = np.array([dec, inc, DipDir, Dip]).transpose()
    drot, irot = pmag.dotilt_V(data)
    drot = (drot-180.) % 360.  #
    VGPs = np.column_stack((drot, irot))
    # rot_data = np.column_stack((drot, irot))
    # VGPs = rot_data.tolist()

    # for data in range(n):
    #     lo, la = pmag.fshdev(K)
    #     drot, irot = pmag.dodirot(lo, la, lon, lat)
    #     VGPs.append([drot, irot])
    return VGPs

def gen_pseudopoles(ref_poles,kappas,ns,N_ref_poles,N_test=0,equal_N=False):
    """
    Generates pseudopole from reference dataset of paleopoles with N and K
    """

    # generate nested list of parametrically sampled VGPs
    nested_VGPs = [fish_VGPs(K=kappas[j],n=ns[j],lon=ref_poles[j][0],lat=ref_poles[j][1]) for j in range(N_ref_poles)]
    # create single list with all simulated VGPs
    sim_VGPs = [nested_VGPs[i][j] for i in range(N_ref_poles) for j in range(ns[i])]
    #print(sim_VGPs)

    # if equal_N==True:
    #     # select random VGPs from dataset
    #     Inds = np.random.randint(len(sim_VGPs), size=N_test)
    #     #print(Inds)
    #     D = np.array(sim_VGPs)
    #     sample = D[Inds]
    # else:
    #     sample = sim_VGPs

    # compute pseudopole
    polepars = pmag.fisher_mean(sim_VGPs)
    return [polepars['dec'], polepars['inc']]

def unb_pseudopoles(ref_poles,kappas,ns,N_ref_poles,N_test=0,equal_N=False):
    """
    Generates pseudopole from reference dataset of paleopoles with N and K. Poles are randomly drawn with replacement.
    """

    # generate random integers between 0 and N_ref_poles
    inds = np.random.randint(N_ref_poles,size=N_ref_poles)
    #print(inds)

    # generate nested list of parametrically sampled VGPs
    nested_VGPs = [fish_VGPs(K=kappas[j],n=ns[j],lon=ref_poles[j][0],lat=ref_poles[j][1]) for j in inds]
    # create single list with all simulated VGPs
    # can this be done better/faster?!
    sim_VGPs = [nested_VGPs[i][j] for i in range(N_ref_poles) for j in range(len(nested_VGPs[i]))]
    #print(sim_VGPs)

    # if equal_N==True:
    #     # select random VGPs from dataset
    #     Inds = np.random.randint(len(sim_VGPs), size=N_test)
    #     #print(Inds)
    #     D = np.array(sim_VGPs)
    #     sample = D[Inds]
    # else:
    #     sample = sim_VGPs

    # compute pseudopole
    polepars = pmag.fisher_mean(sim_VGPs)
    return [polepars['dec'], polepars['inc']]

def gen_pseudopoles_file(ref_VGPs,N_test):
    """
    Generates pseudopole from reference VGPs stored in csv file
    """

    # select random VGPs from dataset
    Inds = np.random.randint(len(ref_VGPs), size=N_test)
    #print(Inds)
    D = np.array(ref_VGPs)
    sample = D[Inds]

    # compute pseudopole
    polepars = pmag.fisher_mean(sample)
    return [polepars['dec'], polepars['inc']]
    
def par_VGPs(ref_poles,kappas,ns,N_ref_poles):
    """
    Generates nb sets of parametrically sampled VGPs from set of reference poles
    """
    # generate nested list of parametrically sampled VGPs
    nested_VGPs = [fish_VGPs(K=kappas[j],n=ns[j],lon=ref_poles[j][0],lat=ref_poles[j][1]) for j in range(N_ref_poles)]
    #print(nested_VGPs)
    # create single list with all simulated VGPs
    sim_VGPs = [nested_VGPs[i][j] for i in range(N_ref_poles) for j in range(ns[i])]
    #print(sim_VGPs)

    return sim_VGPs

def bootstrap_refpole(ref_poles,kappas,ns,nb=1000):
    """
    Computes a bootstrapped reference pole from parametrically sampled VGPs 

    Returns reference pole (Fisher mean of pseudopoles) and nested list of bootstrapped poles
    """
    N_poles = len(ref_poles) # store number of VGPs in reference dataset
    
    # GENERATE nb PSEUDOPOLES FROM REFERENCE DATASET
    ppoles = [gen_pseudopoles(ref_poles,kappas,ns,N_poles) for i in range(nb)]

    # COMPUTE REFERENCE MEAN
    ppole_mean = pmag.fisher_mean(ppoles) # compute mean of bootstrapped poles
    fish_mean = [ppole_mean['dec'],ppole_mean['inc']] # store reference mean

    return fish_mean,ppoles

# ---------
def age_samp(age_min,age_max,N,t_min,t_max):
    """
    Generates N random ages from age uncertainty interval that fall within time window

    Input:
    age_min = lower age bound
    age_max = upper age bound
    N = number of sites used to compute paleopole
    t_min and t_max = time winndow
    """
    ages = random.uniform(low=age_min,high=age_max,size=N)

    sel_ages = [ages[i] for i in range(N) if ages[i]>=t_min and ages[i]<=t_max]
    
    return sel_ages

def rot_VGPs(paleopole,N,K,age_min,age_max,t_min,t_max,plate_ID,ref_plate,plate_circ,return_ages=False):
    """
    Rotates VGPs through plate circuit using specific age for each VGP

    Input:
    paleopole = [longitude,latitude]
    N = number of sites used to compute paleopole
    K = Fisher (1953) precision parameter
    age_min = lower age bound
    age_max = upper age bound
    plate_circ = GPlates rotation (.rot) file converted to Numpy array
    t_min and t_max = time window
    """
    # generate N random ages and select those that fall within time window
    VGP_ages = age_samp(age_min,age_max,N,t_min=t_min,t_max=t_max)
    # print('')
    # print('')
    # print(VGP_ages)

    # determine number of VGPs to be rotated
    N_sel = len(VGP_ages)

    # generate N_sel VGPS from paleopole and K
    #starttime = timeit.default_timer()
    VGPs = fish_VGPs(K=K,n=N_sel,lon=paleopole[0],lat=paleopole[1])
    #print("Time 1 :", timeit.default_timer() - starttime)
    #print(VGPs)

    #starttime = timeit.default_timer()
    EPs = [tr_pole(plate_ID,ref_plate,plate_circ,VGP_ages[i]) for i in range(N_sel)]
    #print("Time 2 :", timeit.default_timer() - starttime)
    #print('EPs=',EPs)

    #starttime = timeit.default_timer()
    rVGPs = [pt_rot(EPs[j],Lats=[VGPs[j][1]],Lons=[VGPs[j][0]]) for j in range(N_sel)]
    #print(rVGPs)
    #print("Time 3 :", timeit.default_timer() - starttime)

    if return_ages==False:
        return rVGPs
    else: 
        return rVGPs,VGP_ages

def age_prop_pseudopoles(lons,lats,Ks,Ns,age_mins,age_maxs,plate_IDs,t_min,t_max,ref_plate,plate_circ):
    """
    Computes pseudopole from parametrically sampled VGPs (derived from paleopole and assocaited N and K), whereby each VGP is rotated through
    plate circuit at randomly drawn age within the age uncertainty range of the given paleopole, given that the age falls within the selected
    time window (t_min<=age<=t_max)
    """
    N_poles = len(lats) # determine number of paleopoles

    # Rotate VGPs through plate circuit using specific age for each VGP
    rVGPs = [rot_VGPs([lons[k],lats[k]],Ns[k],Ks[k],age_mins[k],age_maxs[k],t_min,t_max,plate_IDs[k],ref_plate,plate_circ) for k in range(N_poles)]

    # Store all rotated VGPs in single list
    all_VGPs = [rVGPs[i][j] for i in range(N_poles) for j in range(len(rVGPs[i]))]

    # Compute pseudopole
    polepars = pmag.fisher_mean(all_VGPs)

    return [polepars['dec'], polepars['inc']]