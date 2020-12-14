
#
# actions-sim.py
#
# reading Gaia DR2 (will be replaced with EDR3) Ralph_48-00_actions.fits
# from Jason Hunt (CCA)
#

from astropy.io import fits as pyfits
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib import patches
from scipy import stats
from scipy import optimize
import scipy.interpolate
from sklearn import linear_model
from sklearn.neighbors import KernelDensity

# for not displaying
# matplotlib.use('Agg')

##### main programme start here #####

# flags

print(' assumed zsun and vcirc=', rsun, vcircsun)

# condition to select observed stars
# |z| < zmaxlim
zmaxlim = 0.5
# lz > lzlim
lzlim = 0.0
# rlim
rlimlow = 3.0
rlimhigh = 18.0
# jrlim
jrlim = 0.3
# omega
omegalim = 4.0


### simulation data parameters
vcircsim = 197.0
rsunsim = 8.0
# selection
rmax = 18.0
# bar pattter speed
omega_b = 40.0
# Lz0 and omega0
lz0sim = rsunsim*vcircsim
omega0 = vcircsim/rsunsim
# adjust Lz0sim
olrsim = 1.15
olrobs = 1.323
# lz0sim = lz0sim*(olrsim/olrobs)

#### read the simulation data
infile = '../GaiaEDR3/actions_jashunt/eDR3_actions.fits'
star_hdus = pyfits.open(infile)
starobs = star_hdus[1].data
star_hdus.close()

print(' number of input =', len(starobs['R']))

# selection of stars
sindx = np.where((np.fabs(starobs['z']) < zmaxlim) & \
                 (starobs['lz']>lzlim) & \
                 (starobs['jR']<0.3) &
                 (np.fabs(starobs['Omega_p'])<omegalim) &
                 (np.fabs(starobs['Omega_r'])<omegalim) &        
                 (starobs['R']>rlimlow) & (starobs['R']<rlimhigh))
#                 (star['age']>agelim))
nstars = len(starobs['R'][sindx])

print(' N selected obs stars=',nstars)

# for selected stars (positive rotation in clockwise)
vradobs = starobs['vR'][sindx]
vphiobs = starobs['vphi'][sindx]
vz = starobs['vz'][sindx]
# normalised by rsun and vcirc
jrobs = starobs['jR'][sindx]
lzobs = starobs['lz'][sindx]
jzobs = starobs['jz'][sindx]
omegarobs = starobs['Omega_r'][sindx]*(vcircsun/rsun)
omegazobs = starobs['Omega_z'][sindx]*(vcircsun/rsun)
omegaphiobs = starobs['Omega_p'][sindx]*(vcircsun/rsun)
rgalobs = starobs['R'][sindx]

# Lz histogram of selected obs data
# with Lz
lzmin = 0.0
lzmax = np.max(lzobs)
nhist = 100
print(' Lz range of selected obs data=', lzmin, lzmax)
lzhist_obs, bin_edges = np.histogram(lzobs, bins=nhist, range=(lzmin, lzmax), density=True)
# R
rgalmin = 3.0
rgalmax = np.max(rgalobs)
nhist = 100
print(' R range of selected obs data=', rgalmin, rgalmax)
rgalhist_obs, bin_edges = np.histogram(rgalobs, bins=nhist, range=(rgalmin, rgalmax), density=True)

# read the simulation data
infile = '../BabaMW/actionsFileID7001.fits'
star_hdus = pyfits.open(infile)
star = star_hdus[1].data
star_hdus.close()
# compute probability from KDE for each particles
# normalises Lz
lzsims = -star['Jphi']/lz0sim
rgalsims = star['R']
probs = np.zeros_like(lzsims)
presindx = np.where((np.fabs(star['z']) < zmaxlim) & \
                 (lzsims>lzlim) & \
                 (lzsims<lzmax) & \
                 (star['R']<rmax))
print(' number of pre-selected stars=',len(star['R'][presindx]))

# lz histogram of observational data
lzhist_sim, bin_edges = np.histogram(lzsims[presindx], bins=nhist, range=(lzmin, lzmax), density=True)
# Lz bins
lz_bins = 0.5*(bin_edges[:nhist]+bin_edges[1:])

plt.plot(lz_bins, lzhist_sim, color='black')
plt.plot(lz_bins, lzhist_obs, color='blue')
plt.show()

# probability
# the min fractoin
minfrac_simobs = np.min(lzhist_sim/lzhist_obs)
print(' minimum of sim/obs = ', minfrac_simobs)
if minfrac_simobs>1.0:
      minfrac_simobs =1.0
probs_bins = lzhist_obs/(lzhist_sim/minfrac_simobs)
probs = np.zeros_like(lzsims)
probs[presindx] = np.interp(lzsims[presindx], lz_bins, probs_bins)

# plt.plot(lz_bins, lzhist_sim/minfrac_simobs, color='black')
# plt.plot(lz_bins, lzhist_obs, color='blue')
# plt.show()

# R histogram of simulation data
rgalhist_sim, bin_edges = np.histogram(rgalsims[presindx], bins=nhist, range=(rgalmin, rgalmax), density=True)
# R bins
rgal_bins = 0.5*(bin_edges[:nhist]+bin_edges[1:])

plt.plot(rgal_bins, rgalhist_sim, color='black')
plt.plot(rgal_bins, rgalhist_obs, color='blue')
plt.show()

# probability
# the min fractoin
minfrac_simobs = np.min(rgalhist_sim[rgalhist_obs>0.0]/rgalhist_obs[rgalhist_obs>0.0])
print(' minimum of sim/obs in R= ', minfrac_simobs)
if minfrac_simobs>1.0:
      minfrac_simobs =1.0
probs = np.zeros_like(lzsims)      
probs_bins = rgalhist_obs[rgalhist_sim>0.0]/(rgalhist_sim[rgalhist_sim>0.0]/minfrac_simobs)

probs[presindx] = np.interp(rgalsims[presindx], rgal_bins, probs_bins)

# random number
num_random = np.random.random(len(star['R']))

# selection of stars

sindx = np.where((np.fabs(star['z']) < zmaxlim) & \
                 (lzsims>lzlim) & \
                 (lzsims<lzmax) & \
                 (star['R']<rmax) & \
                 (num_random<probs))

# for selected stars (positive rotation in clockwise)
vrads = star['vR'][sindx]
vphis = -star['vphi'][sindx]
vz = star['vz'][sindx]
# normalised by rsun and vcirc
jrs = star['Jr'][sindx]/lz0sim
lzs = -star['Jphi'][sindx]/lz0sim
jzs = star['Jz'][sindx]/lz0sim
omegars = star['omegar'][sindx]
omegazs = star['omegaz'][sindx]
omegaphis = -star['omegaphi'][sindx]
ages = star['age'][sindx]
rgals = star['R'][sindx]

print(' N selected from simulation =', len(ages))

# histogram of Lz to check the consistency of the data sample
# output
plt.hist(lzobs, bins = lz_bins, fc='blue', density=True, histtype='step')
plt.hist(lzs, bins = lz_bins, fc='green', density=True, histtype='step')
plt.show()
r_bins = np.linspace(3.0, 16.0, nhist)
plt.hist(rgalobs, bins = r_bins, fc='blue', density=True, histtype='step')
plt.hist(rgals, bins = r_bins, fc='green', density=True, histtype='step')
plt.show()

# minimum number of stars in each column
nsmin = 25
# set number of grid
ngridx = 200
ngridy = 250
# npline
npline = 1000
                 
# grid plot for Lz vs. Jr
lzrange = np.array([0.001, 1.75])
jrrange = np.array([0.0, 0.2])

#jrticks = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])

# 2D histogram 
H, xedges, yedges = np.histogram2d(lzs, jrs, \
                    bins=(ngridx, ngridy), \
                    range=(lzrange, jrrange))
# set x-axis (lzs) is axis=1
H = H.T
# normalised by row
# print(' hist = ',np.shape(H))
# print(' np row = ',np.sum(H, axis=1))
# log value
nlogmin = 0.1
# nlogmin = -4.0
nminlim = np.power(10.0, nlogmin)
Hlog = np.zeros_like(H)
for i in range(ngridy):
  nprow = np.sum(H[i, :])
  # if nprow >= nsmin:
  #  H[i, :] /= nprow
  # else:
  #  H[i, :] = 0.0
  Hlog[i, H[i, :]<=nminlim]=nlogmin
  Hlog[i, H[i, :]>nminlim]=np.log10(H[i, H[i, :]>nminlim])
  
# print(' after normalised=',np.sum(H, axis=1))
# print(' H=', H[ngridx-4, :])
print(' Hmax =', np.max(H))

# nslim for fitting
nslimfit = 10

# compute resonance location in action space
domega = 0.05
# pick up stars at CR
mres = 2.0
lres = 0.0
indxcr = np.where(np.fabs((omega_b-omegaphis))<domega)
print(' np CR=', np.shape(indxcr))
if len(lzs[indxcr])>nslimfit:
    # robust fit
    ransac_cr = linear_model.RANSACRegressor()
    y = lzs[indxcr].reshape(-1, 1)
    X = jrs[indxcr].reshape(-1, 1)
    ransac_cr.fit(X, y)
    # Predict data of estimated models
    line_ycr = np.linspace(jrrange[0], jrrange[1],npline).reshape(-1,1)
    line_Xcr = ransac_cr.predict(line_ycr)
else:
    line_ycr = np.linspace(jrrange[0], jrrange[1],npline).reshape(-1,1)
    line_Xcr = np.zeros_like(line_ycr)-1.0
 

# 4:1 resonance
mres = 4.0
lres = 1.0
indx41 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
print(' np 4:1=', np.shape(indx41))
if len(lzs[indx41])>nslimfit:
    # robust fit
    ransac_41 = linear_model.RANSACRegressor()
    y = lzs[indx41].reshape(-1, 1)
    X = jrs[indx41].reshape(-1, 1)
    ransac_41.fit(X, y)
    # Predict data of estimated models
    line_y41 = np.linspace(jrrange[0], jrrange[1],npline).reshape(-1,1)
    line_X41 = ransac_41.predict(line_y41)
else:
    line_y41 = np.linspace(jrrange[0], jrrange[1],npline).reshape(-1,1)
    line_X41 = np.zeros_like(line_y41)-1.0
  
# OLR resonance
mres = 2.0
lres = 1.0
indxolr = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
print(' np OLR=', np.shape(indxolr))
if len(lzs[indxolr])>nslimfit:
    # robust fit
    ransac_olr = linear_model.RANSACRegressor()
    y = lzs[indxolr].reshape(-1, 1)
    X = jrs[indxolr].reshape(-1, 1)
    ransac_olr.fit(X, y)
    # Predict data of estimated models
    line_yolr = np.linspace(jrrange[0], jrrange[1], npline).reshape(-1,1)
    line_Xolr = ransac_olr.predict(line_yolr)
else:
    line_yolr = np.linspace(jrrange[0], jrrange[1], npline).reshape(-1,1)
    line_Xolr = np.zeros_like(line_xolr)
print(' OLR Lz at Jr=0 =', line_Xolr[0])

# 4:3 resonance, but the ridges does not show up well. 
mres = 4.0
lres = 3.0
indx43 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
print(' np 4:3=', np.shape(indx43))
if len(lzs[indx43])>nslimfit:
    # robust fit
    ransac_43 = linear_model.RANSACRegressor()
    X = lzs[indx43].reshape(-1, 1)
    y = jrs[indx43].reshape(-1, 1)
    ransac_43.fit(X, y)
    # Predict data of estimated models
    line_X43 = np.linspace(lzrange[0], lzrange[1],npline).reshape(-1,1)
    line_y43 = ransac_43.predict(line_X43)
else:
    line_X43 = np.linspace(lzrange[0], lzrange[1],npline).reshape(-1,1)
    line_y43 = np.zeros_like(line_X43)

# 1:1 resonance, but the ridges does not show up well. 
mres = 1.0
lres = 1.0
indx11 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
print(' np 1:1=', np.shape(indx11))
if len(lzs[indx11])>nslimfit:
    # robust fit
    ransac_11 = linear_model.RANSACRegressor()
    X = lzs[indx11].reshape(-1, 1)
    y = jrs[indx11].reshape(-1, 1)
    ransac_11.fit(X, y)
    # Predict data of estimated models
    line_X11 = np.linspace(lzrange[0], lzrange[1],npline).reshape(-1,1)
    line_y11 = ransac_11.predict(line_X11)
else:
    line_X11 = np.linspace(lzrange[0], lzrange[1],npline).reshape(-1,1)
    line_y11 = np.zeros_like(line_X11)-1.0

# 4:-1 resonance, but the ridges does not show up well. 
mres = 4.0
lres = -1
indxi41 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
print(' np 4:-1=', np.shape(indxi41))
if len(lzs[indxi41])>nslimfit:
    # robust fit
    ransac_i41 = linear_model.RANSACRegressor()
    X = lzs[indxi41].reshape(-1, 1)
    y = jrs[indxi41].reshape(-1, 1)
    ransac_i41.fit(X, y)
    # Predict data of estimated models
    line_Xi41 = np.linspace(lzrange[0], lzrange[1],npline).reshape(-1,1)
    line_yi41 = ransac_i41.predict(line_Xi41)
else:
    line_Xi41 = np.linspace(lzrange[0], lzrange[1],npline).reshape(-1,1)
    line_yi41 = np.zeros_like(line_Xi41)-1.0
                 
# Final plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 16

# log plot
cmin = nlogmin
cmax = np.max(Hlog)
gauamplim=-10.0
gausiglim=50.0
f, (ax1) = plt.subplots(1, sharex = True, figsize=(6,4))
labpos = np.array([5.0, 40.0])
im1 = ax1.imshow(Hlog, interpolation='gaussian', origin='lower', \
        aspect='auto', vmin=cmin, vmax=cmax, \
        extent=[xedges[0], xedges[-1], \
                yedges[0], yedges[-1]], \
                 cmap=cm.jet)
ax1.set_xlim(xedges[0], xedges[-1])
ax1.set_ylim(yedges[0], yedges[-1])
                 
ax1.set_ylabel(r"J$_{\rm R}$ (L$_{\rm z,0}$)", fontsize=18)
ax1.tick_params(labelsize=16, color='k', direction="in")
# ax2.set_yticks(vrotticks)
ax1.set_xticks([0.5, 1.0, 1.5])    

plt.xlabel(r"L$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=18)
# plt.ylabel(r"$J_{\rm R}$ ($L_{\rm z,0}$)", fontsize=18)

# plot CR cyan
# plt.scatter(lzs[indxcr], jrs[indxcr], c='blue', marker='.',s=1)
plt.plot(line_Xcr, line_ycr, color='cyan')
# plot 4:1 orange
# plt.scatter(lzs[indx41], jrs[indx41], c='yellow', marker='.',s=1)
plt.plot(line_X41, line_y41, color='orange')
# plot OLR red
# plt.scatter(lzs[indxolr], jrs[indxolr], c='white', marker='.',s=1)
plt.plot(line_Xolr, line_yolr, color='red')
# plot 4:3 black
# plt.scatter(lzs[indx43], jrs[indx43], c='black', marker='o',s=1)
plt.plot(line_X43, line_y43, color='green')
# plot 1:1 white
# plt.scatter(lzs[indx11], jrs[indx11], c='white', marker='o',s=1)
plt.plot(line_X11, line_y11, color='white')
# 4:-1
plt.scatter(lzs[indxi41], jrs[indxi41], c='blue', marker='o',s=1)
plt.plot(line_Xi41, line_yi41, color='blue')


f.subplots_adjust(left=0.15, bottom = 0.15, hspace=0.0, right = 0.9)
#cbar_ax1 = f.add_axes([0.8, 0.15, 0.05, 0.725])
#cb1 = f.colorbar(im1, cax=cbar_ax1)
#cb1.ax.tick_params(labelsize=16)

# plt.savefig('lzjr-gedr3.eps')
# plt.savefig('lzjr-gedr3.jpg')
# plt.close(f)

plt.show()


### Analyse Lz distribution at fixed Jr
njrsamp = 4
# select differen Jr sample
# jrsamp_low = np.array([0.05, 0.075, 0.1])
# jrsamp_high = np.array([0.075, 0.01, 0.0125])
# jrsamp_low = np.array([0.1, 0.075, 0.05, 0.025])
# jrsamp_high = np.array([0.125, 0.1, 0.075, 0.05])
jrsamp_low = np.array([0.1, 0.075, 0.05, 0.025])
jrsamp_high = np.array([0.15, 0.1, 0.075, 0.05])

# lzrange
nhist = 200
lzmin_hist = lzrange[0]
lzmax_hist = lzrange[1]
# kernel size
hlz = 0.02
lz_bins = np.linspace(lzmin_hist, lzmax_hist, nhist)
# y range
ymin_hist = 0.0
ymax_hist = 2.9

f, ax = plt.subplots(4, sharex = True, figsize=(5,8))
f.subplots_adjust(hspace=0.0)
# f.subplots_adjust(bottom = 0.15)

jrnorm_all = star['Jr']/lz0sim

for i in range(njrsamp):
  print(' jr range=', jrsamp_low[i], jrsamp_high[i])
  # selection of stars
  sindx = np.where((np.fabs(star['z']) < zmaxlim) & \
                 (lzsims>lzlim) & \
                 (lzsims<lzmax) & \
                 (star['R']<rmax) & \
                 (num_random<probs) & \
                 (jrnorm_all>=jrsamp_low[i]) & \
                 (jrnorm_all<jrsamp_high[i]))
  print(' N selected stars=', len(star['R'][sindx]))
  # for selected stars (positive rotation in clockwise)
  vrads = star['vR'][sindx]
  vphis = star['vphi'][sindx]
  vz = star['vz'][sindx]
  # normalised by rsun and vcirc
  jrs = star['jR'][sindx]/lz0sim
  lzs = -star['jphi'][sindx]/lz0sim
  jzs = star['jz'][sindx]/lz0sim
  omegars = star['omegar'][sindx]
  omegazs = star['omegaz'][sindx]
  omegaphis = -star['omegaphi'][sindx]

  # resonance range
  # cr
  lindx = np.where((line_ycr>jrsamp_low[i]) & (line_ycr<jrsamp_high[i]))
  cr_low = np.min(line_Xcr[lindx])
  cr_high = np.max(line_Xcr[lindx])
  if i==0:
    cr_low0 = cr_low
    cr_high0 = cr_high
  print(' lz region for CR=', cr_low, cr_high)
  ax[i].add_patch(
    patches.Rectangle((cr_low, ymin_hist), cr_high-cr_low, \
                      ymax_hist-ymin_hist, facecolor='cyan', fill=True,alpha=0.5))
  # 4:1
  lindx = np.where((line_y41>jrsamp_low[i]) & (line_y41<jrsamp_high[i]))
  r41_low = np.min(line_X41[lindx])
  r41_high = np.max(line_X41[lindx])
  if i==0:
    r41_low0 = r41_low
    r41_high0 = r41_high
  print(' lz region for 4:1=', r41_low, r41_high)
  ax[i].add_patch(
    patches.Rectangle((r41_low, ymin_hist), r41_high-r41_low, \
                      ymax_hist-ymin_hist, facecolor='orange', fill=True,alpha=0.5))
  
  lindx = np.where((line_yolr>jrsamp_low[i]) & (line_yolr<jrsamp_high[i]))
  # print(' range, olr=', jrsamp_low[i], jrsamp_high[i], line_yolr[lindx], line_Xolr[lindx])
  olr_low = np.min(line_Xolr[lindx])
  olr_high = np.max(line_Xolr[lindx])
  if i==0:
    olr_low0 = olr_low
    olr_high0 = olr_high
  print(' lz region for OLR=', olr_low, olr_high)
  ax[i].add_patch(
    patches.Rectangle((olr_low, ymin_hist), olr_high-olr_low, \
                      ymax_hist-ymin_hist, facecolor='red', fill=True, alpha=0.5))

  # 4:3
  lindx = np.where((line_y43>jrsamp_low[i]) & (line_y43<jrsamp_high[i]))
  # print(' range, olr=', jrsamp_low[i], jrsamp_high[i], line_yolr[lindx], line_Xolr[lindx])
  r43_low = np.min(line_X43[lindx])
  r43_high = np.max(line_X43[lindx])
  if i==0:
    r43_low0 = r43_low
    r43_high0 = r43_high
  
  print(' lz region for 4:3 =', r43_low, r43_high)
  ax[i].add_patch(
    patches.Rectangle((r43_low, ymin_hist), r43_high-r43_low, \
                      ymax_hist-ymin_hist, facecolor='green', fill=True, alpha=0.5))

  # 4:-1
  lindx = np.where((line_yi41>jrsamp_low[i]) & (line_yi41<jrsamp_high[i]))
  # print(' range, olr=', jrsamp_low[i], jrsamp_high[i], line_yolr[lindx], line_Xolr[lindx])
  if len(line_Xi41[lindx])>0:  
      ri41_low = np.min(line_Xi41[lindx])
      ri41_high = np.max(line_Xi41[lindx])
  else:
      ri41_low = -1.0
      ri41_high = -1.0
  if i==0:
    ri41_low0 = ri41_low
    ri41_high0 = ri41_high
  
  print(' lz region for 4:-1 =', ri41_low, ri41_high)
  ax[i].add_patch(
    patches.Rectangle((ri41_low, ymin_hist), ri41_high-ri41_low, \
                      ymax_hist-ymin_hist, facecolor='blue', fill=True, alpha=0.5))
  

  # 1:1
  lindx = np.where((line_y11>jrsamp_low[i]) & (line_y11<jrsamp_high[i]))
  # print(' range, olr=', jrsamp_low[i], jrsamp_high[i], line_yolr[lindx], line_Xolr[lindx])
  if len(line_X11[lindx])>0:
      r11_low = np.min(line_X11[lindx])
      r11_high = np.max(line_X11[lindx])
  else:
      r11_low = -1.0
      r11_high = -1.0
  if i==0:
    r11_low0 = r11_low
    r11_high0 = r11_high
  print(' lz region for 1:1 =', r11_low, r11_high)
  ax[i].add_patch(
    patches.Rectangle((r11_low, ymin_hist), r11_high-r11_low, \
                      ymax_hist-ymin_hist, facecolor='grey', fill=True, alpha=0.5))
  
  
  # histogram 
  # ax[i].hist(lzs, bins = lz_bins, fc='#AAAAFF', density=True)
  # KDE
  kde = KernelDensity(kernel='epanechnikov', \
                      bandwidth=hlz).fit(lzs.reshape(-1, 1))
  log_dens = kde.score_samples(lz_bins.reshape(-1, 1))
  ax[i].plot(lz_bins, np.exp(log_dens), color='black')

  # set tick params
  ax[i].tick_params(labelsize=16, color='k', direction="in")
  ax[i].set_xlim(lzmin_hist, lzmax_hist)
  ax[i].set_ylim(ymin_hist, ymax_hist)
  if i==njrsamp-1:
      ax[i].set_xticks([0.5, 1.0, 1.5])  
  
plt.xlabel(r"L$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=18)


# plt.savefig('lzhist-gedr3.eps')
# plt.close(f)

plt.show()


###  Lz vs. Jz for selected Jr stars

# jzlim
jzlim = 0.3

# selected Jr region
jrmax = 0.15
jrmin = 0.1

sindx = np.where((np.fabs(star['z']) < zmaxlim) & \
                 (lzsims>lzlim) & \
                 (lzsims<lzmax) & \
                 (star['R']<rmax) & \
                 (num_random<probs) & \
                 (jrnorm_all>=jrmin) & \
                 (jrnorm_all<jrmax))
# for selected stars (positive rotation in clockwise)
vrads = star['vR'][sindx]
vphis = -star['vphi'][sindx]
vz = star['vz'][sindx]
# normalised by rsun and vcirc
jrs = star['jR'][sindx]/lz0sim
lzs = -star['jphi'][sindx]/lz0sim
jzs = star['jz'][sindx]/lz0sim
omegars = star['omegar'][sindx]
omegazs = star['omegaz'][sindx]
omegaphis = star['omegaphi'][sindx]

# set number of grid
ngridx = 300
ngridy = 150
                 
# grid plot for Lz vs. Jr
# lzrange = np.array([0.01, 1.75])
jzrange = np.array([0.0, 0.05])

#jrticks = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])

# 2D histogram 
H, xedges, yedges = np.histogram2d(lzs, jzs, \
                    bins=(ngridx, ngridy), \
                    range=(lzrange, jzrange))
# set x-axis (lzs) is axis=1
H = H.T
# normalised by row
# minimum number of stars in each column
# nsmin = 25
# log value
nlogmin = -2.0
# nlogmin = -4.0
nminlim = np.power(10.0, nlogmin)
Hlog = np.zeros_like(H)
for i in range(ngridy):
  # nprow = np.sum(H[i, :])
  # if nprow >= nsmin:
  #  H[i, :] /= nprow
  #else:
  #  H[i, :] = 0.0
  Hlog[i, H[i, :]<=nminlim]=nlogmin
  Hlog[i, H[i, :]>nminlim]=np.log10(H[i, H[i, :]>nminlim])
  
# print(' after normalised=',np.sum(H, axis=1))
# print(' H=', H[ngridx-4, :])
print('Lz-Jz Hmax =', np.max(H))

# high Jz stars Lz distribution
jzselmin = 0.005
jzselmax = 0.05
# jzselmin = 0.01
jzselmax = 0.05
jzindx = np.where((jzs>jzselmin) & (jzs<jzselmax))
print('jz range=', jzselmin, jzselmax, \
      ' N selected stars=', len(jzs[jzindx]))
# for selected stars (positive rotation in clockwise)
lzsels = lzs[jzindx]

# KDE
hlz = 0.03
nhist = 200
lz_bins = np.linspace(lzrange[0], lzrange[1], nhist)
kde = KernelDensity(kernel='epanechnikov', \
                      bandwidth=hlz).fit(lzsels.reshape(-1, 1))
log_dens = kde.score_samples(lz_bins.reshape(-1, 1))
# y range
ymin_hist = 0.0
ymax_hist = 2.2

# Final plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 16

# log plot
cmin = nlogmin
cmax = np.max(Hlog)*0.6
# cmin = 0.0
# cmax = np.max(H)*0.5
f, (ax1,ax2) = plt.subplots(2, sharex = True, figsize=(6,5), gridspec_kw={'height_ratios' : [1, 2]})
# f, (ax1, ax2) = plt.subplots(2, sharex = True, figsize=(6,4))
# f.subplots_adjust(hspace=0.0)
# KDE histogram
ax1.plot(lz_bins, np.exp(log_dens), color='black')
ax1.tick_params(labelsize=16, color='k', direction="in")
ax1.set_xlim(lzrange[0], lzrange[1])
ax1.set_ylim(ymin_hist, ymax_hist)
# add shaded region of resonances
ax1.add_patch(
    patches.Rectangle((cr_low0, ymin_hist), cr_high0-cr_low0, \
                      ymax_hist-ymin_hist, facecolor='cyan', fill=True,alpha=0.5))
ax1.add_patch(
    patches.Rectangle((r41_low0, ymin_hist), r41_high0-r41_low0, \
                      ymax_hist-ymin_hist, facecolor='orange', fill=True,alpha=0.5))
ax1.add_patch(
    patches.Rectangle((olr_low0, ymin_hist), olr_high0-olr_low0, \
                      ymax_hist-ymin_hist, facecolor='red', fill=True,alpha=0.5))
ax1.add_patch(
    patches.Rectangle((r43_low0, ymin_hist), r43_high0-r43_low0, \
                      ymax_hist-ymin_hist, facecolor='green', fill=True,alpha=0.5))
ax1.add_patch(
    patches.Rectangle((ri41_low0, ymin_hist), ri41_high0-ri41_low0, \
                      ymax_hist-ymin_hist, facecolor='green', fill=True,alpha=0.5))
ax1.add_patch(
    patches.Rectangle((r11_low0, ymin_hist), r11_high0-r11_low0, \
                      ymax_hist-ymin_hist, facecolor='green', fill=True,alpha=0.5))

# Lz vs. Jz
ax2.add_patch(
    patches.Rectangle((lzrange[0], jzselmin), \
                      lzrange[1]-lzrange[0], jzselmax-jzselmin, \
                      facecolor='pink', fill=True,alpha=0.5))
#im1 = ax1.imshow(H, interpolation='gaussian', origin='lower', \
im1 = ax2.imshow(Hlog, interpolation='gaussian', origin='lower', \
        aspect='auto', vmin=cmin, vmax=cmax, \
        extent=[xedges[0], xedges[-1], \
                yedges[0], yedges[-1]], \
                 cmap='Greens')
# ax1.scatter(lzs, jzs, marker='.', facecolors=cc, s=1)

ax2.set_xlim(xedges[0], xedges[-1])
ax2.set_ylim(yedges[0], yedges[-1])
                 
ax2.set_ylabel(r"J$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=18)
ax2.tick_params(labelsize=16, color='k', direction="in")
# ax2.set_yticks(vrotticks)
ax2.set_xticks([0.5, 1.0, 1.5])    

plt.xlabel(r"L$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=18)
# plt.ylabel(r"$J_{\rm R}$ ($L_{\rm z,0}$)", fontsize=18)

f.subplots_adjust(left=0.15, bottom = 0.15, hspace=0.0, right = 0.9)
#cbar_ax1 = f.add_axes([0.8, 0.15, 0.05, 0.725])
#cb1 = f.colorbar(im1, cax=cbar_ax1)
#cb1.ax.tick_params(labelsize=16)

#plt.savefig('lzjz-gedr3.eps')
# plt.savefig('lzjz-gedr3.jpg')
# plt.close(f)

plt.show()



