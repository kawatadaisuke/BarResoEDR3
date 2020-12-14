
#
# actions-sim.py
#
# reading BabaMW/actionsFileID7001.fits
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


##### main programme start here #####

# flags
Rweight = True
# making eps and jpg file for figure without showing in windows.
Paper = True

# for not displaying
if Paper==True:
    matplotlib.use('Agg')

# circular velocity at rsun in BabaMW at t=7 Gyr
rsun = 8.0
zsun = 0.0
vcircsun = 197.0

# condition to select stars
# |z| < zmaxlim
zmaxlim = 0.5
# age < agelim
agelim = 7000.0
# lz > lzlim
lzlim = 0.001
# R range
rgalmin = 3.0
rgalmax = 18.0
# Lz0 and omega0
lz0 = rsun*vcircsun
omega0 = vcircsun/rsun

# read the simulation data
infile = '../BabaMW/actionsFileID7001.fits'
star_hdus = pyfits.open(infile)
star = star_hdus[1].data
star_hdus.close()

print(' number of input =', len(star['R']))

rgals_all = star['R']

# selection of stars
sindx = np.where((np.fabs(star['z']) < zmaxlim) & \
                 (-star['Jphi']>lzlim) & \
                 (star['R']>rgalmin) & (star['R']<rgalmax))

nstars = len(star['R'][sindx])    

print(' N selected=',nstars)

# for selected stars (positive rotation in clockwise)
vrads = star['vR'][sindx]
vphis = -star['vphi'][sindx]
vz = star['vz'][sindx]
# normalised by rsun and vcirc
jrs = star['Jr'][sindx]/lz0
lzs = -star['Jphi'][sindx]/lz0
jzs = star['Jz'][sindx]/lz0
omegars = star['omegar'][sindx]
omegazs = star['omegaz'][sindx]
omegaphis = -star['omegaphi'][sindx]
ages = star['age'][sindx]
rgals = star['R'][sindx]

# compute the weights
if Rweight==True:
    # R histogram
    rgalpres = star['R'][sindx]
    nhist = 16
    rgalhist, bin_edges = np.histogram(rgals, bins=nhist, \
                                       range=(rgalmin, rgalmax), density=True)
    rgal_bins = 0.5*(bin_edges[:nhist]+bin_edges[1:])

    # compute probability at bins, max one = 1
    nrmax = np.max(rgalhist)
    print(' max at R hist = ', nrmax)
    probs_bins = nrmax/rgalhist

    # probability for all the particles
    probs_all = np.zeros_like(rgals_all)
    probs_all[sindx] = np.interp(rgals_all[sindx], rgal_bins, probs_bins)
    probs = np.interp(rgals, rgal_bins, probs_bins)

    plt.hist(rgals, bins=rgal_bins, histtype='step', color='blue', \
             label='original')
    plt.hist(rgals, bins=rgal_bins, weights=probs, histtype='step', \
             color='red', label='R weighted')

    plt.show()
else:
    probs_all = np.zeros_like(rgals_all)
    probs_all[sindx] = 1.0
    probs = np.zeros_like(rgals)+1.0

# minimum number of stars in each column
nsmin = 25
# set number of grid
ngridx = 250
ngridy = 300
# npline
npline = 1000
                 
# grid plot for Lz vs. Jr
lzrange = np.array([0.0, 1.75])
jrrange = np.array([0.0, 0.2])

#jrticks = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])

# 2D histogram 
H, xedges, yedges = np.histogram2d(lzs, jrs, \
                    bins=(ngridx, ngridy), \
                    range=(lzrange, jrrange), weights=probs)
# set x-axis (lzs) is axis=1
H = H.T
# normalised by row
# print(' hist = ',np.shape(H))
# print(' np row = ',np.sum(H, axis=1))
# log value
nlogmin = 0.25
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

# compute resonance location in action space
# bar pattern speed
omega_b = 40.0
print(' omega_b=',omega_b, omega_b/omega0,' omega0')
domega = 0.02
# pick up stars at CR
mres = 2.0
lres = 0.0
indxcr = np.where(np.fabs((omega_b-omegaphis))<domega)
print(' np CR=', np.shape(indxcr))
# robust fit
ransac_cr = linear_model.RANSACRegressor()
y = lzs[indxcr].reshape(-1, 1)
X = jrs[indxcr].reshape(-1, 1)
ransac_cr.fit(X,  y)
# Predict data of estimated models
line_ycr = np.linspace(jrrange[0], jrrange[1],npline).reshape(-1,1)
line_Xcr = ransac_cr.predict(line_ycr)

# 4:1 resonance
mres = 4.0
lres = 1.0
indx41 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
# print(' np 4:1=', np.shape(indx41))
# robust fit
ransac_41 = linear_model.RANSACRegressor()
y = lzs[indx41].reshape(-1, 1)
X = jrs[indx41].reshape(-1, 1)
ransac_41.fit(X, y)
# Predict data of estimated models
line_y41 = np.linspace(jrrange[0], jrrange[1],npline).reshape(-1,1)
line_X41 = ransac_41.predict(line_y41)

# OLR resonance
mres = 2.0
lres = 1.0
indxolr = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
# print(' np OLR=', np.shape(indxolr))
# robust fit
ransac_olr = linear_model.RANSACRegressor()
y = lzs[indxolr].reshape(-1, 1)
X = jrs[indxolr].reshape(-1, 1)
ransac_olr.fit(X, y)
# Predict data of estimated models
line_yolr = np.linspace(jrrange[0], jrrange[1], npline).reshape(-1,1)
line_Xolr = ransac_olr.predict(line_yolr)

# 1:1 or 4:3 resonance, but the ridges does not show up well. 
mres = 4.0
lres = 3.0
indx43 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
# print(' np 1:1=', np.shape(indxolr))
# robust fit
ransac_43 = linear_model.RANSACRegressor()
y = lzs[indx43].reshape(-1, 1)
X = jrs[indx43].reshape(-1, 1)
ransac_43.fit(X, y)
# Predict data of estimated models
line_y43 = np.linspace(jrrange[0], jrrange[1], npline).reshape(-1,1)
line_X43 = ransac_43.predict(line_y43)

# 1:1 or 4:3 resonance, but the ridges does not show up well. 
mres = 4.0
lres = -1.0
indxi41 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
# print(' np 1:1=', np.shape(indxolr))
# robust fit
ransac_i41 = linear_model.RANSACRegressor()
y = lzs[indxi41].reshape(-1, 1)
X = jrs[indxi41].reshape(-1, 1)
ransac_i41.fit(X, y)
# Predict data of estimated models
line_yi41 = np.linspace(jrrange[0], jrrange[1], npline).reshape(-1,1)
line_Xi41 = ransac_i41.predict(line_yi41)

# 1:1 or 4:3 resonance, but the ridges does not show up well. 
mres = 1.0
lres = 1.0
indx11 = np.where(np.fabs(mres*(omega_b-omegaphis)-lres*omegars)<domega)
# print(' np 1:1=', np.shape(indxolr))
# robust fit
ransac_11 = linear_model.RANSACRegressor()
y = lzs[indx11].reshape(-1, 1)
X = jrs[indx11].reshape(-1, 1)
ransac_11.fit(X, y)
# Predict data of estimated models
line_y11 = np.linspace(jrrange[0], jrrange[1], npline).reshape(-1,1)
line_X11 = ransac_11.predict(line_y11)


# Final plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 16

# log plot
cmin = nlogmin
cmax = np.max(Hlog)
f, (ax1) = plt.subplots(1, sharex = True, figsize=(6,4))
labpos = np.array([5.0, 40.0])
im1 = ax1.imshow(Hlog, interpolation='gaussian', origin='lower', \
        aspect='auto', vmin=cmin, vmax=cmax, \
        extent=[xedges[0], xedges[-1], \
                yedges[0], yedges[-1]], \
           cmap=cm.jet)
ax1.set_xlim(xedges[0], xedges[-1])
ax1.set_ylim(yedges[0], yedges[-1])
ax1.set_xticks([0.5, 1.0, 1.5])    
                 
ax1.set_ylabel(r"J$_{\rm R}$ (L$_{\rm z,0}$)", fontsize=16)
ax1.tick_params(labelsize=16, color='k', direction="in")
# ax2.set_yticks(vrotticks)

plt.xlabel(r"L$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=16)
# plt.ylabel(r"$J_{\rm R}$ ($L_{\rm z,0}$)", fontsize=16)

# plot CR cyan
# plt.scatter(lzs[indxcr], jrs[indxcr], c='blue', marker='.',s=1)
plt.plot(line_Xcr, line_ycr, color='cyan')
# plot 4:1 orange
# plt.scatter(lzs[indx41], jrs[indx41], c='yellow', marker='.',s=1)
plt.plot(line_X41, line_y41, color='orange')
# plot OLR red
# plt.scatter(lzs[indxolr], jrs[indxolr], c='white', marker='.',s=1)
plt.plot(line_Xolr, line_yolr, color='red')
# plot 4:3 green
# plt.scatter(lzs[indx43], jrs[indx43], c='black', marker='.',s=1)
plt.plot(line_X43, line_y43, color='green')
# plot 4:-1 blue
# plt.scatter(lzs[indxi41], jrs[indxi41], c='blue', marker='.',s=1)
plt.plot(line_Xi41, line_yi41, color='blue')
# plot 1:1 grey
# plt.scatter(lzs[indxi41], jrs[indxi41], c='blue', marker='.',s=1)
plt.plot(line_X11, line_y11, color='grey')

f.subplots_adjust(left=0.15, bottom = 0.15, hspace=0.0, right = 0.9)
#cbar_ax1 = f.add_axes([0.8, 0.15, 0.05, 0.725])
#cb1 = f.colorbar(im1, cax=cbar_ax1)
#cb1.ax.tick_params(labelsize=16)

if Paper==True:
    if Rweight==True:
        plt.savefig('lzjr-sim-wRw.eps')
    else:
        plt.savefig('lzjr-sim-woRw.eps')        
    plt.close(f)
else:
    plt.show()

### Analyse Lz distribution at fixed Jr
njrsamp = 2
# select differen Jr sample
# jrsamp_low = np.array([0.05, 0.075, 0.1])
# jrsamp_high = np.array([0.075, 0.01, 0.0125])
# jrsamp_low = np.array([0.1, 0.075, 0.05, 0.025])
# jrsamp_high = np.array([0.15, 0.1, 0.075, 0.05])

jrsamp_low = np.array([0.07, 0.01])
jrsamp_high = np.array([0.15, 0.02])

# lzrange
nhist = 200
lzmin_hist = lzrange[0]
lzmax_hist = lzrange[1]
# kernel size
hlz = 0.03
lz_bins = np.linspace(lzmin_hist, lzmax_hist, nhist)
# y range
ymin_hist = 0.0
ymax_hist = 1.2

# f, ax = plt.subplots(njrsamp, sharex = True, figsize=(5,8))
f, ax = plt.subplots(njrsamp, sharex = True, figsize=(5,5))
f.subplots_adjust(hspace=0.0, bottom=0.15, left=0.2)
# f.subplots_adjust(bottom = 0.15)

jrnorm_all = star['Jr']/lz0

for i in range(njrsamp):
  print(' jr range=', jrsamp_low[i], jrsamp_high[i])
  # selection of stars
  sindx = np.where((np.fabs(star['z']) < zmaxlim) & \
                   (-star['Jphi']>lzlim) & \
                   (jrnorm_all>=jrsamp_low[i]) &
                   (jrnorm_all<jrsamp_high[i]) &
                   (star['R']>rgalmin) & (star['R']<rgalmax))
  #                 (star['age']<agelim))
  print(' N selected stars=', len(star['R'][sindx]))
  # for selected stars (positive rotation in clockwise)
  vrads = star['vR'][sindx]
  vphis = -star['vphi'][sindx]
  vz = star['vz'][sindx]
  # normalised by rsun and vcirc
  jrs = star['Jr'][sindx]/lz0
  lzs = -star['Jphi'][sindx]/lz0
  jzs = star['Jz'][sindx]/lz0
  omegars = star['omegar'][sindx]
  omegazs = star['omegaz'][sindx]
  omegaphis = -star['omegaphi'][sindx]
  probs = probs_all[sindx]
  rgals = star['R'][sindx]  

  # plt.scatter(rgals, probs, marker='.', facecolors='blue', s=1)
  # plt.show()

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
  print(' lz region for 4:1=', cr_low, cr_high)
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

  # 4:-1
  lindx = np.where((line_yi41>jrsamp_low[i]) & (line_yi41<jrsamp_high[i]))
  ri41_low = np.min(line_Xi41[lindx])
  ri41_high = np.max(line_Xi41[lindx])
  if i==0:
      ri41_low0 = ri41_low
      ri41_high0 = ri41_high
  print(' lz region for 4:-1=', ri41_low, ri41_high)
  ax[i].add_patch(
    patches.Rectangle((ri41_low, ymin_hist), ri41_high-ri41_low, \
                      ymax_hist-ymin_hist, facecolor='blue', fill=True,alpha=0.5))

  # 4:3
  lindx = np.where((line_y43>jrsamp_low[i]) & (line_y43<jrsamp_high[i]))
  r43_low = np.min(line_X43[lindx])
  r43_high = np.max(line_X43[lindx])
  if i==0:
      r43_low0 = r43_low
      r43_high0 = r43_high
  print(' lz region for 4:3=', r43_low, r43_high)
  ax[i].add_patch(
    patches.Rectangle((r43_low, ymin_hist), r43_high-r43_low, \
                      ymax_hist-ymin_hist, facecolor='green', fill=True,alpha=0.5))

  # 1:1
  lindx = np.where((line_y11>jrsamp_low[i]) & (line_y11<jrsamp_high[i]))
  r11_low = np.min(line_X11[lindx])
  r11_high = np.max(line_X11[lindx])
  if i==0:
      r11_low0 = r11_low
      r11_high0 = r11_high
  print(' lz region for 1:1=', r11_low, r11_high)
  ax[i].add_patch(
    patches.Rectangle((r11_low, ymin_hist), r11_high-r11_low, \
                      ymax_hist-ymin_hist, facecolor='grey', fill=True,alpha=0.5))
  

  # histogram 
  # ax[i].hist(lzs, bins = lz_bins, fc='#AAAAFF', density=True)
  # KDE
  kde = KernelDensity(kernel='epanechnikov', \
                      bandwidth=hlz).fit(lzs.reshape(-1, 1), sample_weight=probs)
  log_dens = kde.score_samples(lz_bins.reshape(-1, 1))
  ax[i].plot(lz_bins, np.exp(log_dens), color='black')

  # set tick params
  ax[i].tick_params(labelsize=16, color='k', direction="in")
  ax[i].set_xlim(lzmin_hist, lzmax_hist)
  ax[i].set_ylim(ymin_hist, ymax_hist)
  if i==0:
      ax[i].set_ylabel(r"dN($0.07<{\rm J}_{\rm R}<0.15$)", fontsize=14)
  if i==1:
      ax[i].set_ylabel(r"dN($0.01<{\rm J}_{\rm R}<0.02$)", fontsize=14)
      ax[i].set_xticks([0.5, 1.0, 1.5])          
  
plt.xlabel(r"L$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=16)


if Paper==True:
    if Rweight==True:
        plt.savefig('lzhist-sim-wRw.eps')
    else:
        plt.savefig('lzhist-sim-woRw.eps')      
    plt.close(f)
else:
    plt.show()

### Lz-Jr for selected high Jr stars

# selected Jr region
jrmax = jrsamp_high[0]
jrmin = jrsamp_low[0]
# Jz region
jzselmin = 0.005
jzselmax = 0.05

sindx = np.where((np.fabs(star['z']) < zmaxlim) & \
                 (-star['Jphi']>lzlim) & \
                 (jrnorm_all>=jrmin) &
                 (jrnorm_all<jrmax) &
                 (star['R']>rgalmin) & (star['R']<rgalmax))
print('jr range=', jrmin, jrmax, \
      ' N selected stars=', len(star['R'][sindx]))
# for selected stars (positive rotation in clockwise)
vrads = star['vR'][sindx]
vphis = -star['vphi'][sindx]
vz = star['vz'][sindx]
# normalised by rsun and vcirc
jrs = star['Jr'][sindx]/lz0
lzs = -star['Jphi'][sindx]/lz0
jzs = star['Jz'][sindx]/lz0
omegars = star['omegar'][sindx]
omegazs = star['omegaz'][sindx]
omegaphis = -star['omegaphi'][sindx]
probsjrs = probs_all[sindx]

# set number of grid
ngridx = 300
ngridy = 150
# npline
npline = 10
                 
# grid plot for Lz vs. Jr
# lzrange = np.array([0.01, 1.75])
jzrange = np.array([0.0, 0.05])

#jrticks = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])

# 2D histogram 
H, xedges, yedges = np.histogram2d(lzs, jzs, \
                    bins=(ngridx, ngridy), \
                    range=(lzrange, jzrange), weights=probsjrs)
# set x-axis (lzs) is axis=1
H = H.T
# normalised by row
# minimum number of stars in each column
# nsmin = 25
# log value
nlogmin = 0.0
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
jzindx = np.where((jzs>jzselmin) & (jzs<jzselmax))
print('jz range=', jzselmin, jzselmax, \
      ' N selected stars=', len(jzs[jzindx]))
# for selected stars (positive rotation in clockwise)
lzsels = lzs[jzindx]
probs = probsjrs[jzindx]

# KDE
hlz = 0.03
nhist = 200
lz_bins = np.linspace(lzrange[0], lzrange[1], nhist)
kde = KernelDensity(kernel='epanechnikov', \
                      bandwidth=hlz).fit(lzsels.reshape(-1, 1), sample_weight=probs)
log_dens = kde.score_samples(lz_bins.reshape(-1, 1))
# y range
ymin_hist = 0.01
ymax_hist = 1.5

# Final plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 16

# log plot
cmin = nlogmin
cmax = np.max(Hlog)
# cmin = 0.0
# cmax = np.max(H)*0.5
f, (ax1,ax2) = plt.subplots(2, sharex = True, figsize=(6,5), gridspec_kw={'height_ratios' : [1, 2]})
f.subplots_adjust(hspace=0.0)

# KDE histogram
ax1.plot(lz_bins, np.exp(log_dens), color='black')
ax1.tick_params(labelsize=16, color='k', direction="in")
ax1.set_xlim(lzrange[0], lzrange[1])
ax1.set_ylim(ymin_hist, ymax_hist)
ax1.set_ylabel(r"dN", fontsize=16)
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
                      ymax_hist-ymin_hist, facecolor='blue', fill=True,alpha=0.5))
ax1.add_patch(
    patches.Rectangle((r11_low0, ymin_hist), r11_high0-r11_low0, \
                      ymax_hist-ymin_hist, facecolor='grey', fill=True,alpha=0.5))

# Lz vs. Jz
ax2.add_patch(
    patches.Rectangle((lzrange[0], jzselmin), \
                      lzrange[1]-lzrange[0], jzselmax-jzselmin, \
                      facecolor='pink', fill=True,alpha=0.3))

#im1 = ax1.imshow(H, interpolation='gaussian', origin='lower', \
im1 = ax2.imshow(Hlog, interpolation='gaussian', origin='lower', \
        aspect='auto', vmin=cmin, vmax=cmax, \
        extent=[xedges[0], xedges[-1], \
                yedges[0], yedges[-1]], \
                 cmap='Greens')
# ax1.scatter(lzs, jzs, marker='.', facecolors=cc, s=1)
ax2.set_xlim(xedges[0], xedges[-1])
ax2.set_ylim(yedges[0], yedges[-1])
                 
ax2.set_ylabel(r"J$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=16)
ax2.tick_params(labelsize=16, color='k', direction="in")
ax1.set_xticks([0.5, 1.0, 1.5])    
# ax2.set_yticks(vrotticks)

ax2.plot((lzrange[0],jzselmin),(lzrange[1],jzselmax), color='black')

plt.xlabel(r"L$_{\rm z}$ (L$_{\rm z,0}$)", fontsize=16)
# plt.ylabel(r"$J_{\rm R}$ ($L_{\rm z,0}$)", fontsize=16)

f.subplots_adjust(left=0.15, bottom = 0.15, hspace=0.0, right = 0.9)
#cbar_ax1 = f.add_axes([0.8, 0.15, 0.05, 0.725])
#cb1 = f.colorbar(im1, cax=cbar_ax1)
#cb1.ax.tick_params(labelsize=16)

if Paper==True:
    if Rweight==True:
        plt.savefig('lzjz-sim-wRw.jpg')
    else:
        plt.savefig('lzjz-sim-woRw.jpg')
    plt.close(f)
else:
    plt.show()


