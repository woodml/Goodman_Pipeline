import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import reduce_and_extract as rp

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Georgia']
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.linewidth'] = 1

date = '20210317'
target = 'TIC311449078'
folder = 'SOAR:Goodman/Raw/' + date
print(folder)

rp.clean_folder(folder)
wv, spec, header = rp.reduce_and_extract(folder, target, 'dome1200', arc_id='97') # , exclude=['20230201/0114_30Gaia5294658.fits'])

# Save the spectrum to a fits file
out_file = '~SOAR:Goodman/Reduced/' + date + '/' + target + '_spectra.fits'
# spectrum
hdu_spec = fits.BinTableHDU.from_columns([fits.Column(name='wv(Ang)', format='E', array=wv), fits.Column(name='Counts', format='E', array=spec)])
hdu_spec.name = 'SPECTRUM'
# header
hdr = header
empty_primary = fits.PrimaryHDU(header=hdr)

hdul = fits.HDUList([empty_primary, hdu_spec])
print('\nMy Structure Info:\n')
hdul.info()
hdul.writeto(out_file, overwrite=True)

# Create plots of various regions of the spectrum
# Full Spectrum
out_file = '~/SOAR:Goodman/Reduced/' + date + '/' + target + '_spectra.pdf'
plt.figure(figsize=(10, 4))
plt.step(wv, spec)
# plt.vlines([6562.8, 6708, 7126], 0, 500, color='k', ls='--')
plt.xlim(6250, 7550)
plt.xlabel('wavelength (\u212B)')
plt.savefig(out_file, bbox_inches='tight', pad_inches=0.2)

# Halpha
out_file = '~/SOAR:Goodman/Reduced/' + date + '/' + target + '_spectra_ha.pdf'
plt.figure(figsize=(10, 4))
plt.plot(wv, spec)
plt.minorticks_on()
plt.vlines([6562.8, 6708, 7126], 0, 3050, color='k', ls='--')
plt.xlim(6500, 6600)
# plt.ylim(0, 4050)
plt.xlabel('wavelength (\u212B)')
plt.savefig(out_file, bbox_inches='tight', pad_inches=0.2)

# Li
out_file = '~/SOAR:Goodman/Reduced/' + date + '/' + target + '_spectra_li.pdf'
plt.figure(figsize=(10,4))
plt.plot(wv, spec)
plt.vlines([6562.8, 6708, 7126], 0, 1000, color='k', ls='--')
plt.minorticks_on()
plt.xlim(6690, 6720)
# plt.ylim(0, 200)
plt.xlabel('wavelength (\u212B)')
plt.savefig(out_file, bbox_inches='tight', pad_inches=0.2)

# TiO
out_file = '~/SOAR:Goodman/Reduced/' + date + '/' + target + '_spectra_tio.pdf'
plt.figure(figsize=(10, 4))
plt.plot(wv, spec)
plt.vlines([6562.8, 6708, 7126], 0, 1000, color='k', ls='--')
plt.minorticks_on()
plt.xlim(7050, 7150)
# plt.ylim(0, 1000)
plt.xlabel('wavelength (\u212B)')
plt.savefig(out_file, bbox_inches='tight', pad_inches=0.2)
plt.show()
