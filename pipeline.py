import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import reduce_and_extract as rp
import spectral_typing as spt
import rv_functions as rv
import EqW as eqw

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Georgia']
rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.linewidth'] = 1

# Inputs
date = '20210317'
target = 'TIC309818851'
folder = 'SOAR:Goodman/Raw/' + date
n_obs = 5

# Reduce & Extract
rp.clean_folder(folder)
wv, spec, header = rp.reduce_and_extract(folder, target, 'dome1200', arc_id='145')

# Save data
out_file = '/Users/woodml/Observing/SOAR:Goodman/Reduced/' + date + '/' + target + '_spectra.fits'
# spectrum
hdu_spec = fits.BinTableHDU.from_columns([fits.Column(name='wv(Ang)', format='E', array=wv), fits.Column(name='Counts', format='E', array=spec)])
hdu_spec.name = 'SPECTRUM'
empty_primary = fits.PrimaryHDU(header=header)

hdul = fits.HDUList([empty_primary, hdu_spec])
hdul.writeto(out_file, overwrite=True)

# Spectral Type
temps = spt.load_spt_templates()

best_spt = spt.find_spt(wv, spec/np.median(spec), temps)
print('Spectral Class: ', best_spt)

# Plot Full Spectrum
out_file = '/Users/woodml/Observing/SOAR:Goodman/Reduced/' + date + '/' + target + '_spectra.pdf'
plt.figure(figsize=(10, 4))
plt.step(wv, spec)
# plt.vlines([6562.8, 6708, 7126], 0, 500, color='k', ls='--')
plt.xlim(6250, 7550)
plt.xlabel('wavelength (\u212B)')
plt.text(7400, 0.9*np.max(spec), best_spt)
plt.savefig(out_file, bbox_inches='tight', pad_inches=0.2)
plt.show()

# Get Radial Velocity
final_rv = rv.find_rv(wv, spec, best_spt, header['RA'], header['DEC'], header['DATE-OBS'])

# Rest Frame
out_file = '/Users/woodml/Observing/SOAR:Goodman/Reduced/' + date + '/' + target + '_restspectra.fits'
shifted_wv = rv.correct_spectrum(wv, spec, final_rv, out_file, header)

# Measure EqW
e_flux = np.divide(np.sqrt(spec), np.sqrt(n_obs))

perturbed_ews = []
for x in range(500):
	np.random.seed()
	perturb = [-1 if np.random.rand(1) <= 0.5 else 1 for x in spec]
	perturb = np.array(np.multiply(e_flux, perturb))
	ew = eqw.measure_ew(wv, spec+perturb, [6702, 6713], [6706., 6709.], showplot=False, quiet=True)
	perturbed_ews.append(ew)

print('EW(Li) = ', np.median(perturbed_ews)*1000, ' ± ', np.std(perturbed_ews)*1000)
print('SNR:', np.median(np.divide(spec, e_flux)))

plt.figure()
plt.hist(perturbed_ews)
plt.show()


