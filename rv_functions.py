import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft
import matplotlib.pyplot as plt
from  astropy.table import Table
from astropy.io import fits
import astropy.constants as c
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from numpy.polynomial.polynomial import polyval, polyfit

# PRO
# compute_xcor_rv, wave, $  # wavelength array
# flux, $  # flux array
# spt, $  # spectral type for auto-find template. This requires other programs and data that are not on github... so ignore this
# rv, $  # output RV
# wavelim = wavelim, $  # 2 element array specifying the wavelength region to search around
# norm_reg = norm_reg, $  # 2 element array specifying the normalization region
# twave = twave, $  # Template wavelength array. Supplying this will cause the program to ignore the spt variable
# tflux = tflux, $  # template flux
# maxshift = maxshift, $  # maximum allowed shift (IN PIXELS)
# startshift = startshift, $  # initial shift, sometimes we use this for applying barycentric and other corrections to the data before doing the RV, but I suggest letting it sit at 0 (the default)
# showplot = showplot, $  # plot the chi^2 distribution as a function of the offset above the spectrum and shifted spectrum
# redo = redo, $  # set to 1 to turn off the redo module
# sft = sft, xp = xp, chi = chi, minvel = minvel, normalize = normalize

def compute_xcor_rv(wave, flux, spt, wavelim=[5300, 8500], norm_reg=[7150, 7350], twave=None, tflux=None, maxshift=300,
					startshift=0, showplot = False, redo=False, normalize='median'):
	# Eric J. Hilton
	# modified 20120307
	# enable user to manually adjust fit a bit, so see if it locks on to a
	# better answer
	# also allows for more plotting

	# 20111015
	# compute the radial velocity using cross correlation to templates.
	# this pro is mostly general, although is meant for UH22 SNIFS or MDM spectra.
	# (can be easily adapted for other sets)

	# inputs: wave (in linear dispersion), flux, spectral type
	# keywords:
	# wavelim = [x1,x2] is the wavelength region used for the correlation (in angstroms)
	# default is 5300, 8500]
	# norm_reg = [x1,x2] is wavelength region used to normalize the
	# spectrum and the template spectrum (in angs) default is [6450,6650]
	# twave and tflux - if present, then don't read in the template
	# this allows for faster computation. the wrapper can read them all in
	# once, and just pass the relevant one when necessary.
	# note that in this case, the spt variable is not necessary

	oldwave = np.copy(wave)
	oldflux = np.copy(flux)

	c = 299792.458  # speed of light km/s

	wave = np.divide(wave, (startshift / c + 1.))  # Apply the initial shift

	## Read in Bochanski's template
	if tflux is None: twave, tflux = load_template(spt)

	## normalize both template and spectrum
	idx_t = [True if norm_reg[0] < x < norm_reg[1] else False for x in twave]
	idx = [True if norm_reg[0] < x < norm_reg[1] else False for x in wave]

	if normalize == 'mean':
		# Divide the template flux and the object flux by the mean within the normalization region
		tflux = tflux / np.mean(tflux[idx_t])
		flux = flux / np.mean(flux[idx])
	else:
		# Divide the template flux and the object flux by the median within the normalization region
		tflux = tflux / np.median(tflux[idx_t])
		flux = flux / np.median(flux[idx])

	# Ensure that the template is wider than the object
	if np.min(twave) < np.min(wave):
		temp_min_i = np.where(twave < np.min(wave))[0][-1]  # index of the 1st temp wv less than the min object wavelength
		wv_min_i = 0
	else:
		temp_min_i = 0
		wv_min_i = np.where(wave > np.min(twave))[0][
			0]  # index of the 1st object wv greater than the min temp wavelength

	if np.max(twave) > np.max(wave):
		temp_max_i = np.where(twave > np.max(wave))[0][
			0]  # index of the 1st temp wv larger than the max object wavelength
		wv_max_i = len(wave) - 1
	else:
		temp_max_i = len(twave) - 1
		wv_max_i = np.where(wave < np.max(twave))[0][-1]

	ind = [True if wv_min_i <= i <= wv_max_i else False for i in range(len(wave))]
	ind2 = [True if temp_min_i <= i <= temp_max_i else False for i in range(len(twave))]

	# CREATE WAVELENGTH ARRAY THAT IS EVENLY SPACED IN LOG SPACE
	n_bins = len(wave[ind])
	ln_wave = np.log(wave[ind])
	ln_range = np.max(ln_wave) - np.min(ln_wave)
	#ln_grid = findgen(n_bins) * ln_range / (n_bins - 1) + min(ln_wave)
	ln_grid = np.array(range(n_bins)) * ln_range / (n_bins -1) + np.min(ln_wave)
	newwave = np.exp(1) ** ln_grid
	newwave = newwave[1:-1]  # Cut off the first and last pixel of the new wave

	# Interpolate the flux onto the new evenly space wavelength array
	f_newflux = interp1d(twave[ind2], tflux[ind2])
	tnewflux = f_newflux(newwave)

	f_newflux = interp1d(wave[ind], flux[ind])
	newflux = f_newflux(newwave)

	# Find the approximate velocity per pixel
	if len(newwave)%2 == 0:
		# Even, take the number closest to the median
		ind_med = np.argmin(abs(newwave - np.median(newwave)))  #np.where(newwave == np.median(newwave))
	elif len(newwave)%2 == 1:
		# Odd, take the median
		ind_med = [i for i in range(len(newwave)) if newwave[i] == np.median(newwave)][0]  #np.where(newwave == np.median(newwave))
	velpix = c * (newwave[ind_med] - newwave[ind_med - 1]) / newwave[ind_med - 1]

	#minvel = min([velpix, velpix])
	minvel = np.min(velpix)

	# Do the actual cross correlation
	sft = xcorl(newflux, tnewflux, maxshift)
	print(sft)  # Wavelength Shift
	rv = sft * minvel
	if redo and abs(rv) > 500:
		# If the returned RV is too large, redo  the calculation on a different region of wavelength space
		nwavelim = [5300, 7000]
		sft = xcorl(newflux, tnewflux, maxshift)
		rv = sft * minvel
		if abs(rv) > 500:
			# If it's still too large try one more time
			nwavelim = [7700, 9000]
			sft  = xcorl(newflux, tnewflux, maxshift)
			rv = sft * minvel
	if abs(rv) > 1000:
		print('ERROR: |RV| > 1000km/s')
		rv = 0.0  # do no harm algorithm

	# Make plot
	if showplot:
		plt.figure()
		plt.plot(newwave, tnewflux, label='Template')
		plt.plot(newwave, newflux, c='red', label='Object')
		plt.plot(newwave/(rv / c + 1.), newflux, c='blue', label='Shifted Object')
		plt.legend(ncol=3, loc=(0.0, 1.01))
		plt.show()

	wave = oldwave
	flux = oldflux

	return rv

def xcorl(star, temp, maxshift, mult=False, *args, **kwargs):
	#12-Jun-92 JAV	Added minchi parameter and logic.
	#17-Jun-92 JAV	Added "Max. allowable range" error message.
	#24-Aug-92 JAV 	Supressed output of blank line when print keyword absent.
	#3-Jan-97 GB 	Added "fine" (# pixs finely resolved) and "mult" options
	#  these give finer resolution to the peak, and use products instead of diffs.
	#8-Jan-97 GB 	Added "fshft" to force one of a double peak
	#23-Oct-01 GB 	Added /full keyword to simplify the call
	#28-Feb-13 CAT 	Ported to Python
	#16-Jun-16 AYK 	Added to hammer code

	# Set the defaults
	pr = 0
	fine = False
	fshft = 0
	full = 0
	ff = 0

	# Read the arguments
	for arg in args:
		if arg.lower() == 'fine': fine = 1
		if arg.lower() == 'full': full = 1

	# Read the keywords
	for key in kwargs:
		if key.lower() == 'mult':
			mult = kwargs[key]
		if key.lower() == 'fshft':
			fshft = kwargs[key]

	ln = len(temp)
	ls = len(star)
	length = np.min([ln, ls])  # The length will be the length of the shorter one, template or object
	slen = length
	if maxshift > (length-1)/2:
		print('Maximum allowable shift for this case is' + str((length-1)/2))
		maxshift = (length-1)/2
	newln = length - 2*maxshift  # Leave "RANGE" on ends for overhang.

	# Normalize template and object spectrum
	norm_temp = temp/(np.sum(temp)/ln)
	st = star/(np.sum(star)/ls)

	newend = maxshift + newln - 1
	x = np.arange(2 * maxshift + 1) - maxshift
	chi = np.zeros(2 * maxshift + 1)

	if full == 1:
		pr=1

	for j in range(-maxshift, maxshift+1):  # Goose step, baby
		if mult:
			dif = norm_temp[maxshift:newend+1] * st[maxshift+j:newend+j+1]
			chi[j+maxshift] = np.sum(abs(dif))
		else:
			dif = norm_temp[maxshift:newend+1] - st[maxshift+j:newend+j+1]  # Difference between the template and the shifted spectrum
			chi[j+maxshift] = np.sum(dif*dif)
	xcr = chi

	length = len(x) * 100
	xl = np.arange(length)
	xl = xl/100. - maxshift
	xp = xl[0:length-99]
	function2 = interp1d(x, chi, kind='cubic')
	cp = function2(xp)
	if mult:
		minchi = np.max(cp)
		mm = np.argmax(cp)
	else:
		minchi = np.min(cp)
		mm = np.argmin(cp)
	shft = xp[mm]

	if pr != 0:
		print( 'XCORL: The shift is: %10.2f'%(shft))
	if abs(shft) > maxshift:
		ff=1
		return
	if fshft != 0:
		shft = fshft

	if fine:
		nf = fine*20+1
		rf = fine*10.
		nc = -1
		fchi = np.zeros(nf)
		xl = np.zeros(nf)
		for j in range(int(-rf), int(rf+1)):
			xl[nc+1] = shft + j/10.
			nst = shfour(st, -xl[nc+1])
			nc += 1
			if mult == 1:
				dif = nst[maxshift:newend+1] * norm_temp[maxshift:newend+1]
				fchi[nc] = np.sum(abs(dif))
			else:
				dif = nst[maxshift:newend+1] - norm_temp[maxshift:newend+1]
				fchi[nc] = np.sum(np.real(dif*dif))
		xp = np.arange((nf-1) * 100 + 1) / 1000. + shft - fine
		function3 = interp1d(xl, fchi, kind='cubic')
		cp = function3(xp)
		if mult == 1:
			minchi = np.max(cp)
			mm = np.where(cp == minchi)
		else:
			minchi = np.min(cp)
			mm = np.where(cp == minchi)
		fshft = xp[mm]

		if pr != 0:
			print( 'XCORL: The final shift is: %10.2f'%(fshft))
	else:
		fshft = shft
	shft = fshft

	return shft

def shfour(sp, shift, *args):
	# Shift of sp by (arbitrary, fractional) shift, result in newsp

	# Set Defaults
	pl = 0

	# Read the arguments
	for arg in args:
		if arg.lower() == 'plot': pl = 1

	ln = len(sp)
	nsp = sp

	# Take the inverse Fourier transform and multiply by length to put it in IDL terms
	fourtr = np.fft.ifft(nsp) * len(nsp)
	sig = np.arange(ln)/float(ln) - .5
	sig = np.roll(sig, int(ln/2))
	sh = sig*2. * np.pi * shift

	count=0
	shfourtr = np.zeros( (len(sh), 2) )
	complexarr2 = np.zeros( len(sh), 'complex' )
	for a,b in zip(np.cos(sh), np.sin(sh)):
		comps = complex(a,b)
		complexarr2[count] = comps
		count+=1

	shfourtr = complexarr2 * fourtr

	# Take the Fourier transform
	newsp = np.fft.fft(shfourtr) / len(shfourtr)
	newsp = newsp[0:ln]

	# Plot it
	if pl == 1:
		plt.plot(sp)
		plt.plot(newsp-.5)
		plt.show()

	return newsp

def load_template(spt, active=False):
	if spt == 'K5':
		# filename = '/Users/woodml/Ragnarok Dropbox/Mackenna Wood/Full_set/1787492i copy.ascii'
		# temp_rv = 22.19 # km/s
		# format = 'ascii'
		filename = '/Users/woodml/Ragnarok Dropbox/Mackenna Wood/ESO_spectra/HIP100223_HARPS/ADP.2014-09-25T15.36.35.423.fits'
		temp_rv = 32.8  # km/s
		format = 'fits'
		# HD 157881
	elif spt == 'M0' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m0.nactive.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M0' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m0.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M1' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m1.all.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M1' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m1.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M2' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m2.nactive.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M2' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m2.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M3' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m3.nactive.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M3' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m3.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M4' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m4.nactive.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M4' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m4.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s
	elif spt == 'M5' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m5.nactive.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0  # km/s
	elif spt == 'M5' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m5.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0  # km/s
	elif spt == 'M6' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m6.nactive.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0  # km/s
	elif spt == 'M6' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m6.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0  # km/s
	elif spt == 'M7' and not active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m7.nactive.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0  # km/s
	elif spt == 'M7' and active:
		filename = '/Users/woodml/Documents/Models/Bochanski_templates/m7.active.ha.na.k.fits'
		format = 'fits model'
		temp_rv = 0.0 # km/s

	if format == 'fits':
		hdul = fits.open(filename)
		temp_flux = np.array(Table(hdul['SPECTRUM'].data)['FLUX'][0])
		temp_wv = np.array(Table(hdul['SPECTRUM'].data)['WAVE'][0])

		print(len(temp_wv), len(temp_flux))

	elif format == 'fits model':
		hdul = fits.open(filename)
		temp_wv_vac = np.arange(3825, 3825+53750/10, 0.1)
		temp_flux = hdul['PRIMARY'].data[1]

		# Convert to in air wavelengths
		s = np.divide(10**4, temp_wv_vac)
		n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
		temp_wv = np.divide(temp_wv_vac, n)
	elif format=='ascii':
		t = Table.read(filename, format='ascii', delimiter=' ', comment='#')
		temp_wv = t['Wave(A)']
		temp_flux = t['Flux']

	return temp_wv, temp_flux, temp_rv

def find_rv(wv, flux, spt, ra, dec, obs_date, plots=False):
	# Read in template, correct to rest wavelength
	sdss_wv, sdss_flux, temp_rv = load_template(spt, active=False)
	sdss_wv = np.divide(sdss_wv, (1 + temp_rv / c.c.to('km/s').value))  # Shift template to rest wavelength

	# Cut to common wavelengths
	min_wv = np.max([np.min(sdss_wv), np.min(wv)])
	max_wv = np.min([np.max(sdss_wv), np.max(wv)])

	# The template wavelengths need to be wider than the object to  run the interpolation
	if np.min(sdss_wv) < np.min(wv):
		temp_min_i = np.where(sdss_wv < np.min(wv))[0][
			-1]  # index of the 1st temp wv less than the min object wavelength
		wv_min_i = 0
	else:
		temp_min_i = 0
		wv_min_i = np.where(wv > np.min(sdss_wv))[0][
			0]  # index of the 1st object wv greater than the min temp wavelength

	if np.max(sdss_wv) > np.max(wv):
		temp_max_i = np.where(sdss_wv > np.max(wv))[0][
			0]  # index of the 1st temp wv larger than the max object wavelength
		wv_max_i = len(wv) - 1
	else:
		temp_max_i = len(sdss_wv) - 1
		wv_max_i = np.where(wv < np.max(sdss_wv))[0][-1]

	idx = [True if wv_min_i <= i <= wv_max_i else False for i in range(len(wv))]
	temp_idx = [True if temp_min_i <= i <= temp_max_i else False for i in range(len(sdss_wv))]

	print('Aligning Template')
	# Align template with data
	f_temp = interp1d(sdss_wv[temp_idx], sdss_flux[temp_idx])
	aligned_sdss = f_temp(wv[idx])

	print('Correcting Flux')
	# Now get new ratio
	ratio = flux[idx] / aligned_sdss

	smooth_ratio = polyval(wv, polyfit(wv[idx], ratio, 3))
	if all(np.isnan(smooth_ratio)):
		quarter = int(len(ratio) / 4)
		ys = [np.nanmedian(ratio[:quarter]), np.nanmedian(ratio[quarter:2 * quarter]),
			  np.nanmedian(ratio[2 * quarter:3 * quarter]), np.nanmedian(ratio[3 * quarter:])]
		xs = [np.nanmedian(wv[idx][:quarter]), np.nanmedian(wv[idx][quarter:2 * quarter]),
			  np.nanmedian(wv[idx][2 * quarter:3 * quarter]), np.nanmedian(wv[idx][3 * quarter:])]
		smooth_ratio = polyval(wv, polyfit(xs, ys, 1))

	# The ratio tends to go crazy in regions not covered by the template, so we replace those with the median value
	smooth_ratio = [smooth_ratio[i] if wv[i] < np.max(sdss_wv) else np.median(ratio[int(len(ratio) / 2):]) for i in
					range(len(wv))]
	smooth_flux = np.divide(flux, smooth_ratio)

	# Mask Halpha
	print('Masking Halpha')
	for i in [i for i in range(len(wv)) if 6540 < wv[i] < 6590]:
		smooth_flux[i] = np.median(smooth_flux[i - 30:i - 1])

	# Replace zeros
	smooth_flux = np.array([smooth_flux[i] if smooth_flux[i] != 0. else np.median(
		smooth_flux[i - 20:i + 20][smooth_flux[i - 20:i + 20] > 0.]) for i in range(len(smooth_flux))])

	rv = compute_xcor_rv(wv, smooth_flux, spt, wavelim=[min_wv, max_wv], norm_reg=[6500, 7500], twave=np.copy(sdss_wv),
						 tflux=np.copy(sdss_flux), maxshift=300)
	# rv = -100

	# Correct Wavelength
	shifted_wv = np.divide(wv, (1 + rv / c.c.to('km/s').value))

	# Calculate Barycentric Correction
	soar = EarthLocation.of_site('Cerro Pachon')  # HARPS - La Silla Observatory, SOAR - Cerro Pachon
	sc = SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'degree'))
	time = Time(obs_date)
	print('Object Coordinates: ', sc.to_string())
	print('Date:', time)
	bary = sc.radial_velocity_correction(obstime=time, location=soar).to('km/s').value
	print('Barycentric Correction: ', bary)
	final_rv = rv + bary + rv * bary / c.c.to('km/s').value
	print('Final RV: ', final_rv)

	if plots:
		plt.figure()
		plt.plot(sdss_wv, sdss_flux, label='Template')
		plt.plot(wv, smooth_flux, label='Original Object')
		plt.plot(shifted_wv, smooth_flux, label='Corrected Object')
		plt.legend()
		plt.show()
	return final_rv


def correct_spectrum(wv, flux, rv, out_file, hdr):
	# Given a radial velocity, shift a spectrum into the rest frame and write the results to a fits file
	# Inputs:
	#	- wv: wavelength array (uncorrected)
	#	- flux: flux array
	#	- rv: Barycentric corrected radial velocity in km/s
	#	- out_file: file path to write corrected spectra to
	# Outputs:
	#	None

	# Correct Spectrum
	shifted_wv = np.divide(wv, (1 + rv / c.c.to('km/s').value))

	# Write Out
	hdu_spec = fits.BinTableHDU.from_columns([fits.Column(name='wv(Ang)', format='E', array=shifted_wv), fits.Column(name='Counts', format='E', array=flux)])
	hdu_spec.name = 'SPECTRUM'
	header = hdr
	empty_primary = fits.PrimaryHDU(header=hdr)

	hdul = fits.HDUList([empty_primary, hdu_spec])
	hdul.writeto(out_file, overwrite=True)
	print('Rest Frame Corrected Spectrum Written To:', out_file)
	return shifted_wv
