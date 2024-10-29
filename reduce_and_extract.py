# Reduction Pipeline SOAR Goodman  1200 m5 mode
# Mackenna Wood

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
from astropy.io import ascii
import astropy.constants as c
from astropy.table import Table, Column
import scipy
from scipy.signal import find_peaks
from astroscrappy import detect_cosmics
from time import sleep
from rv_functions import compute_xcor_rv

# from matplotlib import rcParams
rcParams['figure.figsize'] = (6,4)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Georgia']

# Functions
def quickplot_file(file):
	# Shows a grayscale version of the .fits file given by file
	hdul = fits.open(file)
	plt.figure()
	plt.imshow(hdul[0].data, cmap='gray', vmax=100)
	plt.colorbar()
	plt.show()
	hdul.close()
	return

def quickplot_data(data, xlims=None, **kwargs):
	# Shows a grayscale version of the .fits file given by file
	plt.figure()
	plt.imshow(data, cmap='gray', **kwargs)
	plt.colorbar()
	if xlims is not None:
		plt.xlim(xlims)
	plt.show()
	return

def gauss(x, *p):
	A, mu, sigma = p
	return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

def normalize(v):
	# normalizes every item in a vector by dividing each by the sum of all
	total = sum(v)
	return np.divide(v, total)

def get_data(file_list):
	# Quick function to get the data arrays and headers from a list of files
	# Inputs: list of filenames
	# Outputs:  list of data arrays, list of headers
	all_data = []
	all_header = []
	for file in file_list:
		data, header = fits.getdata(file, header=True)
		all_data.append(data[:, :])  # add the image to the stack of images
		all_header.append(header)

	return all_data, all_header

def median_stack(data_arrays, example_header):
	# Stacks and return the median of all the data arrays passed in
	#  Does not open or write files
	# Inputs:
	#   data_arrays - list of the arrays of data
	# Outputs:
	#   data and header for median stacked object
	# 1. Create new header
	new_header = example_header
	# 2. Find the median of each pixel in the row
	median = np.median(data_arrays, axis=0)
	# Update header
	new_header["COMMENT"] = "Median stacked"
	return median, new_header

def overscan_subtraction(data_arrays, headers, comment=None):
	# Removes the overscan region and overscan constant from input images. Uses overscan values for Goodman 1200 m5
	# Inputs:
	#   data arrays - list of arrays containing image data
	#   header - list of header files of the images
	#   comment - string comment to add to headers
	# Outputs:
	#   new_arrays - list of arrays containing image data with overscan removed. Note: size of arrays will be different
	#   new_headers - list of updated headers. if comment is None, headers will be the same as input
	new_arrays = []
	new_headers = []
	for i in range(len(data_arrays)):
		# 1. Get individual data
		data = data_arrays[i]
		header = headers[i]
		# 2. Get overscan constant
		overscan = data[1:200, 5:50]
		median_overscan = np.median(overscan)
		# 3. Remove overscan region
		new_data = data[0:200, 51:]
		# 4. Subtract overscan constant
		new_data = (new_data - median_overscan)
		# 5. Update header
		new_header = header
		if comment is not None:
			new_header["COMMENT"] = comment
		# 6. Add to result lists
		new_arrays.append(new_data)
		new_headers.append(new_header)

	return new_arrays, new_headers

def subtraction(data_arrays, data_headers, calibration_data, calibration_header, comment=None, scale=False):
	# Subtracts a calibration array from a data array, pixel by pixel
	# Inputs: list of data arrays; list of data headers; calibration array; calibration header; comment to add to data
	#         headers; boolean to turn on/off time scaling
	# Outputs: list of data arrays, with the calibration subtracted; list of data headers, with comment added
	new_data_arrays = []
	new_data_headers = []
	for i in range(len(data_arrays)):
		data = data_arrays[i]
		header = data_headers[i]

		# Get time scaling factor
		t_scale = 1
		if scale:
			t_raw = float(header["EXPTIME"])
			t_cal = float(calibration_header["EXPTIME"])
			t_scale = t_raw / t_cal

		# Subtract
		new_data = (data - calibration_data) * t_scale
		# Update header
		new_header = header
		if comment is not None:
			new_header["COMMENT"] = str(new_header["COMMENT"]) + comment

		new_data_arrays.append(new_data)
		new_data_headers.append(new_header)

	return new_data_arrays, new_data_headers

def division(data_arrays, data_headers, calibration_data, calibration_header, comment=None):
	# inputs:
	#   files - list of files to process
	#   calibration_image - .fits file with calibration image to divide by
	#   suffix - suffix to be added to end of processed image files (includes .fits)
	#   comment - comment to be added to header files
	new_data_arrays = []
	new_data_headers = []
	for i in range(len(data_arrays)):
		data = data_arrays[i]
		header = data_headers[i]
		# Divide
		new_data = (data / calibration_data)
		# update header
		new_header = header
		if comment is not None:
			new_header["COMMENT"] = str(new_header["COMMENT"]) + comment

		new_data_arrays.append(new_data)
		new_data_headers.append(new_header)
	return new_data_arrays, new_data_headers

def arc_extraction(data, header):
	x_size = header['NAXIS1']
	y_size = header['NAXIS2']
	x_pixels = np.arange(0, x_size)
	# a. Locate object spectra
	# assumes the brightest row in a column is the location of the object spectra in that column
	brightest_row = []
	# Find the brightest row in each column
	for i in range(x_size):
		col = list(data[:, i])
		brightest_row.append(col.index(max(col)))

	# Iteratively fit spectrum location w/ 3rd order polynomial
	outliers = []
	new_outliers = list(range(0, 100))  # initialize outliers with a list of 0-100
	clean_indices = list(range(0, x_size))  # initialize clean columns list with all columns
	i_count = 0
	while new_outliers != outliers and i_count < 20:
		i_count += 1
		[a, b, c] = np.polyfit([x_pixels[i] for i in clean_indices], [brightest_row[i] for i in clean_indices], 2)
		ys = a * x_pixels ** 2 + b * x_pixels + c
		std_dev = np.std(ys)

		outliers = new_outliers
		new_outliers = [i for i in range(0, x_size) if abs(brightest_row[i] - ys[i]) / std_dev > 16]
		clean_indices = [x for x in np.arange(0, x_size) if x not in new_outliers]
	# Store position of pixels
	spectrum_loc = ys

	# Extract standard spectrum
	aperture = 5
	std_spectrum = [np.sum(data[int(spectrum_loc[x] - aperture):int(spectrum_loc[x] + aperture), x], 0) for x in x_pixels]

	return std_spectrum

def choose_peaks(image_lines, template_lines, tol=10):
	# Match the lines in image to the lines in template
	# I need to have a match for every template line, but do not need a match for every image line
	select_peaks = []
	missing_peaks = []
	matched_lines = []
	select_template = []

	for line in template_lines:
		dist = [abs(line - i_line) for i_line in image_lines]
		if min(dist) < tol:
			if image_lines[np.argmin(dist)] not in select_peaks:
				i_line = image_lines[np.argmin(dist)]
				dist_red = [abs(i_line - t_line) for t_line in template_lines]
				if template_lines[np.argmin(dist_red)] != line:
					missing_peaks.append(line)
				else:
					select_peaks.append(i_line)
					matched_lines.append(line)
			else:
				missing_peaks.append(line)
		else:
			missing_peaks.append(line)

	return matched_lines, select_peaks, missing_peaks

def cross_correlate(series1, series2, max_delay):
	N = len(series1)
	m1 = np.mean(series1)
	m2 = np.mean(series2)

	# Calculate the denominator
	s1 = sum([np.square(series1[i] - m1) for i in range(N)])
	s2 = sum([np.square(series2[i] - m2) for i in range(N)])
	denom = np.sqrt(s1 * s2)

	# Calculate the correlation series
	delays = range(-max_delay, max_delay)
	r = np.zeros(len(delays))
	for k in range(len(delays)):
		delay = delays[k]
		sxy = 0;
		for i in range(N):
			j = i + delay;
			if (j < 0 or j >= N):
				continue
			else:
				sxy += (series1[i] - m1) * (series2[j] - m2);
		r[k] = sxy / denom;
	return r

def new_cc(series1, series2, delta, max_delay): # never used??
	shifts = np.arange(-max_delay, delta, max_delay)
	for shift in shifts:
		shifted_wv = np.subtract(obj_wv, shift)

		f_spectrum = scipy.interpolate.interp1d(shifted_wv, obj_spec, fill_value=float('nan'), bounds_error=False)
		shifted_spec = f_spectrum(obj_wv)

def wavelength_calibration(image_pixels, image_flux, temp_wavelengths, template_flux, guess_coeffs, prom):
	# Inputs, arrays containg pixels, spectra to calibrate, wavelength range, template spectra
	# Outputs:
	#    image_wavelengths - an array containing the wavelength calibrated x of image spectra, or a boolean False
	#                        indicating a failure
	# 1. Initial guess
	guess_wavelengths = np.polynomial.polynomial.polyval(image_pixels, guess_coeffs)

	# 2. Find peaks
	guess_lines, _ = scipy.signal.find_peaks(image_flux, prominence=prom, width=2) # peak index
	# print('Image Line Locations: ', guess_lines)  # DEBUGGING

	#   Find centroids of peaks
	guess_centroids = []  # Values for all the centroids

	for peak in guess_lines:
		# print('\nFitting line at: ', guess_wavelengths[peak])  # TESTING
		# Calculate how far backward and forward to go from the peak
		half_max = (image_flux[peak] - np.median(image_flux))/2
		try:
			xmin = (np.where(image_flux[peak::-1] - np.median(image_flux) < half_max)[0])[0]
			xmax = (np.where(image_flux[peak+1:] - np.median(image_flux) < half_max)[0])[0]+1
		except Exception as e:
			continue

		# print(guess_wavelengths[np.max([peak-3*xmin, 0])], ' - ', guess_wavelengths[np.min([peak+3*xmax,len(guess_wavelengths)-1])])  # DEBUGGING

		aoi = [True if guess_wavelengths[np.max([peak-3*xmin, 0])] <= x <= guess_wavelengths[np.min([peak+3*xmax,len(guess_wavelengths)-1])] else False for x in guess_wavelengths]
		if len(guess_wavelengths[aoi]) == 0:
			print('ERROR, failed to locate area area of interest around peak at: ', peak)
			print('  xmin - ', xmin)
			print('  xmax - ', xmax)
			return 0

		guess_centroids.append(better_centroid(guess_wavelengths[peak], guess_wavelengths[aoi], image_flux[aoi], 0.1, absorption=False, plot=False))

	guess_lines = guess_wavelengths[guess_lines]

	temp_lines, _ = scipy.signal.find_peaks(np.divide(template_flux, np.max(template_flux)), prominence=0.01)
	temp_lines = temp_wavelengths[temp_lines]

	# 3. Select peaks
	matched_lines, select_lines, missing_lines = choose_peaks(guess_centroids, temp_lines, tol=4)
	print('Template Matched: %r  Image Selected: %r  Template Missed: %r' %(len(matched_lines), len(select_lines), len(missing_lines)))
	if len(missing_lines) > 5: # len(matched_lines) < 12:
		print('Unable to match peaks, calibration failed. Matched %r lines.' %len(matched_lines))
		print('Missing Lines: ', missing_lines)
		# #   Plot selected peaks
		plt.figure()
		plt.plot(guess_wavelengths, np.divide(image_flux, np.max(image_flux)), label='arc')
		plt.vlines(guess_lines, .5, 1.05, ls='--', color='red', alpha=0.5, label='unselected arc lines')
		plt.vlines(select_lines, .5, 1.05, ls='--', color='red', label='selected arc lines')
		plt.plot(temp_wavelengths, np.divide(template_flux, np.max(template_flux)), label='template')
		plt.vlines(temp_lines, -0.05, 0.5, ls='--', color='blue', alpha=0.5, label='unselected temp lines')
		plt.vlines(matched_lines, -0.05, 0.5, ls='--', color='blue', label='selected temp lines')
		plt.legend()
		plt.show()
		raise RuntimeError('Too many lines were not matched.')
	elif len(missing_lines) + len(matched_lines) != len(temp_lines):
		print('Unknown error, unable to match peaks. Calibration Failed')
		print('Select Lines: ', select_lines)
		print('Template Lines: ', temp_lines)
		raise RuntimeError

	#   Plot selected peaks
	plt.figure()
	plt.plot(guess_wavelengths, np.divide(image_flux, np.max(image_flux)))
	plt.vlines(guess_lines, .5, 1.05, ls='--', color='red', alpha=0.5)
	plt.vlines(select_lines, .5, 1.05, ls='--', color='red')
	plt.plot(temp_wavelengths, np.divide(template_flux, np.max(template_flux)))
	plt.vlines(temp_lines, -0.05, 0.5, ls='--', color='blue', alpha=0.5)
	plt.vlines(matched_lines, -0.05, 0.5, ls='--', color='blue')
	plt.show()

	# 4. Find image peaks in pixels
	#   Since the line centroids that I found are not necessarily going to fall on the exact pixel wavelengths, I need
	#   to interpolate from wavelength to pixel, and then find the decimal pixel value for the line centroids
	f_pixels = scipy.interpolate.interp1d(guess_wavelengths, image_pixels)
	pixel_lines = f_pixels(select_lines)

	# 5. Convert pixels to wavelength
	coeffs = np.polynomial.polynomial.Polynomial.fit(pixel_lines, matched_lines, 4).convert().coef
	print('Fit Coeffs: ', coeffs)
	image_wavelength = np.array(np.polynomial.polynomial.polyval(image_pixels, coeffs))

	return image_wavelength

def wv_calibration_check(calibrated_wv, flux):
	# Check the goodness of the wavelength calibration by finding the centroid of spectral lines across the spectrum,
	# and comparing the resultant RVs

	# Spectral Lines in K and M  dwarfs
	lines = [5889.95, 5895.92, 6322.7, 6562.7, 6750.152, 6978.851, 7244.853, 7251.708, 8183.25, 8194.79, 8542.09, 8806.757]
	absorption = [True, True, True, True, True, True, True, True, True, True]
	# Cut to only lines present in the current wavelength range
	absorption = [absorption[i] for i in range(len(lines)) if np.min(calibrated_wv) < lines[i] < np.max(calibrated_wv)]
	lines = [lines[i] for i in range(len(lines)) if np.min(calibrated_wv) < lines[i] < np.max(calibrated_wv)]

	delta_ang = []
	rvs = []
	for i in range(len(lines)):
		# Isolate area around line
		aoi = [True if lines[i] - 3 < x < lines[i] + 3 else False for x in calibrated_wv]

		# Locate line centroids
		cen = better_centroid(lines[i], calibrated_wv[aoi], flux[aoi], .01, absorption=absorption[i], plot=True)

		# Compare to Known locations
		# cen - lines[i] = delta
		delta = cen - lines[i]
		print('Angstrom Shift at', lines[i] ,': ', delta)
		rv = delta / lines[i] * c.c.to('km/s').value
		print('RV: ', rv)
		delta_ang.append(delta)
		rvs.append(rv)

		# Plot check
		plt.figure()
		plt.plot(calibrated_wv[aoi], np.divide(flux[aoi], np.median(flux[aoi])))
		plt.vlines(lines, 0., 1., 'k', '--', label='Line')
		plt.vlines([cen], 0., 1., 'k', ':', label='Centroid')
		plt.xlim(calibrated_wv[aoi][0], calibrated_wv[aoi][-1])
		plt.xlabel(r'$\lambda (\AA)$')
		plt.ylabel('Normalized\nFlux')
	plt.show()

	# Check range, mean, std  of delta and rv
	print('Angstrom Shift Range: ', np.max(delta_ang) - np.min(delta_ang))
	print('Angstrom Shift Mean: ', np.mean(delta_ang))
	print('Angstrom Shift Std: ', np.std(delta_ang))

	print('\nRV Range: ', np.max(rvs) - np.min(rvs))
	print('RV Mean: ', np.mean(rvs))
	print('RV Std: ', np.std(rvs))

def better_centroid(peak, x, signal, prominence, absorption=False, plot=False):
	# Locate the centroid of a line by fitting a gaussian profile to it
	# Inputs:
	#	- x, the wavelength array
	#	- signal, the signal array
	#	- prominence, the prominence of a peak necessary to consider it a peak
	#	- absorption, boolean, set to True if the line is in absorbtion to flip the signal
	#	- plot, boolean,  set to True to output a plot of the fit
	# Outputs:
	#	- centroid, the x position of the centroid
	# Flip the  signal to find absorption lines
	if absorption:
		signal = signal * -1.

	shifted_signal = signal - np.median(signal)

	# Intitial Guesses
	p0 = [abs(np.median(signal)), peak, 1.]

	# Fit Gaussian
	coeff, var_matrix = scipy.optimize.curve_fit(gauss, x, shifted_signal, p0=p0)

	if plot:
		plt.figure()
		plt.plot(x, shifted_signal, label='Signal')
		plt.plot(x, gauss(x, coeff[0], coeff[1], coeff[2]), label='Fit')
		plt.vlines([coeff[1]], 0.75 * np.median(signal), 1.25 * np.median(signal), 'k', '--', label='Centroid')
		plt.legend()
		plt.show()

	centroid = coeff[1]
	return centroid

def find_bias(folder):
	# Finds all bias images in a given folder, returns list of file names
	bias_list = []
	for file in os.listdir(folder):
		# print(file)
		if file.endswith('.fits'):
			full_path = folder+'/'+file
			_, header = fits.getdata(full_path, header=True)
			if (header['OBJECT'] == 'bias' or header['OBJECT'] == 'ZERO') and not file.endswith('-R.fits') and not file.startswith('Master'):
				bias_list.append(full_path)
	return bias_list

def find_flat(folder, flat_type='dome'):
	# finds all fits files in folder with flat data (as identified by the object field in the header)
	# flat_type default dome, other option quartz
	flat_list = []
	for file in os.listdir(folder):
		if file.endswith('.fits'):
			full_path = folder+'/'+file
			_, header = fits.getdata(full_path, header=True)
			if header['OBJECT'] == flat_type and not file.endswith('-R.fits') and not file.startswith('Master'):
				flat_list.append(full_path)
	return flat_list

def find_target(folder, target):
	# finds all fits files in folder on specified target using the object field of the header
	file_list = []
	for file in os.listdir(folder):
		if file.endswith('.fits'):
			full_path = folder+'/'+file
			_, header = fits.getdata(full_path, header=True)
			if header['OBJECT'] == target and not file.endswith('-R.fits') and not file.startswith('Master'):
				file_list.append(full_path)
	return file_list

def find_arcs(folder, arc_type, arc_id=None):
	# finds all fits files in folder on specified target using the object field of the header
	arc_list = []
	if arc_id:
		arc_name = str(arc_id) + arc_type + '.fits'
		print(arc_name)
		for file in os.listdir(folder):
			if file.endswith(arc_name) and not file.endswith('-R.fits') and not file.startswith('Master'):
				full_path = folder + '/' + file
				arc_list.append(full_path)
	else:
		for file in os.listdir(folder):
			if file.endswith('.fits'):
				full_path = folder+'/'+file
				_, header = fits.getdata(full_path, header=True)
				if header['OBJECT'] == arc_type and not file.endswith('-R.fits') and not file.startswith('Master'):
					arc_list.append(full_path)
	return arc_list

def get_sky_image(image, spectrum_loc, fit_order=3, outlier_tol=16):
	y_size = image.shape[0]
	x_size = image.shape[1]
	sky_image = np.zeros(image.shape)  # pre-allocate
	for j in range(x_size):
		column = image[:, j]
		object_loc = int(round(spectrum_loc[j]))  # The location of the object spectra in this column
		y_pixels = np.arange(0, y_size)
		sky_only = np.hstack([column[0:object_loc - 5], column[object_loc + 5:y_size]])
		sky_pixels = np.hstack([np.arange(0, object_loc - 5), np.arange(object_loc + 5, y_size)])

		# Iteratively fit polynomial to sky profile and remove outliers
		outliers = []
		new_outliers = [0]
		clean_indices = list(range(0, len(sky_pixels)))
		i_count = 0
		while outliers != new_outliers and i_count < 20:
			i_count += 1
			# Fit polynomial, get profile
			c = np.polyfit([sky_pixels[i] for i in clean_indices], [sky_only[i] for i in clean_indices], fit_order)
			full_sky_profile = np.polyval(c, y_pixels)
			short_sky_profile = np.polyval(c, sky_pixels)  # does not include the fit over the object area

			# Find outliers. Pixels are considered outliers either if they are a poor fit (high residual), or if
			# they have values much larger than the rest of the column
			outliers = new_outliers
			new_outliers = []
			clean_column = [sky_only[i] for i in clean_indices]  # the column with current outlier list removed
			residual = abs(np.subtract(short_sky_profile, sky_only))  # diff btwn fit and actual

			new_outliers = [x for x in range(len(sky_pixels)) if (
						residual[x] - np.median(residual) > 4 * np.std(residual) or sky_only[x] > np.median(
					clean_column) + 4 * np.std(clean_column))]
			clean_indices = [x for x in range(0, len(sky_pixels)) if x not in new_outliers]

		# Save to sky image
		sky_image[:, j] = full_sky_profile

	return sky_image

def get_skyline_template(wavelength_range):
	skyline = Table.read('UVES_sky_spectra.txt', format='ascii')
	skyline['CENTER'] = [float(x) for x in skyline['CENTER']]
	skyline['FLUX'] = [float(x) for x in skyline['FLUX']]

	skyline = skyline['CENTER', 'FLUX', 'FWHM']

	skyline = skyline[[True if x > 6300 and x < 7400 else False for x in skyline['CENTER']]]
	skyline = skyline[[True if x >= 1 else False for x in skyline['FLUX']]]

	for row in skyline:
		center = row['CENTER']
		fwhm = row['FWHM']
		if fwhm > 0.:
			left = center - fwhm / 2
			right = center + fwhm / 2
			left_flux = row['FLUX'] / 2
			right_flux = row['FLUX'] / 2
			skyline.add_row([left, left_flux, 0.0])  # half max
			skyline.add_row([right, right_flux, 0.0])
		else:
			break

	skyline.sort('CENTER')
	skyline.rename_columns(['CENTER', 'FLUX'], ['wavelength', 'flux'])

	return skyline

def clean_folder(folder):
	# Remove all existant reduction files in a folder
	for file in os.listdir(folder):
		if file.endswith('-R.fits'):
			full_path = folder + '/' + file
			os.remove(full_path)
		elif file.startswith('Master'):
			full_path = folder + '/' + file
			os.remove(full_path)

def reduce_and_extract(folder, target, flatname, arc_id='', exclude=[]):
	# Full Reduction and Spectral Extraction
	# Does not correct for RV
	# 1. Basic Reduction
	#    i.   Gather data (Bias, Flat, Arc, Image)
	bias_list = find_bias(folder)
	if len(bias_list) == 0:
		print('ERROR: No bias images found')
		return
	bias_data, bias_headers = get_data(bias_list)
	flat_list = find_flat(folder, flatname)
	if len(flat_list) == 0:
		print('ERROR: No flat images found')
		return
	flat_data, flat_headers = get_data(flat_list)
	image_list = find_target(folder, target)
	image_list.sort()
	print('Images: ', image_list)
	# exclude images
	for x in exclude:
		print('Exclude Image:', x)
		if x in image_list:
			image_list.remove(x)
		else:
			print('WARNING: Excluded image not in list')

	image_data, image_headers = get_data(image_list)

	#    ii.  Overscan subtraction  (Bias, Flat, Arc, Image)
	subtracted_bias, bias_headers = overscan_subtraction(bias_data, bias_headers)
	subtracted_flat, flat_headers = overscan_subtraction(flat_data, flat_headers)
	subtracted_image, image_headers = overscan_subtraction(image_data, image_headers)
	#    iii. Subtract Bias (Flat, Arc, Image)
	median_bias, bias_header = median_stack(subtracted_bias, bias_headers[0])
	subtracted_flat, flat_headers = subtraction(subtracted_flat, flat_headers, median_bias, bias_header)
	subtracted_image, image_headers = subtraction(subtracted_image, image_headers, median_bias, bias_header)
	#    iv.  Normalize Flat (Flat)
	median_flat, flat_header = median_stack(subtracted_flat, flat_headers[0])
	flat_median = np.median(median_flat)
	normalized_flat = np.divide(median_flat, flat_median)
	#    v.   Divide Flat (Arc, Image)
	reduced_images, image_headers = division(subtracted_image, image_headers, normalized_flat, flat_headers[0])
	# remove cosmic rays using astroscrappy
	reduced_images = [detect_cosmics(image, sigclip=3, readnoise=flat_header['RDNOISE'])[1] for image in reduced_images]

	# Chop off secondary stars
	# for i in range(len(reduced_images)):
	# 	reduced_images[i] = reduced_images[i][0:130, :]

	# # 2. Image Spectral Extraction and Calibration
	# #    i.   Median stack images
	# median_image, _ = median_stack(reduced_images, image_headers[0])
	# #    ii.  Find trace
	# # Find brightest row
	# x_size = median_image.shape[1]
	# print(x_size)
	# x_pixels = np.arange(0, x_size)
	# y_size = median_image.shape[0]
	# brightest_row = []
	# for i in range(x_size):
	# 	col = list(median_image[:, i])
	# 	brightest_row.append(col.index(max(col)))
	# # Find spectrum loc by iteratively fitting brightest w/ 3rd order polynomial
	# outliers = []
	# new_outliers = list(range(0, 100))  # initialize outliers with a list of 0-100
	# clean_indices = list(range(0, x_size))  # initialize clean columns list with all columns
	# i_count = 0
	# while new_outliers != outliers and i_count < 20:
	# 	i_count += 1
	# 	coeffs = np.polyfit([x_pixels[i] for i in clean_indices], [brightest_row[i] for i in clean_indices], 2)
	# 	ys = np.polyval(coeffs, x_pixels)
	# 	std_dev = np.std(ys)
	#
	# 	outliers = new_outliers
	# 	new_outliers = [i for i in range(0, x_size) if abs(brightest_row[i] - ys[i]) / std_dev > 16]
	# 	clean_indices = [x for x in np.arange(0, x_size) if x not in new_outliers]
	# spectrum_loc = ys
	#
	# # # Check trace
	# # plt.figure()
	# # plt.imshow(median_image)
	# # plt.scatter(x_pixels, brightest_row)
	# # plt.plot(x_pixels, spectrum_loc)
	# # plt.show()
	#
	# #    iv. Apply trace to individuals to get individual object spectra
	# object_spectra = []
	# for image in reduced_images:
	# 	aperture = 5
	# 	spectrum = [np.sum(image[int(spectrum_loc[x] - aperture):int(spectrum_loc[x] + aperture), x], 0) for x in
	# 				x_pixels]
	# 	object_spectra.append(spectrum)
	#
	# #    v. Get sky spectra, from above and below object, summed over vertical pixels
	# gap = 25
	# sky_spectra = []
	# for image in reduced_images:
	# 	above = [np.sum(image[int(spectrum_loc[x] + gap - aperture):int(spectrum_loc[x] + gap + aperture), x], 0) for x in
	# 			 x_pixels]
	# 	below = [np.sum(image[int(spectrum_loc[x] - gap - aperture):int(spectrum_loc[x] - gap + aperture), x], 0) for x in
	# 			 x_pixels]
	# 	sky_spectra.append(np.mean([above, below], axis=0))
	#
	# #  Subtract sky from object
	# sub_object_flux = np.subtract(object_spectra, sky_spectra)

	# 2. Image Spectral Extraction and Calibration (Individual version)
	object_spectra = []
	sky_spectra = []
	sub_object_flux = []
	for image in reduced_images:
		#    i.  Find trace
		# Find brightest row
		x_size = image.shape[1]
		x_pixels = np.arange(0, x_size)
		y_size = image.shape[0]
		brightest_row = []
		for i in range(x_size):
			col = list(image[:, i])
			brightest_row.append(col.index(max(col)))
		# Find spectrum loc by iteratively fitting brightest w/ 3rd order polynomial
		outliers = []
		new_outliers = list(range(0, 100))  # initialize outliers with a list of 0-100
		clean_indices = list(range(0, x_size))  # initialize clean columns list with all columns
		i_count = 0
		while new_outliers != outliers and i_count < 20:
			i_count += 1
			coeffs = np.polyfit([x_pixels[i] for i in clean_indices], [brightest_row[i] for i in clean_indices], 2)
			ys = np.polyval(coeffs, x_pixels)
			std_dev = np.std(ys)

			outliers = new_outliers
			new_outliers = [i for i in range(0, x_size) if abs(brightest_row[i] - ys[i]) / std_dev > 16]
			clean_indices = [x for x in np.arange(0, x_size) if x not in new_outliers]
		spectrum_loc = ys

		# # Check trace
		# plt.figure()
		# plt.imshow(median_image)
		# plt.scatter(x_pixels, brightest_row)
		# plt.plot(x_pixels, spectrum_loc)
		# plt.show()

		#    iv. Apply trace to individuals to get individual object spectra
		aperture = 5
		spectrum = [np.sum(image[int(spectrum_loc[x] - aperture):int(spectrum_loc[x] + aperture), x], 0) for x in
					x_pixels]
		object_spectra.append(spectrum)

		#    v. Get sky spectra, from above and below object, summed over vertical pixels
		gap = 30
		above = [np.sum(image[int(spectrum_loc[x] + gap - aperture):int(spectrum_loc[x] + gap + aperture), x], 0) for x in
				 x_pixels]
		below = [np.sum(image[int(spectrum_loc[x] - gap - aperture):int(spectrum_loc[x] - gap + aperture), x], 0) for x in
				 x_pixels]
		sky_spectra.append(np.mean([above, below], axis=0))

		#  Subtract sky from object
		sub_object_flux.append(np.transpose(np.subtract(object_spectra[-1], sky_spectra[-1])))

	fig, ax = plt.subplots(1, 3, figsize=(18, 4))
	ax[0].plot(x_pixels, object_spectra[0])
	ax[0].set_title('Object')
	ax[1].plot(x_pixels, sky_spectra[0])
	ax[1].set_title('Sky')
	ax[2].plot(x_pixels, sub_object_flux[0])
	ax[2].set_title('Sky Subtracted')
	plt.show()

	plt.figure()
	for i in range(len(image_list)):
		plt.plot(x_pixels, sub_object_flux[i], label=i)
	plt.xlabel('pixel')
	plt.title('Subtracted Spectra - Check for bad spectra/shifting solution')
	plt.legend()
	plt.show()

	# 3. Wavelength Calibrate
	# Test if sky spectra contains lines
	skylines, _ = scipy.signal.find_peaks(sky_spectra[0], prominence=120, width=2)
	print('Skylines:', len(skylines))

	# Get Guess coefficients
	# guess_coeffs =  [6.28e+03,  3.10e-01, -1.86e-06,  5.07e-11, -4.43e-14] # 1200 m5, (2019-12-11)
	# guess_coeffs = [6280,  3.10e-01, -1.58e-07, -8.07e-10, 8.84e-14] # 1200 m5 (2020-02-06)
	# guess_coeffs = [ 6288,  3.097e-01, -1.12e-06, -2.997e-10, 8.75e-15] # 1200 m5 (2020-03-05)
	guess_coeffs = [6286, 3.10e-01, -1.09e-06, -3.15e-10, 1.16e-14]  # 1200 m5 (2021-02-22, 2021-03-17, 2021-04-23, 2021-09-20)
	# guess_coeffs = [6300, 3.10e-01, -1.09e-06, -3.15e-10, 1.16e-14]  # 1200 m5 (2021-02-22, 2021-04-23, 2021-09-20)
	# guess_coeffs = [6280, 3.11e-01, -1.849e-06, -1.00e-10, -8.26e-15]  # 1200 m5, (2021-08-10)
	# guess_coeffs = [6275, 3.11e-01, -1.25e-06, -2.47e-10, 2.31e-15]  # 1200 m5 (2021-09-19)
	# guess_coeffs = [6280, 3.11e-01, -1.25e-06, -2.47e-10, 2.31e-15]  # 1200 m5 (2021-09-19)
	# guess_coeffs = [6281.6,  3.10e-01, - 9.61e-07, - 3.71e-10, 1.88e-14] # 1200 m5 (2021-09-19)
	# guess_coeffs = [6297, 3.11e-01, -1.73611691e-06, -5.28334589e-11, -2.207e-14] # 1200 m5 (2022-02-19)
	# guess_coeffs = [6300,  3.09e-01, -6.80406766e-07, -4.77929556e-10, 3.27277111e-14] # 1200 m5 (2022-04-05)
	# guess_coeffs = [6296,  3.10e-01, - 8.62439198e-07, - 4.45040365e-10, 3.16323542e-14]  # 1200 m5 (2022-04-11, 2022-04-19)
	# guess_coeffs = [6295.53,  3.10e-01, -1.382e-06, - 2.135e-10, - 3.268e-16] # 1200 m5 (2022-10-21)
	# guess_coeffs = [6287.87,  3.100e-01, -8.734e-07, -4.22e-10, 2.53e-14] # 1200m5 20221213
	# guess_coeffs = [6277.14,  3.100e-01, -9.797e-07, -3.063e-10, 2.387e-15] # 1200m5 20230108
	# guess_coeffs = [6264.42,  3.098e-01, -8.671e-07, -4.2816e-10, 2.675e-14] # 1200m5 20230201

	if len(skylines) <= 8:
		print('Arc Lamp Wavelength Calibration')
		# Use arcs for calibration
		#  obtain arc spectrum
		arc_list = find_arcs(folder, 'Ne', arc_id=arc_id)
		arc_data, arc_headers = get_data(arc_list)
		subtracted_arc, arc_headers = overscan_subtraction(arc_data, arc_headers)
		subtracted_arc, arc_headers = subtraction(subtracted_arc, arc_headers, median_bias, bias_header)
		reduced_arc, arc_headers = division(subtracted_arc, arc_headers, normalized_flat, flat_headers[0])
		arc_spectra = []
		for arc in reduced_arc:
			spectrum = [np.sum(arc[int(spectrum_loc[x] - aperture):int(spectrum_loc[x] + aperture), x], 0) for x in
						x_pixels]
			arc_spectra.append(spectrum)
		stacked_arc, _ = median_stack(arc_spectra, arc_headers[0])

		arc_peaks, _ = scipy.signal.find_peaks(stacked_arc, prominence=5000)
		ob_peaks, _ = scipy.signal.find_peaks(sub_object_flux[0], prominence=100)

		print(arc_peaks)

		#  Obtain template
		line_list = [6334.4278, 6382.9971, 6402.2480, 6506.5281, 6532.8822, 6598.9529, 6678.2762, 6717.0430, 6929.4673,
					 7032.4131, 7173.9381, 7245.1666, 7438.8981]
		heights = np.multiply([1, 1.5, 2, 1.5, 0.5, 1, 1, 0.5, 1.5, 4, 0.5, 2, 0.5], 200)
		temp_wv = np.arange(line_list[0]-10, line_list[-1]+10, 0.01)
		temp_flux = [heights[np.argmin(np.abs(np.subtract(line_list, x)))] if np.min(np.abs(np.subtract(line_list, x))) < 0.01 else 0. for x in temp_wv]
		temp_flux = [heights[np.argmin(np.abs(np.subtract(line_list, temp_wv[i])))]/2 if 0.01 < np.min(np.abs(np.subtract(line_list, temp_wv[i]))) < 0.025 else temp_flux[i] for i in range(len(temp_wv))]

		#  Calibrate
		x_wavelengths = wavelength_calibration(x_pixels, stacked_arc, np.array(temp_wv), np.array(temp_flux), guess_coeffs, 300)

		# Check for a correction on the calibration using the telluric lines
		#   Load Telluric Model
		t = Table.read('/Users/woodml/Documents/Models/Telluric.txt', format='ascii', delimiter=' ')
		tell_wv = t['wv(Ang)']
		tell_flux = t['Flux']

		# Mask non-Telluric lines in the object
		window = 50  # 100
		rolling_median = [[np.median(sub_object_flux[j][np.max([0, i - window]):np.min([len(sub_object_flux[j]), i + window])]) for i in range(len(x_wavelengths))] for j in range(len(image_list))]

		plt.figure()
		plt.plot(x_wavelengths, sub_object_flux[0])
		plt.plot(x_wavelengths, rolling_median[0])
		plt.title('Rolling Median - Check for appropriate Window')

		masked_flux = np.divide(sub_object_flux, rolling_median)
		for i in range(len(image_list)):
			masked_flux[i] = [1. if 6330. < x_wavelengths[j] < 6450. else masked_flux[i][j] for j in range(len(x_wavelengths))]  # few tellurics region
			masked_flux[i] = [1. if 6615. < x_wavelengths[j] < 6860. else masked_flux[i][j] for j in range(len(x_wavelengths))]  # few tellurics region
			masked_flux[i] = [1. if 6880. < x_wavelengths[j] < 6883. else masked_flux[i][j] for j in range(len(x_wavelengths))]  # few tellurics region

			lines = [5889.95, 5895.92, 6322.7, 6562.7, 6750.152, 6978.851, 7149., 7244.853, 7251.708, 8183.25, 8194.79, 8542.09, 8806.757]  # Some spectral lines, late Ks and Ms
			for line in lines:
				masked_flux[i] = [1. if line - 10 < x_wavelengths[j] < line + 10 else masked_flux[i][j] for j in range(len(x_wavelengths))]

			masked_flux[i] = np.array([1. if masked_flux[i][j] > 1. else masked_flux[i][j] for j in range(len(x_wavelengths))])

		# Plot masked flux and telluric
		plt.figure()
		plt.plot(tell_wv, tell_flux, label='Tellurics')
		plt.plot(x_wavelengths, np.divide(sub_object_flux[0], np.median(sub_object_flux[0])), label='Object')
		plt.plot(x_wavelengths, masked_flux[0], label='Masked Object')
		plt.title('Telluric Offset - Check for appropriate Masking')
		plt.legend()
		plt.xlabel(r'$\lambda (\AA)$')
		plt.ylabel('Flux')
		plt.show()

		# Cross Correlate
		all_wavelengths = []
		corrections = []
		snr = []
		for i in range(len(image_list)):
			telluric_correction = compute_xcor_rv(x_wavelengths, masked_flux[i], '00', wavelim=[6300, 7300], norm_reg=[6750, 7000], twave=np.copy(tell_wv), tflux=np.copy(tell_flux), maxshift=50)
			corrections.append(telluric_correction)
			snr.append(np.median(np.divide(sub_object_flux[i], np.std(sub_object_flux[i]))))

		telluric_correction = np.average(corrections, weights=snr)
		print('Telluric Correction: ', telluric_correction, ' km/s')
		for i in range(len(image_list)):
			all_wavelengths.append(np.divide(x_wavelengths, (1 + telluric_correction / c.c.to('km/s').value)))

		# Copy wavelength solution
		# x_wavelengths = [x_wavelengths]*len(image_list)
		x_wavelengths = all_wavelengths
	else:
		# use skylines for calibration
		print('Skyline Wavelength Calibration')
		template = get_skyline_template([6305, 7500])  # 1200 m5
		x_wavelengths = []

		# Do the first one by hand
		x, coeffs = human_wavelength_calibration(guess_coeffs, x_pixels, sky_spectra[0], np.array(template['wavelength']), np.array(template['flux']))
		x_wavelengths.append(x)

		# Now see if that solution works for all the others, otherwise do it by hand again
		for i in range(1, len(image_list)):
			try:
				x = wavelength_calibration(x_pixels, sky_spectra[i], np.array(template['wavelength']), np.array(template['flux']), coeffs, 60)
			except Exception as e:
				print('Error: ', e)
				print('Using human wavelength calibration...')
				x, coeffs = human_wavelength_calibration(coeffs, x_pixels, sky_spectra[i], np.array(template['wavelength']), np.array(template['flux']))
			x_wavelengths.append(x)


	#  Plot results
	plt.figure()
	for i in range(len(image_list)):
		plt.plot(x_wavelengths[i], sub_object_flux[i], label=i)
	plt.legend()
	plt.title('Wavelength Calibrated - Check for Alignment')
	plt.show()

	# Align all spectra to the same wavelength array
	shifted_flux = [[] for x in image_list]
	shifted_flux[0] = sub_object_flux[0]
	for i in range(1, len(image_list)):
		f_int = scipy.interpolate.interp1d(x_wavelengths[i], sub_object_flux[i], fill_value="extrapolate")
		shifted_flux[i] = f_int(x_wavelengths[0])
	x_wavelengths = x_wavelengths[0]
	shifted_flux = np.array(shifted_flux)

	# Now that everything is lined up very nice I can stack them without any fear of losing resolution
	stacked_flux, _ = median_stack(shifted_flux, image_headers[0])

	return x_wavelengths, stacked_flux, image_headers[0]

def human_wavelength_calibration(guess_coeffs, image_pixels, image_spectra, temp_wv, temp_spec):
	# Inputs, arrays containg pixels, spectra to calibrate, wavelength range, template spectra
	# Outputs:
	#    image_wavelengths - an array containing the wavelength calibrated x of image spectra, or a boolean False
	#                        indicating a failure
	# 1. Initial guess
	guess_wavelengths = np.polynomial.polynomial.polyval(image_pixels, guess_coeffs)

	# 2. Find Peaks
	guess_lines, _ = scipy.signal.find_peaks(image_spectra, prominence=90, width=2)
	guess_lines = guess_wavelengths[np.array(guess_lines)]
	temp_lines, _ = scipy.signal.find_peaks(np.divide(temp_spec, np.max(temp_spec)), prominence=0.008)
	temp_lines = temp_wv[temp_lines]

	# Select Peaks
	image_matches = {}
	select_lines = []
	matched_lines = []

	def onclick(event):
		line = event.artist
		x_data = line.get_segments()[0][0]
		if x_data is not None:
			line.set_alpha(1.)
			plt.plot([t, x_data[0]], [0.5, 0.5])
			matched_lines.append(t)
			select_lines.append(x_data[0])
		plt.draw()  # update plot
		return

	for t in temp_lines:
		unused_guess_lines = [x for x in  guess_lines if x not in select_lines]
		fig = plt.figure()
		plt.plot(guess_wavelengths, np.divide(image_spectra, np.max(image_spectra) * 2) + 0.5)
		[plt.vlines(x, .5, 1.05, ls='--', color='red', alpha=0.5, picker=5) for x in unused_guess_lines]
		plt.plot(temp_wv, np.divide(temp_spec, np.max(temp_spec) * 2) + 0.45)
		plt.vlines(temp_lines, -0.05, 0.5, ls='--', color='blue', alpha=0.5)  # all the template lines
		plt.vlines(t, -0.05, 0.5, ls='--', color='blue', alpha=1.)  # the one I'm looking at
		# previous matches
		plt.vlines(select_lines, 0.5,  1.05, ls='--', color='red', alpha=0.8)
		plt.vlines(matched_lines, -0.5, 0.5, ls='--', color='blue', alpha=0.7)
		[plt.plot([select_lines[i], matched_lines[i]],[0.5, 0.5], color='purple', alpha=0.75) for i in range(len(select_lines))]
		# lims
		plt.ylim(0.4, 1.05)
		plt.xlim(t-50, t+50)
		fig.canvas.mpl_connect('pick_event', onclick)
		plt.show()

	# Find image peaks in pixels
	peak_index = np.where(np.isin(guess_wavelengths, np.array(select_lines)))[0]
	pixel_lines = image_pixels[peak_index]

	# Convert pixels to wavelength
	coeffs = np.polynomial.polynomial.Polynomial.fit(pixel_lines, matched_lines, 4).convert().coef
	print(coeffs)
	image_wavelength = np.array(np.polynomial.polynomial.polyval(image_pixels, coeffs))

	return image_wavelength, coeffs


# code for cross correlating between object spectra
# 	#    vii.  Cross Correlate all sky spectra to the best one
# 	#    viii. Apply offset from cross correlation to the other spectra
# 	#          a. Found an x km/s difference between spectra A and template spectra T
# 	#          b. Shift wavelength arrays by x
# 	#          c. Interpolate the flux array
# 	#          d. flux_1@λ_2=interpolate(flux_1,λ_1,λ_2 )
# 	#          e. Now have flux1, flux2, … flux5, @ λ2
# 	#    ix.  Stack all the spectra fluxes
# 	#
# 	# Select the best by the one with the smallest std
# 	best_idx = np.argmin([np.std(x) for x in sub_object_flux])
#
# 	other_idx = list(range(len(image_list)))
# 	other_idx.remove(best_idx)
#
# 	# Align the spectra by interpolating each one to be sampled at the same wavelengths as the best one
# 	aligned_sky_spectra = [np.zeros(len(sky_spectra[best_idx]))] * len(image_list)
# 	aligned_sky_spectra[best_idx] = sky_spectra[best_idx]
# 	aligned_spectra = [np.zeros(len(sub_object_flux[best_idx]))] * len(image_list)
# 	aligned_spectra[best_idx] = sub_object_flux[best_idx]
# 	for i in other_idx:
# 		f_int = scipy.interpolate.interp1d(x_wavelengths[i], sky_spectra[i], fill_value="extrapolate")
# 		aligned_sky_spectra[i] = f_int(x_wavelengths[best_idx])
# 		f_int = scipy.interpolate.interp1d(x_wavelengths[i], sub_object_flux[i], fill_value="extrapolate")
# 		aligned_spectra[i] = f_int(x_wavelengths[best_idx])
#
# 	# All spectra now use the same wavelength array, x_wavelengths[best_idx]
#
# 	# Determine cross-correlation shifts by using the sky spectra
# 	shifted_wavelengths = [np.zeros(len(x_wavelengths[best_idx]))] * len(image_list)
# 	shifted_spectra = [np.zeros(len(aligned_spectra[best_idx]))] * len(image_list)
# 	for i in range(len(image_list)):
# 		cs = cross_correlate(sub_object_flux[best_idx], aligned_spectra[i], 1000)
# 		best_shift = np.argmax(cs) - 1000
# 		print('Shift: ', best_shift)  # TESTING
#
# 		shifted_spectra[i] = [aligned_spectra[i][j + best_shift] if 0 <= (j + best_shift) < len(aligned_spectra[i]) else 0. for j in range(len(x_wavelengths[best_idx]))]
