# Goodman Pipeline

This is a pipeline to reduce and extract spectra from the SOAR Goodman High-Throughput Spectrograph. This pipeline is specifically designed for the 1200 grating in the m5 mode, which covers a wavelength range of ~6400-7400 angstroms, although it may work on other settings with different line lists for calibration. 

An example use is included in example.py. To use:

`import reduce_and_extract as rp

date = '20210317'
target = 'TIC311449078'
folder = 'SOAR:Goodman/Raw/' + date

rp.clean_folder(folder)
wv, spec, header = rp.reduce_and_extract(folder, target, flatname='dome1200', arc_id='97')`

In this example the flatname is the OBJECT in the header of your flatfield images, and the arc_id is the OBJECT in the header of the arc lamp images for wavelength calibration.
