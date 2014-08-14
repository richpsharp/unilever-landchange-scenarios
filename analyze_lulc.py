import os
import numpy
import gdal
import collections
import time
import osr

def get_cell_size_from_uri(dataset_uri):
    """Returns the cell size of the dataset in meters.  Raises an exception
        if the raster is not square since this'll break most of the raster_utils
        algorithms.

        dataset_uri - uri to a gdal dataset

        returns cell size of the dataset in meters"""

    srs = osr.SpatialReference()
    dataset = gdal.Open(dataset_uri)
    if dataset == None:
        raise IOError(
            'File not found or not valid dataset type at: %s' % dataset_uri)
    srs.SetProjection(dataset.GetProjection())
    linear_units = srs.GetLinearUnits()
    geotransform = dataset.GetGeoTransform()
    #take absolute value since sometimes negative widths/heights
    try:
        numpy.testing.assert_approx_equal(
            abs(geotransform[1]), abs(geotransform[5]))
        size_meters = abs(geotransform[1]) * linear_units
    except AssertionError as e:
        print e
        size_meters = (
            abs(geotransform[1]) + abs(geotransform[5])) / 2.0 * linear_units

    #Make sure the dataset is closed and cleaned up
    gdal.Dataset.__swig_destroy__(dataset)
    dataset = None

    return size_meters

lulc_iowa_global = "C:/Users/rich/Documents/Dropbox/Unilever_data_from_Stacie/Input_Iowa_global/LULC_MCD12Q1_2006_Iowa_HUC8.tif"
lulc_iowa_national = "C:/Users/rich/Documents/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/LULC_NLCD_2006_Iowa_HUC8_uncompressed.tif"
lulc_mg_global = "C:/Users/rich/Documents/Dropbox/Unilever_data_from_Stacie/Input_MatoGrosso_global/LULC_MCD12Q1_2012_MG_MT_G.tif"
lulc_iowa_global_table = "C:/Users/rich/Documents/Dropbox/Unilever_data_from_Stacie/Input_Iowa_global/biophysical_coeffs_Iowa_Unilever_global.csv"
lulc_iowa_national_table = "C:/Users/rich/Documents/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/biophysical_coeffs_Iowa_Unilever_national.csv"
lulc_mg_global_table = "C:/Users/rich/Documents/Dropbox/Unilever_data_from_Stacie/Input_MatoGrosso_global/biophysical_coeffs_Brazil_Unilever.csv"



for ds_uri, table_uri in zip([lulc_iowa_global, lulc_iowa_national, lulc_mg_global], [lulc_iowa_global_table, lulc_iowa_national_table, lulc_mg_global_table]):
	print 'processing ', ds_uri
	ds = gdal.Open(ds_uri)
	band = ds.GetRasterBand(1)
	lulc_count = collections.defaultdict(int)
	start_time = time.time()
	
	n_rows = ds.RasterYSize
	n_cols = ds.RasterXSize

	for row_index in xrange(band.YSize):
		current_time = time.time()
		if current_time - start_time > 5.0:
			print '%.2f%% complete' % (100.0*float(row_index)/band.YSize)
			start_time = current_time
		array = band.ReadAsArray(yoff=row_index, win_xsize=band.XSize, win_ysize=1, xoff=0)
		for lulc_value in numpy.unique(array):
			lulc_count[lulc_value] += numpy.count_nonzero(array==lulc_value)
	print lulc_count


	cell_size = get_cell_size_from_uri(ds_uri)
	print 'cell size is %f' % cell_size
    
	f=open(os.path.splitext(os.path.basename(ds_uri))[0] + '.csv', 'w')
	f.write('lucode,count,area(ha)\n')
	for lulc_value in sorted(lulc_count.keys()):
		f.write('%d,%d,%f\n' % (lulc_value, lulc_count[lulc_value], cell_size**2/10000.0 * lulc_count[lulc_value]))
	f.close()
