import os
import gdal
import numpy
import math

from invest_natcap import raster_utils

def calculate_pixels_per_step_for_full_conversion(
    landuse_uri, convert_from_lulc_codes, number_of_steps):

    landuse_ds = gdal.Open(landuse_uri)
    landuse_band = landuse_ds.GetRasterBand(1)
    n_rows = landuse_band.YSize
    n_cols = landuse_band.XSize
    block_col_size, block_row_size = landuse_band.GetBlockSize()

    convertable_pixels = 0

    for global_block_row in xrange(int(numpy.ceil(float(n_rows) / block_row_size))):
        for global_block_col in xrange(int(numpy.ceil(float(n_cols) / block_col_size))):
            global_col = global_block_col*block_col_size
            global_row = global_block_row*block_row_size
            global_col_size = min((global_block_col+1)*block_col_size, n_cols) - global_col
            global_row_size = min((global_block_row+1)*block_row_size, n_rows) - global_row
            landuse_block = landuse_band.ReadAsArray(
                global_col, global_row, global_col_size, global_row_size)
            for lulc_code in convert_from_lulc_codes:
                convertable_pixels += len(numpy.nonzero(landuse_block == lulc_code)[0])
    return int(math.ceil(convertable_pixels / float(number_of_steps)))


dem_lulc_list = [
    ("C:/Users/rich/Documents/Dropbox/unilever_sdr_ndr_run_data/Input_MatoGrosso_global_Unilever_10_09_2014/SRTM_90m_MatoGrosso_final_basins.tif",
    "C:/Users/rich/Documents/Dropbox/unilever_sdr_ndr_run_data/Input_MatoGrosso_global_Unilever_10_09_2014/MCD12Q1_2012_Type2_MatoGrosso_final_basins.tif"),
#    ("C:/Users/rich/Documents/Dropbox/unilever_sdr_ndr_run_data/Input_Heilongjiang_global_Unilever_10_09_2014/SRTM_90m_Heilongjiang_final_basin.tif",
#    "C:/Users/rich/Documents/Dropbox/unilever_sdr_ndr_run_data/Input_Heilongjiang_global_Unilever_10_09_2014/MCD12Q1_2012_Type2_Heilongjiang_final_basin.tif"),
#    ("C:/Users/rich/Documents/Dropbox/unilever_sdr_ndr_run_data/Input_Jiangxi_global_Unilever_10_09_2014/SRTM_90m_Jiangxi_final_basin_fill.tif",
#    "C:/Users/rich/Documents/Dropbox/unilever_sdr_ndr_run_data/Input_Jiangxi_global_Unilever_10_09_2014/MCD12Q1_2012_Type2_Jiangxi_final_basin.tif"),
]

for dem_uri, landuse_uri in dem_lulc_list:
    aligned_lulc_uri = raster_utils.temporary_filename()
    tmp_dem_uri = raster_utils.temporary_filename()

    out_pixel_size = raster_utils.get_cell_size_from_uri(dem_uri)
    raster_utils.align_dataset_list(
        [dem_uri, landuse_uri], [tmp_dem_uri, aligned_lulc_uri],
        ['nearest'] * 2, out_pixel_size, 'dataset',
        0, dataset_to_bound_index=0)

    pixels_per_step = calculate_pixels_per_step_for_full_conversion(aligned_lulc_uri, range(1,11), 20)
    area_per_step = pixels_per_step * out_pixel_size**2 / 100**2
    print os.path.basename(landuse_uri), area_per_step
