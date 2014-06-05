import os
import hashlib
import pickle

import gdal
import numpy

from invest_natcap.routing import routing_utils
from invest_natcap import raster_utils
from invest_natcap.scenario_generator import disk_sort
import invest_natcap.sdr.sdr

def hashfile(filename, blocksize=65536):
    afile = open(filename, 'rb')
    buf = afile.read(blocksize)
    hasher = hashlib.sha256()
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    return hasher.digest()


def save_obj(filename, object):
    with open(filename, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def initialize_simulation(parameters):
    for dirname in [parameters['temporary_file_directory'],
                    parameters['output_file_directory']]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    parameters['dem_hash'] = hashfile(parameters['dem_filename'])
    parameters['lulc_hash'] = hashfile(parameters['lulc_filename'])
    
    previous_run_file = os.path.join(
        parameters['temporary_file_directory'], 'previous_run.obj')
    if os.path.isfile(previous_run_file):
        previous_parameters = load_obj(previous_run_file)
        
        for parameter in [
                'dem_filename', 'lulc_filename', 'dem_hash', 'lulc_hash', 
                'flow_accumulation_threshold_for_streams']:
            if previous_parameters[parameter] != parameters[parameter]:
                break
        else:
            parameters['distance_from_stream_filename'] = (
                previous_parameters['distance_from_stream_filename'])
    
    if 'distance_from_stream_filename' not in parameters:
        parameters['flow_direction_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'flow_direction.tif')
        parameters['flow_accumulation_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'flow_accumulation.tif')
        parameters['stream_uri'] = os.path.join(
            parameters['temporary_file_directory'], 'streams.tif')
        parameters['distance_from_stream_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'distance_from_stream.tif')
        
        print 'resolving filling pits'
        dem_pit_filled_uri = os.path.join(
            parameters['temporary_file_directory'], 'pit_filled_dem.tif')
        routing_utils.fill_pits(parameters['dem_filename'], dem_pit_filled_uri)
        
        print 'resolving plateaus'
        dem_plateau_resolved_uri = os.path.join(
            parameters['temporary_file_directory'], 'plateau_resolved_dem.tif')
        routing_utils.resolve_flat_regions_for_drainage(
            dem_pit_filled_uri, dem_plateau_resolved_uri)
        
        print 'calculate flow direction'
        routing_utils.flow_direction_inf(
            dem_plateau_resolved_uri, parameters['flow_direction_filename'])
        
        print 'calculate flow accumulation'
        routing_utils.flow_accumulation(
            parameters['flow_direction_filename'], dem_plateau_resolved_uri,
            parameters['flow_accumulation_filename'])
        
        print 'calculate stream threshold'
        routing_utils.stream_threshold(
            parameters['flow_accumulation_filename'], 
            parameters['flow_accumulation_threshold_for_streams'],
            parameters['stream_uri'])
            
        print 'calculate distance from streams'
        raster_utils.distance_transform_edt(
            parameters['stream_uri'],
            parameters['distance_from_stream_filename'])
    else:
        print 'streams already calculated, use those'
        
    save_obj(previous_run_file, parameters)

 
def step_land_change_from_streams(
    parameters, base_name, mode, stream_buffer_width):
    """
        parameters - the context from the main function
        base_name - base of the filename
        mode - one of "to_stream" or "from_stream"
        stream_buffer_width - the width in pixels 
        
        returns a list of land cover change from base to increasing expansion
    """
    
    if mode == "to_stream":
        direction_factor = -1
    elif mode == "from_stream":
        direction_factor = 1
    conversion_priority_filename = os.path.join(
        parameters['temporary_file_directory'], 'conversion_priority.tif')
    conversion_nodata = direction_factor * 9999
    conversion_pixel_size = raster_utils.get_cell_size_from_uri(
        parameters['lulc_filename'])
    lulc_nodata = raster_utils.get_nodata_from_uri(parameters['lulc_filename'])
        
    def valid_distance(distance, lulc):
        conversion_array = numpy.empty(distance.shape, dtype=numpy.float32)
        conversion_array[:] = conversion_nodata
        for convert_code in parameters['convert_from_lulc_codes']:
            mask = (lulc == convert_code)
            #invert the distance for sorting
            conversion_array[mask] = -distance[mask] * direction_factor
        return conversion_array
    
    print 'building the prioritization from stream raster'
    raster_utils.vectorize_datasets(
        [parameters['distance_from_stream_filename'], parameters['lulc_filename']],
        valid_distance, conversion_priority_filename, gdal.GDT_Float32,
        conversion_nodata, conversion_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False)

    #build iterator
    priority_pixels = disk_sort.sort_to_disk(conversion_priority_filename, 0)
    lulc_ds = gdal.Open(parameters['lulc_filename'])
    lulc_band = lulc_ds.GetRasterBand(1)
    lulc_array = lulc_band.ReadAsArray()
    output_lulc_list = []
    for step_index in range(parameters['number_of_steps']):
        print 'making lulc %d' % step_index

        converted_pixels = 0
        if step_index != 0:
            for value, flat_index, _ in priority_pixels:
            
                if (value * direction_factor) < stream_buffer_width:
                    continue
            
                if value == -conversion_nodata:
                    print 'all pixels converted, breaking loop'
                    break
                numpy.reshape(lulc_array, -1)[flat_index] = (
                    parameters['convert_to_lulc_code'])
                converted_pixels += 1
                if converted_pixels >= parameters['pixels_per_step_to_convert']:
                    break
        
        print 'saving lulc %d' % step_index
        output_lulc_uri = os.path.join(
            parameters['temporary_file_directory'],
            '%s_%d.tif' % (base_name, step_index))
        raster_utils.new_raster_from_base_uri(
            parameters['lulc_filename'], output_lulc_uri, 'GTiff', lulc_nodata,
            gdal.GDT_Int32)
        output_lulc_ds = gdal.Open(output_lulc_uri, gdal.GA_Update)
        output_lulc_band = output_lulc_ds.GetRasterBand(1)
        output_lulc_band.WriteArray(lulc_array)
        output_lulc_list.append(output_lulc_uri)
    return output_lulc_list

    
def run_sediment_analysis(parameters, land_cover_uri_list, summary_table_uri):
    sed_export_table_uri = os.path.join(
        parameters['output_file_directory'], summary_table_uri)
    sed_export_table = open(sed_export_table_uri, 'w')
    sed_export_table.write('step,value\n')
    for index, lulc_uri in enumerate(land_cover_uri_list):
        sdr_args = {
            'workspace_dir': parameters['output_file_directory'],
            'suffix': str(index),
            'dem_uri': parameters['dem_filename'],
            'erosivity_uri': parameters['erosivity_uri'],
            'erodibility_uri': parameters['erodibility_uri'],
            'landuse_uri': lulc_uri,
            'watersheds_uri': parameters['watersheds_uri'],
            'biophysical_table_uri': parameters['biophysical_table_uri'],
            'threshold_flow_accumulation': parameters['flow_accumulation_threshold_for_streams'],
            'k_param': 2,
            'sdr_max': 0.8,
            'ic_0_param': 0.5,
        }
        invest_natcap.sdr.sdr.execute(sdr_args)
        sdr_export_uri = os.path.join(sdr_args['workspace_dir'], 'output', "sed_export_%d.tif" % index)
        sed_export_ds = gdal.Open(sdr_export_uri)
        sed_export_band = sed_export_ds.GetRasterBand(1)
        nodata = raster_utils.get_nodata_from_uri(sdr_export_uri)
        sed_export_total = 0.0
        for row_index in xrange(sed_export_ds.RasterYSize):
            sed_array = sed_export_band.ReadAsArray(
                0, row_index, sed_export_ds.RasterXSize, 1)
            sed_export_total += numpy.sum(sed_array[(sed_array != nodata) & (~numpy.isnan(sed_array))])
        sed_export_table.write('%d,%f\n' % (index, sed_export_total))
        
        
if __name__ == '__main__':
    PARAMETERS = {
        #'dem_filename': 'C:/Users/rich/Dropbox/unilever_data/mg_dem_90f/w001001.adf',
        #'lulc_filename': 'C:/Users/rich/Dropbox/unilever_data/lulc_2008.tif',
        'dem_filename': "C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Freshwater/dem/w001001.adf",
        'lulc_filename': 'C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Terrestrial/landuse_90/w001001.adf',
        'erosivity_uri': "C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Freshwater/erosivity/w001001.adf",
        'erodibility_uri': "C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Freshwater/erodibility/w001001.adf",
        'watersheds_uri': "C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Freshwater/watersheds.shp",
        'biophysical_table_uri': "C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Freshwater/biophysical_table.csv",
        'flow_accumulation_threshold_for_streams': 1000,
        'convert_from_lulc_codes': range(49, 67) + [95, 98], #read from biophysical table
        'convert_to_lulc_code':82, #this is 'field crop'
        'pixels_per_step_to_convert': 100000,
        'number_of_steps': 10,
        'temporary_file_directory': 'temp',
        'output_file_directory': 'output',
    }
    initialize_simulation(PARAMETERS)
    for MODE, FILENAME, BUFFER in [("to_stream", "to_stream", 0),
        ("from_stream", "from_stream", 0),
        ("from_stream", "from_stream_with_buffer_1", 1),
        ("from_stream", "from_stream_with_buffer_3", 3),
        ("from_stream", "from_stream_with_buffer_9", 9)]:
        #make the filename the mode, thus mode is passed in twice
        LAND_COVER_URI_LIST = step_land_change_from_streams(PARAMETERS, FILENAME, MODE, BUFFER)
        run_sediment_analysis(PARAMETERS, LAND_COVER_URI_LIST, FILENAME + ".csv")
