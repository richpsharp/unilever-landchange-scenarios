import os
import hashlib
import pickle

import gdal
import numpy

from invest_natcap.routing import routing_utils
from invest_natcap import raster_utils

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

 
def step_land_change(parameters):
    conversion_priority_filename = os.path.join(
        parameters['temporary_file_directory'], 'conversion_priority.tif')
    conversion_nodata = -1
    conversion_pixel_size = raster_utils.get_cell_size_from_uri(
        parameters['lulc_filename'])
    lulc_nodata = raster_utils.get_nodata_from_uri(parameters['lulc_filename'])
        
    def valid_distance(distance, lulc):
        conversion_array = numpy.empty(distance.shape, dtype=numpy.float32)
        conversion_array[:] = conversion_nodata
        for convert_code in parameters['convert_from_lulc_codes']:
            mask = (lulc == convert_code)
            conversion_array[mask] = distance[mask]
    
        return conversion_array
    
    print 'building the prioritization from stream raster'
    raster_utils.vectorize_datasets(
        [parameters['distance_from_stream_filename'], parameters['lulc_filename']],
        valid_distance, conversion_priority_filename, gdal.GDT_Float32,
        conversion_nodata, conversion_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False)



if __name__ == '__main__':
    PARAMETERS = {
        #'dem_filename': 'C:/Users/rich/Dropbox/unilever_data/mg_dem_90f/w001001.adf',
        #'lulc_filename': 'C:/Users/rich/Dropbox/unilever_data/lulc_2008.tif',
        'dem_filename': "C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Freshwater/dem/w001001.adf",
        'lulc_filename': 'C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Terrestrial/landuse_90/w001001.adf',
        'flow_accumulation_threshold_for_streams': 1000,
        'convert_from_lulc_codes': range(49, 67) + [95, 98],
        'convert_to_lulc_code': 18,
        'pixels_per_step_to_convert': 100,
        'number_of_steps': 10,
        'temporary_file_directory': 'temp',
        'output_file_directory': 'output',
    }
    initialize_simulation(PARAMETERS)
    step_land_change(PARAMETERS)
    
