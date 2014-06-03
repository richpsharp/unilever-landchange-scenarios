import os
import hashlib
import pickle

from invest_natcap.routing import routing_utils


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


def run_simulation(parameters):
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
            parameters['stream_uri'] = previous_parameters['stream_uri']
    
    if 'stream_uri' not in parameters:
        parameters['flow_direction_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'flow_direction.tif')
        parameters['flow_accumulation_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'flow_accumulation.tif')
        parameters['stream_uri'] = os.path.join(
            parameters['temporary_file_directory'], 'streams.tif')
        
        print 'resolving filling pits'
        dem_pit_filled_uri =  os.path.join(
            parameters['temporary_file_directory'], 'pit_filled_dem.tif')
        routing_utils.fill_pits(parameters['dem_filename'], dem_pit_filled_uri)
        
        print 'resolving plateaus'
        dem_plateau_resolved_uri =  os.path.join(
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
    else:
        print 'streams already calculated, use those'
        
    save_obj(previous_run_file, parameters)


if __name__ == '__main__':
    PARAMETERS = {
        'dem_filename': 'C:/Users/rich/Dropbox/unilever_data/mg_dem_90f/w001001.adf',
        #'dem_filename': "C:/InVEST_dev39_3_0_1 [6d541e569a05]_x86/Base_Data/Freshwater/dem/w001001.adf",
        'flow_accumulation_threshold_for_streams': 1000, 
        'lulc_filename': 'C:/Users/rich/Dropbox/unilever_data/lulc_2008.tif',
        'convert_from_lulc_codes': [2],
        'convert_to_lulc_code': 18,
        'pixels_per_step_to_convert': 100,
        'number_of_steps': 10,
        'temporary_file_directory': 'temp',
        'output_file_directory': 'output',
    }
    run_simulation(PARAMETERS)
    
    
    
    