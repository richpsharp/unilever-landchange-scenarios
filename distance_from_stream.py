import os

from invest_natcap.routing import routing_utils

if __name__ == '__main__':
    parameters = {
        'dem_filename': 'C:/Users/rich/Dropbox/unilever_data/mg_dem_90f',
        'lulc_filename': 'C:/Users/rich/Dropbox/unilever_data/lulc_2008.tif',
        'convert_from_lulc_codes': [2],
        'convert_to_lulc_code': 18,
        'pixels_per_step_to_convert': 100,
        'number_of_steps': 10,
        'temporary_file_directory': 'temp',
        'output_file_directory': 'output',
    }
    
    for dirname in [parameters['temporary_file_directory'],
        parameters['output_file_directory']]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
