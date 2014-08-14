import os
import hashlib
import pickle
import threading

import gdal
import numpy
import pympler.tracker

from invest_natcap.routing import routing_utils
from invest_natcap import raster_utils
from invest_natcap.scenario_generator import disk_sort
import invest_natcap.sdr.sdr


class PerpetualTimer():
   
    def __init__(self, time_to_wait, callback):
       self.time_to_wait = time_to_wait
       self.callback = callback
       self.thread = threading.Timer(self.time_to_wait, self.process_callback)

    def process_callback(self):
       self.callback()
       self.thread = threading.Timer(self.time_to_wait, self.process_callback)
       self.thread.start()

    def start(self):
       self.thread.start()

    def cancel(self):
       self.thread.cancel()


def memory_report():
    summary_tracker = pympler.tracker.SummaryTracker()
    summary_tracker.print_diff() 
    
      
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
    parameters['dem_hash'] = hashfile(parameters['dem_uri'])
    parameters['lulc_hash'] = hashfile(parameters['landuse_uri'])
    
    previous_run_file = os.path.join(
        parameters['temporary_file_directory'], 'previous_run.obj')
    if os.path.isfile(previous_run_file):
        previous_parameters = load_obj(previous_run_file)
        
        for parameter in [
                'dem_uri', 'landuse_uri', 'dem_hash', 'lulc_hash', 
                'threshold_flow_accumulation']:
            if previous_parameters[parameter] != parameters[parameter]:
                break
        else:
            parameters['distance_from_stream_filename'] = (
                previous_parameters['distance_from_stream_filename'])
            parameters['distance_from_forest_edge_filename'] = (
                previous_parameters['distance_from_forest_edge_filename'])
                
    
    if 'distance_from_stream_filename' not in parameters:
        parameters['flow_direction_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'flow_direction.tif')
        parameters['flow_accumulation_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'flow_accumulation.tif')
        parameters['stream_uri'] = os.path.join(
            parameters['temporary_file_directory'], 'streams.tif')
        parameters['forest_uri'] = os.path.join(
            parameters['temporary_file_directory'], 'forest.tif')
        parameters['distance_from_stream_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'distance_from_stream.tif')
        parameters['distance_from_forest_edge_filename'] = os.path.join(
            parameters['temporary_file_directory'], 'distance_from_forest_edge.tif')
        
        print 'resolving filling pits'
        dem_pit_filled_uri = os.path.join(
            parameters['temporary_file_directory'], 'pit_filled_dem.tif')
        routing_utils.fill_pits(parameters['dem_uri'], dem_pit_filled_uri)
        
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
            parameters['threshold_flow_accumulation'],
            parameters['stream_uri'])
            
        print 'calculate distance from streams'
        raster_utils.distance_transform_edt(
            parameters['stream_uri'],
            parameters['distance_from_stream_filename'])
            
        
        forest_pixel_size = raster_utils.get_cell_size_from_uri(
            parameters['landuse_uri'])
        lulc_nodata = raster_utils.get_nodata_from_uri(parameters['landuse_uri'])
        
        forest_nodata = 255
        def classify_forest(lulc):
            forest_mask = numpy.empty(lulc.shape)
            forest_mask[:] = 1
            for lulc_code in parameters['convert_from_lulc_codes']:
                lulc_mask = (lulc == lulc_code)
                forest_mask[lulc_mask] = 0
            return numpy.where(lulc == lulc_nodata, forest_nodata, forest_mask)
        
        raster_utils.vectorize_datasets(
            [parameters['landuse_uri']],
            classify_forest, parameters['forest_uri'], gdal.GDT_Byte,
            forest_nodata, forest_pixel_size, 'intersection',
            dataset_to_align_index=0, vectorize_op=False)
        
        
        print 'calculate distance from forest'
        raster_utils.distance_transform_edt(
            parameters['forest_uri'],
            parameters['distance_from_forest_edge_filename'])
        
    else:
        print 'streams already calculated, use those'
        
    save_obj(previous_run_file, parameters)


def step_land_change(
    parameters, base_name, mode, stream_buffer_width):
    
    if mode in ['to_stream', 'from_stream']:
        return step_land_change_streams(
            parameters, base_name, mode, stream_buffer_width)
    elif mode in ['core', 'edge']:
        return step_land_change_forest(
            parameters, base_name, mode, stream_buffer_width)
    else:
        raise Exception("Unknown mode %s" % mode)
            
def step_land_change_forest(
    parameters, base_name, mode, stream_buffer_width):
    
    if mode == "core":
        direction_factor = -1
    elif mode == "edge":
        direction_factor = 1
    conversion_priority_filename = os.path.join(
        parameters['temporary_file_directory'], 'conversion_priority.tif')
    conversion_nodata = direction_factor * 9999
    conversion_pixel_size = raster_utils.get_cell_size_from_uri(
        parameters['landuse_uri'])
    lulc_nodata = raster_utils.get_nodata_from_uri(parameters['landuse_uri'])
        
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
        [parameters['distance_from_forest_edge_filename'], parameters['landuse_uri']],
        valid_distance, conversion_priority_filename, gdal.GDT_Float32,
        conversion_nodata, conversion_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False)

    #build iterator
    priority_pixels = disk_sort.sort_to_disk(conversion_priority_filename, 0)
    lulc_ds = gdal.Open(parameters['landuse_uri'])
    lulc_band = lulc_ds.GetRasterBand(1)
    lulc_array = lulc_band.ReadAsArray()
    output_lulc_list = []
    for step_index in range(parameters['number_of_steps']):
        print 'making lulc %d' % step_index

        converted_pixels = 0
        if step_index != 0:
            for value, flat_index, _ in priority_pixels:
            
                if value == -conversion_nodata:
                    print 'all pixels converted, breaking loop'
                    break
                numpy.reshape(lulc_array, -1)[flat_index] = (
                    parameters['convert_to_lulc_code'])
                converted_pixels += 1
                if converted_pixels >= parameters['pixels_per_step_to_convert']:
                    break
        
        if converted_pixels == 0 and step_index != 0:
            print 'everything converted already, breaking loop'
            break
        
        print 'saving lulc %d' % step_index
        output_lulc_uri = os.path.join(
            parameters['temporary_file_directory'],
            '%s_%d.tif' % (base_name, step_index))
        raster_utils.new_raster_from_base_uri(
            parameters['landuse_uri'], output_lulc_uri, 'GTiff', lulc_nodata,
            gdal.GDT_Int32)
        output_lulc_ds = gdal.Open(output_lulc_uri, gdal.GA_Update)
        output_lulc_band = output_lulc_ds.GetRasterBand(1)
        output_lulc_band.WriteArray(lulc_array)
        output_lulc_list.append(output_lulc_uri)
    return output_lulc_list

def step_land_change_streams(
    parameters, base_name, mode, stream_buffer_width):
    """
        parameters - the context from the main function
        base_name - base of the filename
        mode - one of "to_stream" or "from_stream",
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
        parameters['landuse_uri'])
    lulc_nodata = raster_utils.get_nodata_from_uri(parameters['landuse_uri'])
        
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
        [parameters['distance_from_stream_filename'], parameters['landuse_uri']],
        valid_distance, conversion_priority_filename, gdal.GDT_Float32,
        conversion_nodata, conversion_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False)

    #build iterator
    priority_pixels = disk_sort.sort_to_disk(conversion_priority_filename, 0)
    lulc_ds = gdal.Open(parameters['landuse_uri'])
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
        
        if converted_pixels == 0 and step_index != 0:
            print 'everything converted already, breaking loop'
            break
        
        print 'saving lulc %d' % step_index
        output_lulc_uri = os.path.join(
            parameters['temporary_file_directory'],
            '%s_%d.tif' % (base_name, step_index))
        raster_utils.new_raster_from_base_uri(
            parameters['landuse_uri'], output_lulc_uri, 'GTiff', lulc_nodata,
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
    sed_export_table.write('step,%s\n' % os.path.splitext(summary_table_uri)[0])
    for index, lulc_uri in enumerate(land_cover_uri_list):
        sdr_args = {
            'workspace_dir': parameters['output_file_directory'],
            'suffix': str(index),
            'dem_uri': parameters['dem_uri'],
            'erosivity_uri': parameters['erosivity_uri'],
            'erodibility_uri': parameters['erodibility_uri'],
            'landuse_uri': lulc_uri,
            'watersheds_uri': parameters['watersheds_uri'],
            'biophysical_table_uri': parameters['biophysical_table_uri'],
            'threshold_flow_accumulation': parameters['threshold_flow_accumulation'],
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
        memory_report()
        
if __name__ == '__main__':
    DROPBOX_FOLDER = u'C:/Users/rich/Documents/Dropbox/'
    OUTPUT_FOLDER = u'C:/Users/rich/Documents/unilever_iowa_outputs'
    TEMPORARY_FOLDER = os.path.join(OUTPUT_FOLDER, 'temp')

    for tmp_variable in ['TMP', 'TEMP', 'TMPDIR']:

        if tmp_variable in os.environ:
            print 'Updating os.environ["%s"]=%s to %s' % (tmp_variable, os.environ[tmp_variable], TEMPORARY_FOLDER)
        else:
            print 'Setting os.environ["%s"]=%s' % (tmp_variable, TEMPORARY_FOLDER)

        os.environ[tmp_variable] = TEMPORARY_FOLDER
    
    PARAMETERS = {
        'convert_from_lulc_codes': range(1, 5), #read from biophysical table
        'convert_to_lulc_code':12, #this is 'field crop'
        'pixels_per_step_to_convert': 100000,
        
        'temporary_file_directory': TEMPORARY_FOLDER,
        'output_file_directory': OUTPUT_FOLDER,
    }
    
    mg_args = {
        u'convert_from_lulc_codes': range(1, 5), #read from biophysical table
        u'convert_to_lulc_code':12, #this is 'field crop'
        u'pixels_per_step_to_convert': 100000,
        u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/biophysical_coeffs_Brazil_Unilever.csv"),
        u'dem_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/DEM_SRTM_MT_filled.tif"),
        u'erodibility_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/erodibility_MT.tif"),
        u'erosivity_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/erosivity_MT.tif"),
        u'ic_0_param': u'0.5',
        u'k_param': u'2',
        u'landuse_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/LULC_MCD12Q1_2012_MT.tif"),
        u'sdr_max': u'0.8',
        u'threshold_flow_accumulation': 1000,
        u'watersheds_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/Mato_Grosso.shp"),
        u'workspace_dir': os.path.join(OUTPUT_FOLDER, u'Mato_Grosso_global/'),
        u'suffix': 'mato_grosso',
    }
    mg_args.update(PARAMETERS)
    
    iowa_national_args = {
        u'convert_from_lulc_codes': [41, 42, 43, 90], #these are the forest types
        u'convert_to_lulc_code':82, #this is 'cultivated crops'
        u'pixels_per_step_to_convert': 2616659/20, #this is the number of forest pixels divided by number of steps
        u'number_of_steps': 20,
        u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/biophysical_coeffs_Iowa_Unilever_national.csv"),
        u'dem_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/DEM_SRTM_Iowa_HUC8_v2_uncompressed_striped.tif"),
        u'erodibility_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/erodibility_STATSGO_Iowa_HUC8.tif"),
        u'erosivity_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/erosivity_Iowa_HUC8.tif"),
        u'ic_0_param': u'0.5',
        u'k_param': u'2',
        u'landuse_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/LULC_NLCD_2006_Iowa_HUC8_uncompressed.tif"),
        u'sdr_max': u'0.8',
        u'threshold_flow_accumulation': u'1000',
        u'watersheds_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/HUC8_Iowa_intersect_dissolve.shp"),
        u'workspace_dir': os.path.join(OUTPUT_FOLDER, u'Iowa_national'),
        u'suffix': 'iowa_local',
    }
    iowa_national_args.update(PARAMETERS)
    
    iowa_global_args = {
        u'convert_from_lulc_codes': range(1, 5), #forest lulcs from biophysical table
        u'convert_to_lulc_code':12, #this is 'field crop'
        u'pixels_per_step_to_convert': 9772/20, #this is the number of forest pixels divided by number of steps
        u'number_of_steps': 20,
        u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/biophysical_coeffs_Iowa_Unilever_global.csv"),
        u'dem_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/DEM_SRTM_Iowa_HUC8_v2_uncompressed_striped.tif"),
        u'erodibility_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/erodibility_HWSD_Iowa_HUC8.tif"),
        u'erosivity_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/erosivity_Iowa_HUC8.tif"),
        u'ic_0_param': u'0.5',
        u'k_param': u'2',
        u'landuse_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/LULC_MCD12Q1_2006_Iowa_HUC8.tif"),
        u'sdr_max': u'0.8',
        u'threshold_flow_accumulation': u'1000',
        u'watersheds_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/HUC8_Iowa_intersect_dissolve.shp"),
        u'workspace_dir': os.path.join(OUTPUT_FOLDER, u'Iowa_global'),
        u'suffix': 'iowa_global',
    }
    iowa_global_args.update(PARAMETERS)
    
    #summary_reporter = PerpetualTimer(1.0, memory_report)
    #summary_reporter.start()
    
    for args, simulation in [(iowa_national_args, 'iowa_national_'), (iowa_global_args, 'iowa_global_')]:
    
        initialize_simulation(args)
        print 'preparing sdr'
        args['_prepare'] = invest_natcap.sdr.sdr._prepare(**args)
        for MODE, FILENAME, BUFFER in [
            ("core", "core", 0),
            ("edge", "edge", 0),
            ("to_stream", "to_stream", 0),
            ("from_stream", "from_stream", 0),
            #("from_stream", "from_stream_with_buffer_1", 1),
            #("from_stream", "from_stream_with_buffer_2", 2),
            #("from_stream", "from_stream_with_buffer_3", 3),
            #("from_stream", "from_stream_with_buffer_9", 9)
            ]:
            #make the filename the mode, thus mode is passed in twice
            LAND_COVER_URI_LIST = step_land_change(args, simulation+FILENAME, MODE, BUFFER)
            run_sediment_analysis(args, LAND_COVER_URI_LIST, simulation+FILENAME + ".csv")

    #summary_reporter.cancel()
