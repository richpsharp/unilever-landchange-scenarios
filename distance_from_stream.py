import os
import hashlib
import threading
import shutil
import math
import multiprocessing

import gdal
import numpy

from invest_natcap.routing import routing_utils
from invest_natcap import raster_utils
from invest_natcap.scenario_generator import disk_sort
import invest_natcap.sdr.sdr
import invest_natcap.ndr.ndr


def lowpriority():
    """ Set the priority of the process to below-normal."""

    import sys
    try:
        sys.getwindowsversion()
    except:
        is_windows = False
    else:
        is_windows = True

    if is_windows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api, win32process, win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)
    else:
        import os
        os.nice(1)

      

def initialize_simulation(parameters):
    raster_utils.create_directories([parameters['temporary_file_directory'],
        parameters['output_file_directory'],
        parameters['land_use_directory']])
    
    parameters['flow_direction_filename'] = os.path.join(
        parameters['temporary_file_directory'], 'flow_direction.tif')
    parameters['tiled_dem_uri'] = os.path.join(
        parameters['temporary_file_directory'], 'tiled_dem.tif')
    parameters['flow_accumulation_filename'] = os.path.join(
        parameters['temporary_file_directory'], 'flow_accumulation.tif')
    parameters['stream_uri'] = os.path.join(
        parameters['temporary_file_directory'], 'streams.tif')
    parameters['non_forest_uri'] = os.path.join(
        parameters['temporary_file_directory'], 'non_forest.tif')
    parameters['non_ag_uri'] = os.path.join(
        parameters['temporary_file_directory'], 'non_ag.tif')
    parameters['distance_from_stream_filename'] = os.path.join(
        parameters['temporary_file_directory'], 'distance_from_stream.tif')
    parameters['distance_from_forest_edge_filename'] = os.path.join(
        parameters['temporary_file_directory'], 'distance_from_forest_edge.tif')
    parameters['distance_from_ag_edge_filename'] = os.path.join(
        parameters['temporary_file_directory'], 'distance_from_ag_edge.tif')

    aligned_lulc_uri =  os.path.join(
        parameters['temporary_file_directory'], 'aligned_lulc.tif')

    out_pixel_size = raster_utils.get_cell_size_from_uri(parameters['dem_uri'])
    tmp_dem_uri = raster_utils.temporary_filename()
    raster_utils.align_dataset_list(
        [parameters['dem_uri'], args['landuse_uri']], [tmp_dem_uri, aligned_lulc_uri],
        ['nearest'] * 2, out_pixel_size, 'dataset',
        0, dataset_to_bound_index=0, aoi_uri=args['watersheds_uri'])
    os.remove(tmp_dem_uri)

    parameters['landuse_uri'] = aligned_lulc_uri

    dem_ds = gdal.Open(parameters['dem_uri'])
    dem_band = dem_ds.GetRasterBand(1)
    dem_pixel_size = raster_utils.get_cell_size_from_uri(parameters['dem_uri'])

    masked_dem_uri = os.path.join(
        parameters['temporary_file_directory'], 'masked_dem.tif')

    raster_utils.vectorize_datasets(
        [parameters['dem_uri']],
        lambda x: x, masked_dem_uri, dem_band.DataType,
        dem_band.GetNoDataValue(), dem_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False,
        aoi_uri=parameters['watersheds_uri'])

    raster_utils.tile_dataset_uri(masked_dem_uri, parameters['tiled_dem_uri'], 256)
    print 'resolving filling pits'
    dem_pit_filled_uri = os.path.join(
        parameters['temporary_file_directory'], 'pit_filled_dem.tif')
    routing_utils.fill_pits(parameters['tiled_dem_uri'], dem_pit_filled_uri)
    
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
    def classify_non_forest(lulc):
        forest_mask = numpy.empty(lulc.shape)
        forest_mask[:] = 1
        for lulc_code in parameters['convert_from_lulc_codes']:
            lulc_mask = (lulc == lulc_code)
            forest_mask[lulc_mask] = 0
        return numpy.where(lulc == lulc_nodata, forest_nodata, forest_mask)
    
    raster_utils.vectorize_datasets(
        [parameters['landuse_uri']],
        classify_non_forest, parameters['non_forest_uri'], gdal.GDT_Byte,
        forest_nodata, forest_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False,
        aoi_uri=parameters['watersheds_uri'])
    
    print 'calculate distance from forest edge'
    raster_utils.distance_transform_edt(
        parameters['non_forest_uri'],
        parameters['distance_from_forest_edge_filename'])

    ag_nodata = 255
    def classify_ag(lulc):
        ag_mask = numpy.empty(lulc.shape)
        ag_mask[:] = 0
        ag_mask[lulc ==  parameters['convert_to_lulc_code']] = 1
        return numpy.where(lulc == lulc_nodata, ag_nodata, ag_mask)

    raster_utils.vectorize_datasets(
        [parameters['landuse_uri']],
        classify_ag, parameters['non_ag_uri'], gdal.GDT_Byte,
        forest_nodata, forest_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False,
        aoi_uri=parameters['watersheds_uri'])

    print 'calculate distance from agriculture edge'
    raster_utils.distance_transform_edt(
        parameters['non_ag_uri'],
        parameters['distance_from_ag_edge_filename'])


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

def step_land_change(
    parameters, base_name, mode, stream_buffer_width):
    
    if mode in ['to_stream', 'from_stream']:
        return step_land_change_streams(
            parameters, base_name, mode, stream_buffer_width)
    elif mode in ['core', 'edge']:
        return step_land_change_forest(
            parameters, base_name, mode, stream_buffer_width)
    elif mode in ['fragmentation']:
        return step_land_change_fragmentation(
            parameters, base_name, mode)
    elif mode in ['ag']:
        return step_land_change_ag(
            parameters, base_name, mode, stream_buffer_width)
    else:
        raise Exception("Unknown mode %s" % mode)
    

def step_land_change_ag(
    parameters, base_name, mode, stream_buffer_width):
    
    conversion_priority_filename = os.path.join(
        parameters['temporary_file_directory'], 'conversion_priority_%s.tif' % base_name)
    conversion_nodata = 9999

    conversion_pixel_size = raster_utils.get_cell_size_from_uri(
        parameters['landuse_uri'])

    aligned_distance_from_ag_edge_filename = os.path.join(
        parameters['temporary_file_directory'], 'aligned_distance_from_ag_edge_%s.tif' % base_name)
    aligned_landuse_uri = os.path.join(
        parameters['temporary_file_directory'], 'aligned_landuse_%s.tif' % base_name)

    raster_utils.align_dataset_list(
        [parameters['distance_from_ag_edge_filename'], parameters['landuse_uri']], 
        [aligned_distance_from_ag_edge_filename, aligned_landuse_uri], ['nearest']*2,
        conversion_pixel_size, 'intersection', 0,
        dataset_to_bound_index=None, aoi_uri=parameters['watersheds_uri'])

    
    lulc_nodata = raster_utils.get_nodata_from_uri(aligned_landuse_uri)
    distance_nodata = raster_utils.get_nodata_from_uri(parameters['distance_from_ag_edge_filename'])

    def valid_distance(distance, lulc):
        conversion_array = numpy.empty(distance.shape, dtype=numpy.float32)
        conversion_array[:] = conversion_nodata
        for convert_code in parameters['convert_from_lulc_codes']:
            mask = (lulc == convert_code)
            #invert the distance for sorting
            conversion_array[mask] = -distance[mask]
        return numpy.where(distance != distance_nodata, conversion_array, conversion_nodata)
    
    print 'building the prioritization from stream raster'
    raster_utils.vectorize_datasets(
        [aligned_distance_from_ag_edge_filename, aligned_landuse_uri],
        valid_distance, conversion_priority_filename, gdal.GDT_Float32,
        conversion_nodata, conversion_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False, aoi_uri=parameters['watersheds_uri'])

    pixels_per_step_to_convert = calculate_pixels_per_step_for_full_conversion(
        aligned_landuse_uri, parameters['convert_from_lulc_codes'], parameters['number_of_steps'])
    
    #build iterator
    priority_pixels = disk_sort.sort_to_disk(conversion_priority_filename, 0)
    lulc_ds = gdal.Open(aligned_landuse_uri)
    lulc_band = lulc_ds.GetRasterBand(1)
    lulc_array = lulc_band.ReadAsArray()
    output_lulc_list = []

    for step_index in range(parameters['number_of_steps']+1):
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
                if converted_pixels >= pixels_per_step_to_convert:
                    break
        
        if converted_pixels == 0 and step_index != 0:
            print 'everything converted already, breaking loop'
            break
        
        print 'saving lulc %d, converted pixels %d' % (step_index, converted_pixels)
        output_lulc_uri = os.path.join(parameters['land_use_directory'],
            '%s_%d.tif' % (base_name, step_index))
        raster_utils.new_raster_from_base_uri(
            aligned_distance_from_ag_edge_filename, output_lulc_uri, 'GTiff', lulc_nodata,
            gdal.GDT_Int32)
        output_lulc_ds = gdal.Open(output_lulc_uri, gdal.GA_Update)
        output_lulc_band = output_lulc_ds.GetRasterBand(1)
        output_lulc_band.WriteArray(lulc_array)
        output_lulc_list.append(output_lulc_uri)
    return output_lulc_list


def step_land_change_fragmentation(
    parameters, base_name, mode):
    
    direction_factor = -1
    raster_utils.create_directories([parameters['workspace_dir']])
    conversion_priority_filename = os.path.join(
        parameters['temporary_file_directory'], 'conversion_priority_%s.tif' % base_name)
    non_forest_uri = os.path.join(
        parameters['workspace_dir'], 'non_forest.tif')
    distance_from_forest_uri = os.path.join(
        parameters['workspace_dir'], 'distance_from_forest.tif')
    conversion_nodata = direction_factor * 9999
    conversion_pixel_size = raster_utils.get_cell_size_from_uri(
        parameters['landuse_uri'])
    aligned_distance_from_forest_edge_filename = os.path.join(
        parameters['temporary_file_directory'], 'aligned_distance_from_forest_edge_%s.tif' % base_name)
    aligned_landuse_uri = os.path.join(
        parameters['temporary_file_directory'], 'aligned_landuse_%s.tif' % base_name)
    raster_utils.align_dataset_list(
        [parameters['distance_from_forest_edge_filename'], parameters['landuse_uri']], 
        [aligned_distance_from_forest_edge_filename, aligned_landuse_uri], ['nearest']*2,
        conversion_pixel_size, 'intersection', 0,
        dataset_to_bound_index=None, aoi_uri=parameters['watersheds_uri'])
    lulc_nodata = raster_utils.get_nodata_from_uri(aligned_landuse_uri)
    distance_nodata = raster_utils.get_nodata_from_uri(parameters['distance_from_forest_edge_filename'])

    def valid_distance(distance, lulc):
        conversion_array = numpy.empty(distance.shape, dtype=numpy.float32)
        conversion_array[:] = conversion_nodata
        for convert_code in parameters['convert_from_lulc_codes']:
            mask = (lulc == convert_code)
            #invert the distance for sorting
            conversion_array[mask] = -distance[mask] * direction_factor
        return numpy.where(distance != distance_nodata, conversion_array, conversion_nodata)
    
    output_lulc_list = []
    previous_aligned_landuse_uri = aligned_landuse_uri

    pixels_per_step_to_convert = calculate_pixels_per_step_for_full_conversion(
        aligned_landuse_uri, parameters['convert_from_lulc_codes'], parameters['number_of_steps'])

    for step_index in range(parameters['number_of_steps']+1):
        print step_index
        forest_nodata = 255
        def classify_non_forest(lulc):
            forest_mask = numpy.empty(lulc.shape)
            forest_mask[:] = 1
            for lulc_code in parameters['convert_from_lulc_codes']:
                lulc_mask = (lulc == lulc_code)
                forest_mask[lulc_mask] = 0
            return numpy.where(lulc == lulc_nodata, forest_nodata, forest_mask)
        print previous_aligned_landuse_uri
        raster_utils.vectorize_datasets(
            [previous_aligned_landuse_uri],
            classify_non_forest, non_forest_uri, gdal.GDT_Byte,
            forest_nodata, conversion_pixel_size, 'intersection',
            dataset_to_align_index=0, vectorize_op=False,
            aoi_uri=parameters['watersheds_uri'])
        raster_utils.distance_transform_edt(
            non_forest_uri, distance_from_forest_uri)

        print 'making lulc %d' % step_index

        print 'building the prioritization from forest edge'
        raster_utils.vectorize_datasets(
            [distance_from_forest_uri, previous_aligned_landuse_uri],
            valid_distance, conversion_priority_filename, gdal.GDT_Float32,
            conversion_nodata, conversion_pixel_size, 'intersection',
            dataset_to_align_index=0, vectorize_op=False, aoi_uri=parameters['watersheds_uri'])

        #build iterator
        priority_pixels = disk_sort.sort_to_disk(conversion_priority_filename, 0)
        lulc_ds = gdal.Open(previous_aligned_landuse_uri)
        lulc_band = lulc_ds.GetRasterBand(1)
        lulc_array = lulc_band.ReadAsArray()
        
        converted_pixels = 0
        if step_index != 0:
            for value, flat_index, _ in priority_pixels:
            
                if value == -conversion_nodata:
                    print 'all pixels converted, breaking loop'
                    break
                numpy.reshape(lulc_array, -1)[flat_index] = (
                    parameters['convert_to_lulc_code'])
                converted_pixels += 1
                if converted_pixels >= pixels_per_step_to_convert:
                    break
        
        if converted_pixels == 0 and step_index != 0:
            print 'everything converted already, breaking loop'
            break
        
        print 'saving lulc %d' % step_index
        output_lulc_uri = os.path.join(parameters['land_use_directory'],
            '%s_%d.tif' % (base_name, step_index))
        previous_aligned_landuse_uri = output_lulc_uri
        raster_utils.new_raster_from_base_uri(
            aligned_distance_from_forest_edge_filename, output_lulc_uri, 'GTiff', lulc_nodata,
            gdal.GDT_Int32)
        output_lulc_ds = gdal.Open(output_lulc_uri, gdal.GA_Update)
        output_lulc_band = output_lulc_ds.GetRasterBand(1)
        output_lulc_band.WriteArray(lulc_array)
        output_lulc_band.FlushCache()

        output_lulc_list.append(output_lulc_uri)
    return output_lulc_list

            
def step_land_change_forest(
    parameters, base_name, mode, stream_buffer_width):
    
    if mode == "core":
        direction_factor = -1
    elif mode == "edge":
        direction_factor = 1
    conversion_priority_filename = os.path.join(
        parameters['temporary_file_directory'], 'conversion_priority_%s.tif' % base_name)
    conversion_nodata = direction_factor * 9999

    conversion_pixel_size = raster_utils.get_cell_size_from_uri(
        parameters['landuse_uri'])

    aligned_distance_from_forest_edge_filename = os.path.join(
        parameters['temporary_file_directory'], 'aligned_distance_from_forest_edge_%s.tif' % base_name)
    aligned_landuse_uri = os.path.join(
        parameters['temporary_file_directory'], 'aligned_landuse_%s.tif' % base_name)

    raster_utils.align_dataset_list(
        [parameters['distance_from_forest_edge_filename'], parameters['landuse_uri']], 
        [aligned_distance_from_forest_edge_filename, aligned_landuse_uri], ['nearest']*2,
        conversion_pixel_size, 'intersection', 0,
        dataset_to_bound_index=None, aoi_uri=parameters['watersheds_uri'])

    
    lulc_nodata = raster_utils.get_nodata_from_uri(aligned_landuse_uri)
    distance_nodata = raster_utils.get_nodata_from_uri(parameters['distance_from_forest_edge_filename'])

    def valid_distance(distance, lulc):
        conversion_array = numpy.empty(distance.shape, dtype=numpy.float32)
        conversion_array[:] = conversion_nodata
        for convert_code in parameters['convert_from_lulc_codes']:
            mask = (lulc == convert_code)
            #invert the distance for sorting
            conversion_array[mask] = -distance[mask] * direction_factor
        return numpy.where(distance != distance_nodata, conversion_array, conversion_nodata)
    
    print 'building the prioritization from stream raster'
    raster_utils.vectorize_datasets(
        [aligned_distance_from_forest_edge_filename, aligned_landuse_uri],
        valid_distance, conversion_priority_filename, gdal.GDT_Float32,
        conversion_nodata, conversion_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False, aoi_uri=parameters['watersheds_uri'])

    pixels_per_step_to_convert = calculate_pixels_per_step_for_full_conversion(
        aligned_landuse_uri, parameters['convert_from_lulc_codes'], parameters['number_of_steps'])

    #build iterator
    priority_pixels = disk_sort.sort_to_disk(conversion_priority_filename, 0)
    lulc_ds = gdal.Open(aligned_landuse_uri)
    lulc_band = lulc_ds.GetRasterBand(1)
    lulc_array = lulc_band.ReadAsArray()
    output_lulc_list = []
    for step_index in range(parameters['number_of_steps']+1):
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
                if converted_pixels >= pixels_per_step_to_convert:
                    break
        
        if converted_pixels == 0 and step_index != 0:
            print 'everything converted already, breaking loop'
            break
        
        print 'saving lulc %d' % step_index
        output_lulc_uri = os.path.join(parameters['land_use_directory'],
            '%s_%d.tif' % (base_name, step_index))
        raster_utils.new_raster_from_base_uri(
            aligned_distance_from_forest_edge_filename, output_lulc_uri, 'GTiff', lulc_nodata,
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
        parameters['temporary_file_directory'], 'conversion_priority_%s.tif' % base_name)
    conversion_nodata = direction_factor * 9999
    conversion_pixel_size = raster_utils.get_cell_size_from_uri(
        parameters['landuse_uri'])
    lulc_nodata = raster_utils.get_nodata_from_uri(parameters['landuse_uri'])
    distance_nodata = raster_utils.get_nodata_from_uri(parameters['distance_from_stream_filename'])

    aligned_distance_from_stream_filename = os.path.join(
        parameters['temporary_file_directory'], 'aligned_distance_from_stream_%s.tif' % base_name)
    aligned_landuse_uri = os.path.join(
        parameters['temporary_file_directory'], 'aligned_landuse_%s.tif' % base_name)

    raster_utils.align_dataset_list(
        [parameters['distance_from_stream_filename'], parameters['landuse_uri']], 
        [aligned_distance_from_stream_filename, aligned_landuse_uri], ['nearest']*2,
        conversion_pixel_size, 'intersection', 0,
        dataset_to_bound_index=None, aoi_uri=parameters['watersheds_uri'])

    def valid_distance(distance, lulc):
        conversion_array = numpy.empty(distance.shape, dtype=numpy.float32)
        conversion_array[:] = conversion_nodata
        for convert_code in parameters['convert_from_lulc_codes']:
            mask = (lulc == convert_code)
            #invert the distance for sorting
            conversion_array[mask] = -distance[mask] * direction_factor
        return numpy.where(distance != distance_nodata, conversion_array, conversion_nodata)
    
    print 'building the prioritization from stream raster'
    raster_utils.vectorize_datasets(
        [aligned_distance_from_stream_filename, aligned_landuse_uri],
        valid_distance, conversion_priority_filename, gdal.GDT_Float32,
        conversion_nodata, conversion_pixel_size, 'intersection',
        dataset_to_align_index=0, vectorize_op=False, aoi_uri=parameters['watersheds_uri'])

    pixels_per_step_to_convert = calculate_pixels_per_step_for_full_conversion(
        aligned_landuse_uri, parameters['convert_from_lulc_codes'], parameters['number_of_steps'])

    #build iterator
    priority_pixels = disk_sort.sort_to_disk(conversion_priority_filename, 0)
    lulc_ds = gdal.Open(aligned_landuse_uri)
    lulc_band = lulc_ds.GetRasterBand(1)
    lulc_array = lulc_band.ReadAsArray()
    output_lulc_list = []
    for step_index in range(parameters['number_of_steps']+1):
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
                if converted_pixels >= pixels_per_step_to_convert:
                    break
        
        if converted_pixels == 0 and step_index != 0:
            print 'everything converted already, breaking loop'
            break
        
        print 'saving lulc %d' % step_index
        output_lulc_uri = os.path.join(
            parameters['land_use_directory'],
            '%s_%d.tif' % (base_name, step_index))
        raster_utils.new_raster_from_base_uri(
            aligned_landuse_uri, output_lulc_uri, 'GTiff', lulc_nodata,
            gdal.GDT_Int32)
        output_lulc_ds = gdal.Open(output_lulc_uri, gdal.GA_Update)
        output_lulc_band = output_lulc_ds.GetRasterBand(1)
        output_lulc_band.WriteArray(lulc_array)
        output_lulc_list.append(output_lulc_uri)
    return output_lulc_list

    
def run_sdr_analysis(parameters, land_cover_uri_list, summary_table_uri):
    sed_export_table_uri = os.path.join(
        parameters['output_file_directory'], summary_table_uri)
    sed_export_table = open(sed_export_table_uri, 'w')
    sed_export_table.write('step,%s\n' % os.path.splitext(summary_table_uri)[0])

    parameters['_prepare'] = invest_natcap.sdr.sdr._prepare(**parameters)

    for index, lulc_uri in enumerate(land_cover_uri_list):
        sdr_args = {
            'workspace_dir': os.path.join(parameters['workspace_dir'], str(index)),
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
            '_prepare': parameters['_prepare'],
        }
        invest_natcap.sdr.sdr.execute(sdr_args)

        sdr_export_uri = os.path.join(sdr_args['workspace_dir'], 'output', "sed_export_%d.tif" % index)
        sed_export_ds = gdal.Open(sdr_export_uri)
        sed_export_band = sed_export_ds.GetRasterBand(1)
        nodata = raster_utils.get_nodata_from_uri(sdr_export_uri)
        sed_export_total = 0.0
        print 'summing the sediment export'

        n_rows = sed_export_band.YSize
        n_cols = sed_export_band.XSize
        block_col_size, block_row_size = sed_export_band.GetBlockSize()
        for global_block_row in xrange(int(numpy.ceil(float(n_rows) / block_row_size))):
            for global_block_col in xrange(int(numpy.ceil(float(n_cols) / block_col_size))):
                global_col = global_block_col*block_col_size
                global_row = global_block_row*block_row_size
                global_col_size = min((global_block_col+1)*block_col_size, n_cols) - global_col
                global_row_size = min((global_block_row+1)*block_row_size, n_rows) - global_row

        #for row_index in xrange(sed_export_ds.RasterYSize):
                sed_array = sed_export_band.ReadAsArray(
                    global_col, global_row, global_col_size, global_row_size)
                sed_export_total += numpy.sum(sed_array[(sed_array != nodata) & (~numpy.isnan(sed_array))])
        sed_export_table.write('%d,%f\n' % (index, sed_export_total))
        sed_export_table.flush()

        sed_export_band = None
        gdal.Dataset.__swig_destroy__(sed_export_ds)
        sed_export_ds = None
        #no need to keep output and intermediate directories
        for directory in [os.path.join(sdr_args['workspace_dir'], 'output'), os.path.join(sdr_args['workspace_dir'], 'intermediate')]:
            try:
                shutil.rmtree(directory)
            except OSError as e:
                print "can't remove directory " + str(e)


def run_ndr_analysis(parameters, land_cover_uri_list, summary_table_uri):
    ndr_export_table_uri = os.path.join(
        parameters['output_file_directory'], summary_table_uri)
    ndr_export_table = open(ndr_export_table_uri, 'w')
    ndr_export_table.write('step,%s\n' % os.path.splitext(summary_table_uri)[0])

#    parameters['_prepare'] = invest_natcap.ndr.ndr._prepare(**parameters)

    for index, lulc_uri in enumerate(land_cover_uri_list):
        ndr_args = {
            'workspace_dir': os.path.join(parameters['workspace_dir'], str(index)),
            'suffix': str(index),
            'dem_uri': parameters['dem_uri'],
            'lulc_uri': lulc_uri,
            'watersheds_uri': parameters['watersheds_uri'],
            'biophysical_table_uri': parameters['biophysical_table_uri'],
            'threshold_flow_accumulation': parameters['threshold_flow_accumulation'],
            'k_param': 2,
#            '_prepare': parameters['_prepare'],
            'calc_n': parameters['calc_n'],
            'calc_p': parameters['calc_p'],
            'depth_to_root_rest_layer_uri': parameters['depth_to_root_rest_layer_uri'],
            'eto_uri': parameters['eto_uri'],
            'k_param': parameters['k_param'],
            'pawc_uri': parameters['pawc_uri'],
            'precipitation_uri': parameters['precipitation_uri'],
            'seasonality_constant': parameters['seasonality_constant']
        }
        invest_natcap.ndr.ndr.execute(ndr_args)

        ndr_export_uri = os.path.join(ndr_args['workspace_dir'], 'output', "nut_export_%d.tif" % index)
        nut_export_ds = gdal.Open(ndr_export_uri)
        nut_export_band = nut_export_ds.GetRasterBand(1)
        nodata = raster_utils.get_nodata_from_uri(ndr_export_uri)
        nut_export_total = 0.0
        print 'summing the nutiment export'

        n_rows = nut_export_band.YSize
        n_cols = nut_export_band.XSize
        block_col_size, block_row_size = nut_export_band.GetBlockSize()
        for global_block_row in xrange(int(numpy.ceil(float(n_rows) / block_row_size))):
            for global_block_col in xrange(int(numpy.ceil(float(n_cols) / block_col_size))):
                global_col = global_block_col*block_col_size
                global_row = global_block_row*block_row_size
                global_col_size = min((global_block_col+1)*block_col_size, n_cols) - global_col
                global_row_size = min((global_block_row+1)*block_row_size, n_rows) - global_row

        #for row_index in xrange(nut_export_ds.RasterYSize):
                nut_array = nut_export_band.ReadAsArray(
                    global_col, global_row, global_col_size, global_row_size)
                nut_export_total += numpy.sum(nut_array[(nut_array != nodata) & (~numpy.isnan(nut_array))])
        nut_export_table.write('%d,%f\n' % (index, nut_export_total))
        nut_export_table.flush()

        nut_export_band = None
        gdal.Dataset.__swig_destroy__(nut_export_ds)
        nut_export_ds = None
        #no need to keep output and intermediate directories
        for directory in [os.path.join(sdr_args['workspace_dir'], 'output'), os.path.join(sdr_args['workspace_dir'], 'intermediate')]:
            try:
                shutil.rmtree(directory)
            except OSError as e:
                print "can't remove directory " + str(e)

def worker(input, output):
    lowpriority()
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)
        input.task_done()
    input.task_done()


if __name__ == '__main__':
    DROPBOX_FOLDER = u'e:/dropboxcopy'
    OUTPUT_FOLDER = u'e:/distance_to_stream_outputs_nutrient'
    TEMPORARY_FOLDER = os.path.join(OUTPUT_FOLDER, 'temp')
    LAND_USE_FOLDER = os.path.join(OUTPUT_FOLDER, 'land_use_directory')
    NUMBER_OF_PROCESSES = 4

    for tmp_variable in ['TMP', 'TEMP', 'TMPDIR']:

        if tmp_variable in os.environ:
            print 'Updating os.environ["%s"]=%s to %s' % (tmp_variable, os.environ[tmp_variable], TEMPORARY_FOLDER)
        else:
            print 'Setting os.environ["%s"]=%s' % (tmp_variable, TEMPORARY_FOLDER)

        os.environ[tmp_variable] = TEMPORARY_FOLDER
    
    PARAMETERS = {
        'temporary_file_directory': TEMPORARY_FOLDER,
        'output_file_directory': OUTPUT_FOLDER,
        'land_use_directory': LAND_USE_FOLDER,
        'number_of_steps': 20,
        'calc_n': True,
        'calc_p': False,
        'threshold_flow_accumulation': 1000,
    }
    
    willamette_local_args = {
        'depth_to_root_rest_layer_uri': "E:/repositories/Base_Data/Freshwater/depth_to_root_rest_layer",
        'eto_uri': "E:/repositories/Base_Data/Freshwater/eto",
        'pawc_uri': "E:/repositories/Base_Data/Freshwater/pawc",
        'precipitation_uri': "E:/repositories/Base_Data/Freshwater/precip",
        'seasonality_constant': 5,
        u'convert_from_lulc_codes': range(51,66) + range(62, 65),  #read from biophysical table
        u'convert_to_lulc_code':87, #some other kind of crop 71, #this is 'field crop'
        u'biophysical_table_uri': "E:/repositories/Base_Data/Freshwater/biophysical_table.csv",
        u'dem_uri': "E:/repositories/Base_Data/Freshwater/dem/w001001.adf",
        u'erodibility_uri': "E:/repositories/Base_Data/Freshwater/erodibility/w001001.adf",
        u'erosivity_uri': "E:/repositories/Base_Data/Freshwater/erosivity/w001001.adf",
        u'ic_0_param': u'0.5',
        u'k_param': u'2',
        u'landuse_uri': "E:/repositories/Base_Data/Terrestrial/lulc_samp_cur/w001001.adf",
        u'sdr_max': u'0.8',
        u'accum_threshold': u'1000',
        u'watersheds_uri': "E:/repositories/Base_Data/Freshwater/watersheds.shp",
        u'workspace_dir': os.path.join(OUTPUT_FOLDER, u'willamette_local/'),
        u'suffix': 'willamette_local',
    }
    willamette_local_args.update(PARAMETERS)

    willamette_global_args = {
         'depth_to_root_rest_layer_uri': "E:/repositories/Base_Data/Freshwater/depth_to_root_rest_layer",
        'eto_uri': "E:/repositories/Base_Data/Freshwater/eto",
        'pawc_uri': "E:/repositories/Base_Data/Freshwater/pawc",
        'precipitation_uri': "E:/repositories/Base_Data/Freshwater/precip",
        u'convert_from_lulc_codes': range(1, 5), #read from biophysical table
        u'convert_to_lulc_code':12, #this is 'field crop'
        u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/biophysical_coeffs_Brazil_Unilever.csv"),
        u'dem_uri': "E:/repositories/Base_Data/Freshwater/dem",
        u'erodibility_uri': "E:/repositories/Willamette_global_Unilever/erodibility_HWSD_Will.tif",
        u'erosivity_uri': "E:/repositories/Willamette_global_Unilever/erosivity_Will_UTM.tif",
        u'ic_0_param': u'0.5',
        u'k_param': u'2',
        u'landuse_uri': "E:/repositories/Willamette_global_Unilever/MCD12Q1_type1_2012_Willamette_UTM.tif",
        u'sdr_max': u'0.8',
        u'threshold_flow_accumulation': 1000,
        u'watersheds_uri': "E:/repositories/Base_Data/Freshwater/watersheds.shp",
        u'workspace_dir': os.path.join(OUTPUT_FOLDER, u'willamette_global/'),
        u'suffix': 'willamette_global',
    }
    willamette_global_args.update(PARAMETERS)

    mg_args = {
        'depth_to_root_rest_layer_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/soil_depth_MT.tif"), 
        'eto_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/PET_MT.tif"),
        'pawc_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/PAWC_MT.tif"), 
        'precipitation_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/precip_MT.tif"),
        'seasonality_constant': 5,
        u'convert_from_lulc_codes': range(1, 5), #read from biophysical table
        u'convert_to_lulc_code':12, #this is 'field crop'
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
        'depth_to_root_rest_layer_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/soil_depth_STATSGO_Iowa_HUC8.tif"), 
        'eto_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/eto_Iowa_HUC8_Hargreaves_PRISM.tif"),
        'pawc_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/AWC_STATSGO_Iowa_HUC8.tif"), 
        'precipitation_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/precip_PRISM_Iowa_HUC8.tif"),
        'seasonality_constant': 5,
        u'convert_from_lulc_codes': [41, 42, 43, 90], #these are the forest types
        u'convert_to_lulc_code':82, #this is 'cultivated crops'
        u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/biophysical_coeffs_Iowa_Unilever_national_rich_fake_data.csv"),
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
        'depth_to_root_rest_layer_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/soil_depth_HWSD_Iowa_HUC8.tif"), 
        'eto_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/PET_CRU_2009_Iowa_HUC8.tif"),
        'pawc_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/AWC_HWSD_Iowa_HUC8.tif"), 
        'precipitation_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/precip_CRU_2009_Iowa_HUC8.tif"),
        'seasonality_constant': 5,
        u'convert_from_lulc_codes': range(1, 5), #forest lulcs from biophysical table
        u'convert_to_lulc_code':12, #this is 'field crop'
        u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_global/biophysical_coeffs_Iowa_Unilever_global_rich_fake_data.csv"),
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
    
    if os.path.exists(OUTPUT_FOLDER):
        backup_folder = os.path.join(os.path.split(OUTPUT_FOLDER)[0], 'distance_to_stream_backup')
        if os.path.exists(backup_folder):
            shutil.rmtree(backup_folder)
        os.rename(OUTPUT_FOLDER, backup_folder)

    #worker_pool = multiprocessing.Pool()
    for args, simulation in [
        (willamette_local_args, 'willamette_local_'),
        #(willamette_global_args, 'willamette_global_'),
        #(mg_args, 'mg_global'),
        #(iowa_global_args, 'iowa_global_'),
        #(iowa_national_args, 'iowa_national_'),
        ]:
    
        initialize_simulation(args)

        simulation_list = [
            #("ag", "ag", 0, ['ndr', 'sdr']),
            ("core", "core", 0, ['ndr']),
            ("edge", "edge", 0, ['ndr']),
            ("to_stream", "to_stream", 0, ['ndr']),
            ("fragmentation", "fragmentation", 0, ['ndr']),
            #("from_stream", "from_stream", 0, ['ndr', 'sdr']),
            #("from_stream", "from_stream_with_buffer_1", 1, ['ndr', 'sdr']),
            #("from_stream", "from_stream_with_buffer_2", 2, ['ndr', 'sdr']),
            #("from_stream", "from_stream_with_buffer_3", 3, ['ndr', 'sdr']),
            #("from_stream", "from_stream_with_buffer_9", 9),
            ]

        landcover_uri_dictionary = {}

        input_queue = multiprocessing.JoinableQueue()
        output_queue = multiprocessing.Queue()

        result_dictionary = {}
        for MODE, FILENAME, BUFFER, _ in simulation_list:
            input_queue.put((step_land_change, [args, simulation+FILENAME, MODE, BUFFER]))
            #result_dictionary[FILENAME] = worker_pool.apply_async(step_land_change, [args, simulation+FILENAME, MODE, BUFFER])
 
        for _ in xrange(NUMBER_OF_PROCESSES):
            multiprocessing.Process(target=worker, args=(input_queue, output_queue)).start()

        result_list = []
        for MODE, FILENAME, BUFFER, SIMULATION_TYPES in simulation_list:
            landcover_uri_dictionary[FILENAME] = output_queue.get() #result_dictionary[FILENAME].get(0xFFFF)
            args_copy = args.copy()
            args_copy['workspace_dir'] = os.path.join(args['workspace_dir'], FILENAME)
            if 'sdr' in SIMULATION_TYPES:
                input_queue.put((run_sdr_analysis, [args_copy, landcover_uri_dictionary[FILENAME], simulation+FILENAME + ".csv"]))
            if 'ndr' in SIMULATION_TYPES:
                input_queue.put((run_ndr_analysis, [args_copy, landcover_uri_dictionary[FILENAME], simulation+FILENAME + ".csv"]))
            #result_list.append(worker_pool.apply_async(
            #    run_sdr_analysis, [args_copy, landcover_uri_dictionary[FILENAME], simulation+FILENAME + ".csv"]))

        for _ in xrange(NUMBER_OF_PROCESSES):
            input_queue.put('STOP')

        input_queue.join()

        #for result in result_list:
        #    result.get(0xFFFF)

        #aggregate all the .csv results into one big csv
        #get area of a pixel
        try:
            out_pixel_size = raster_utils.get_cell_size_from_uri(landcover_uri_dictionary.values()[0][0])
        except IndexError:
            out_pixel_size = 1
    
        simulation_result_dictionary = {
            filename:['']*(args['number_of_steps']+1) for _, filename, _ in simulation_list
        }

        print out_pixel_size
        #open all the csvs and dump them to a dictionary
        #loop through each step of the dictionary and output a row

        for MODE, FILENAME, BUFFER in simulation_list:
            summary_table_uri = simulation+FILENAME + ".csv"
            sed_export_table_uri = os.path.join(
                args['output_file_directory'], summary_table_uri)
            sed_export_table = open(sed_export_table_uri, 'r')
            sed_export_table.readline()
            step_index = 0
            for line in sed_export_table:
                sediment_export_value = (''.join(line.split(',')[1:])).rstrip()
                simulation_result_dictionary[FILENAME][step_index] = sediment_export_value
                step_index += 1
        #    sed_export_table.write('step,%s\n' % os.path.splitext(summary_table_uri)[0])
        
        print simulation_result_dictionary
        summary_table_uri = os.path.join(args['output_file_directory'], simulation + '_summary_table.csv')
        summary_table = open(summary_table_uri, 'w')
        summary_table.write('area converted (Ha),')
        summary_table.write(','.join([filename for (_, filename, _) in simulation_list]) + '\n')
        for step_number in xrange(args['number_of_steps'] + 1):
            summary_table.write('%f' % (step_number * out_pixel_size))
            for _, FILENAME, _ in simulation_list:
                summary_table.write(','+str(simulation_result_dictionary[FILENAME][step_number]))
            summary_table.write('\n')
        summary_table.close()
