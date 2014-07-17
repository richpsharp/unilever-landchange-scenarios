""""
This is a saved model run from invest_natcap.sdr.sdr.
Generated: 06/16/14 11:08:22
InVEST version: dev81:3.0.1 [e65614c6b30b]
"""

import os

import invest_natcap.sdr.sdr

DROPBOX_FOLDER = 'C:/Users/rich/Dropbox/'
OUTPUT_FOLDER = u'C:/Users/rich/Documents/unilever_outputs'

iowa_args = {
    u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/biophysical_coeffs_Iowa_Unilever_national.csv"),
    u'dem_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/DEM_SRTM_Iowa_HUC8_v2.tif"),
    u'erodibility_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/erodibility_STATSGO_Iowa_HUC8.tif"),
    u'erosivity_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/erosivity_Iowa_HUC8.tif"),
    u'ic_0_param': u'0.5',
    u'k_param': u'2',
    u'landuse_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/LULC_NLCD_2006_Iowa_HUC8.tif"),
    u'sdr_max': u'0.8',
    u'threshold_flow_accumulation': u'1000',
    u'watersheds_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_Iowa_national/HUC8_Iowa_intersect_dissolve.shp"),
    u'workspace_dir': os.path.join(OUTPUT_FOLDER, u'Iowa_national'),
    u'suffix': 'iowa',
}

mg_args = {
    u'biophysical_table_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/biophysical_coeffs_Brazil_Unilever.csv"),
    u'dem_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/DEM_SRTM_MT_filled.tif"),
    u'erodibility_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/erodibility_MT.tif"),
    u'erosivity_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/erosivity_MT.tif"),
    u'ic_0_param': u'0.5',
    u'k_param': u'2',
    u'landuse_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/LULC_MCD12Q1_2012_MT.tif"),
    u'sdr_max': u'0.8',
    u'threshold_flow_accumulation': u'1000',
    u'watersheds_uri': os.path.join(DROPBOX_FOLDER, u"Unilever_data_from_Stacie/Input_MatoGrosso_global/Mato_Grosso.shp"),
    u'workspace_dir': os.path.join(OUTPUT_FOLDER, u'Mato_Grosso_global/'),
    u'suffix': 'mato_grosso',
}


for args in [iowa_args]:

    print 'preparing sdr'
    args['_prepare'] = invest_natcap.sdr.sdr._prepare(**args)

    print 'calling sdr'
    invest_natcap.sdr.sdr.execute(args)
