""""
This is a saved model run from invest_natcap.sdr.sdr.
Generated: 06/16/14 11:08:22
InVEST version: dev81:3.0.1 [e65614c6b30b]
"""

import invest_natcap.sdr.sdr

args = {
        u'biophysical_table_uri': u'C:/Users/rich/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/biophysical_coeffs_Iowa_Unilever_national.csv',
        u'dem_uri': u"C:/Users/rich/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/DEM_SRTM_Iowa_HUC8.tif",
        u'erodibility_uri': u"C:/Users/rich/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/erodibility_STATSGO_Iowa_HUC8.tif",
        u'erosivity_uri': u"C:/Users/rich/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/erosivity_Iowa_HUC8.tif",
        u'ic_0_param': u'0.5',
        u'k_param': u'2',
        u'landuse_uri': u"C:/Users/rich/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/LULC_NLCD_2006_Iowa_HUC8.tif",
        u'sdr_max': u'0.8',
        u'threshold_flow_accumulation': u'1000',
        u'watersheds_uri': u"C:/Users/rich/Dropbox/Unilever_data_from_Stacie/Input_Iowa_national/HUC8_Iowa_intersect_dissolve.shp",
        u'workspace_dir': u'C:/Users/rich/Documents/Iowa_national',
        u'suffix': 'test',
}

print 'preparing sdr'
args['_prepare'] = invest_natcap.sdr.sdr._prepare(**args)

print 'calling sdr'
invest_natcap.sdr.sdr.execute(args)
