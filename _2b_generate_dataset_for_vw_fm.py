import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
import utils
from utils import *
import os

from joblib import dump, load

t0 = load(utils.tmp_data_path + 't0.joblib_dat')
print "t0 loaded with shape", t0.shape

t0['dev_id_cnt2'] = np.minimum(t0.cnt_dev_id.astype('int32').values, 300)
t0['dev_ip_cnt2'] = np.minimum(t0.cnt_dev_ip.astype('int32').values, 300)

t0['dev_id2plus'] = t0.device_id.values
t0.ix[t0.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
t0['dev_ip2plus'] = t0.device_ip.values
t0.ix[t0.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'

t0['device_ip_only_hour_for_day'] = t0.cnt_device_ip_day_hour.values == t0.cnt_device_ip_pday.values

vns0 = ['app_or_web', 'banner_pos', 'C1', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
for vn in vns0 + ['C14']:
    print vn
    vn2 = '_A_' + vn
    t0[vn2] = np.add(t0['app_site_id'].values, t0[vn].astype('string').values)
    t0[vn2] = t0[vn2].astype('category')

t3 = t0
vns1 = vns0 + ['hour1'] + ['_A_' + vn for vn in vns0] + \
 ['device_model', 'device_type', 'device_conn_type', 'app_site_id', 'as_domain', 'as_category',
      'cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
     'cnt_diff_device_ip_day_pday', 'as_model'] + \
    [ 'dev_id_cnt2', 'dev_ip_cnt2', 'C14', '_A_C14', 'dev_ip2plus', 'dev_id2plus']
 
#'cnt_device_ip_day', 'device_ip_only_hour_for_day'
    
t3a = t3.ix[:, ['click']].copy()
idx_base = 3000
for vn in vns1:
    if vn in ['cnt_device_ip_day_hour', 'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next', 'cnt_device_ip_pday',
     'cnt_diff_device_ip_day_pday', 'cnt_device_ip_day', 'cnt_device_ip_pday']:
        _cat = pd.Series(np.maximum(-100, np.minimum(200, t3[vn].values))).astype('category').values.codes
    elif vn in ['as_domain']:
        _cat = pd.Series(np.add(t3['app_domain'].values, t3['site_domain'].values)).astype('category').values.codes
    elif vn in ['as_category']:
        _cat = pd.Series(np.add(t3['app_category'].values, t3['site_category'].values)).astype('category').values.codes
    elif vn in ['as_model']:
        _cat = pd.Series(np.add(t3['app_site_id'].values, t3['device_model'].values)).astype('category').values.codes
    else:
        _cat = t3[vn].astype('category').values.codes
    _cat = np.asarray(_cat, dtype='int32')
    _cat1 = _cat + idx_base
    t3a[vn] = _cat1
    print vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size
    idx_base += _cat.max() + 1

print "to save t3a ..."
t3a_save = {}
t3a_save['t3a'] = t3a
t3a_save['idx_base'] = idx_base
dump(t3a_save, utils.tmp_data_path + 't3a.joblib_dat')
