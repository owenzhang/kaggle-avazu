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

t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx3.joblib_dat')
t0tv_mx3 = t0tv_mx_save['t0tv_mx']
click_values = t0tv_mx_save['click']
day_values = t0tv_mx_save['day']
print "t0tv_mx3 loaded with shape", t0tv_mx3.shape


from sklearn.ensemble import RandomForestClassifier

day_test = 30
if utils.tvh == 'Y':
    day_test = 31

print "to create Random Forest using day", day_test, " as validation"

clf = RandomForestClassifier(n_estimators=32, max_depth=40, min_samples_split=100, min_samples_leaf=10, random_state=0, criterion='entropy',
                             max_features=8, verbose = 1, n_jobs=-1, bootstrap=False)

_start_day = 22


predv = 0
ctr = 0
xv = t0tv_mx3[day_values==day_test, :]
yv = click_values[day_values==day_test]
nn = t0tv_mx3.shape[0]



for i1 in xrange(8):
    clf.random_state = i1
    np.random.seed(i1)
    r1 = np.random.uniform(0, 1, nn)
    filter1 = np.logical_and(np.logical_and(day_values >= _start_day, day_values < day_test), np.logical_and(r1 < .3, True))
    xt1 = t0tv_mx3[filter1, :]
    yt1 = click_values[filter1]
    rf1 = clf.fit(xt1, yt1)
    y_hat = rf1.predict_proba(xv)[:, 1]
    predv += y_hat
    ctr += 1
    ll = logloss(predv/ctr, yv)
    print "iter", i1, ", logloss = ", ll
    sys.stdout.flush()

list_param = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
feature_list = list_param + \
                            ['exptv_' + vn for vn in ['app_site_id', 'as_domain', 
                             'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                             'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']] + \
                            ['cnt_diff_device_ip_day_pday', 
                             'app_cnt_by_dev_ip', 'cnt_device_ip_day_hour', 'app_or_web',
                             'rank_dev_ip', 'rank_day_dev_ip', 'rank_app_dev_ip',
                             'diff_cnt_dev_ip_hour_phour_aw2_prev', 'diff_cnt_dev_ip_hour_phour_aw2_next',
                             'exp2_device_ip', 'exp2_app_site_id', 'exp2_device_model', 'exp2_app_site_model',
                             'exp2_app_site_model_aw', 'exp2_dev_ip_app_site',
                             'cnt_dev_ip', 'cnt_dev_id', 'hour1_web'] + \
                            ['all_withid', 'all_noid', 'all_but_ip', 'fm_5vars']

rf1_imp = pd.DataFrame({'feature':feature_list, 'impt': clf.feature_importances_})
print rf1_imp.sort('impt')

print "to save validation predictions ..."
dump(predv / ctr, utils.tmp_data_path + 'rf_pred_v.joblib_dat')

