import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
sys.path.append('/home/zzhang/Downloads/xgboost/wrapper')
import xgboost as xgb
from joblib import dump, load, Parallel, delayed
import utils
from utils import *


raw_data_path = utils.raw_data_path
tmp_data_path = utils.tmp_data_path


t0org0 = pd.read_csv(open(raw_data_path + "train", "ra"))
h0org = pd.read_csv(open(raw_data_path + "test", "ra"))


if utils.sample_pct < 1.0:
    np.random.seed(999)
    r1 = np.random.uniform(0, 1, t0org0.shape[0])
    t0org0 = t0org0.ix[r1 < utils.sample_pct, :]
    print "testing with small sample of training data, ", t0org0.shape


h0org['click'] = 0
t0org = pd.concat([t0org0, h0org])
print "finished loading raw data, ", t0org.shape

print "to add some basic features ..."
t0org['day']=np.round(t0org.hour % 10000 / 100)
t0org['hour1'] = np.round(t0org.hour % 100)
t0org['day_hour'] = (t0org.day.values - 21) * 24 + t0org.hour1.values
t0org['day_hour_prev'] = t0org['day_hour'] - 1
t0org['day_hour_next'] = t0org['day_hour'] + 1
t0org['app_or_web'] = 0
t0org.ix[t0org.app_id.values=='ecad2386', 'app_or_web'] = 1

t0 = t0org

t0['app_site_id'] = np.add(t0.app_id.values, t0.site_id.values)

print "to encode categorical features using mean responses from earlier days -- univariate"
sys.stdout.flush()

calc_exptv(t0,  ['app_or_web'])

exptv_vn_list = ['app_site_id', 'as_domain', 'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                'app_site_model', 'site_model','app_model', 'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']

calc_exptv(t0, exptv_vn_list)

calc_exptv(t0, ['app_site_id'], add_count=True)


print "to encode categorical features using mean responses from earlier days -- multivariate"
vns = ['app_or_web',  'device_ip', 'app_site_id', 'device_model', 'app_site_model', 'C1', 'C14', 'C17', 'C21',
                        'device_type', 'device_conn_type','app_site_model_aw', 'dev_ip_app_site']
dftv = t0.ix[np.logical_and(t0.day.values >= 21, t0.day.values < 32), ['click', 'day', 'id'] + vns].copy()

dftv['app_site_model'] = np.add(dftv.device_model.values, dftv.app_site_id.values)
dftv['app_site_model_aw'] = np.add(dftv.app_site_model.values, dftv.app_or_web.astype('string').values)
dftv['dev_ip_app_site'] = np.add(dftv.device_ip.values, dftv.app_site_id.values)
for vn in vns:
    dftv[vn] = dftv[vn].astype('category')
    print vn

n_ks = {'app_or_web': 100, 'app_site_id': 100, 'device_ip': 10, 'C14': 50, 'app_site_model': 50, 'device_model': 100, 'device_id': 50,
        'C17': 100, 'C21': 100, 'C1': 100, 'device_type': 100, 'device_conn_type': 100, 'banner_pos': 100,
        'app_site_model_aw': 100, 'dev_ip_app_site': 10 , 'device_model': 500}

exp2_dict = {}
for vn in vns:
    exp2_dict[vn] = np.zeros(dftv.shape[0])

days_npa = dftv.day.values
    
for day_v in xrange(22, 32):
    df1 = dftv.ix[np.logical_and(dftv.day.values < day_v, dftv.day.values < 31), :].copy()
    df2 = dftv.ix[dftv.day.values == day_v, :]
    print "Validation day:", day_v, ", train data shape:", df1.shape, ", validation data shape:", df2.shape
    pred_prev = df1.click.values.mean() * np.ones(df1.shape[0])
    for vn in vns:
        if 'exp2_'+vn in df1.columns:
            df1.drop('exp2_'+vn, inplace=True, axis=1)
    for i in xrange(3):
        for vn in vns:
            p1 = calcLeaveOneOut2(df1, vn, 'click', n_ks[vn], 0, 0.25, mean0=pred_prev)
            pred = pred_prev * p1
            print day_v, i, vn, "change = ", ((pred - pred_prev)**2).mean()
            pred_prev = pred    
            
        pred1 = df1.click.values.mean()
        for vn in vns:
            print "="*20, "merge", day_v, vn
            diff1 = mergeLeaveOneOut2(df1, df2, vn)
            pred1 *= diff1
            exp2_dict[vn][days_npa == day_v] = diff1
        
        pred1 *= df1.click.values.mean() / pred1.mean()
        print "logloss = ", logloss(pred1, df2.click.values)
        #print my_lift(pred1, None, df2.click.values, None, 20, fig_size=(10, 5))
        #plt.show()

for vn in vns:
    t0['exp2_'+vn] = exp2_dict[vn]


print "to count prev/current/next hour by ip ..."
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour_prev', fill_na=0)
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour', fill_na=0)
cntDualKey(t0, 'device_ip', None, 'day_hour', 'day_hour_next', fill_na=0)

print "to create day diffs"
t0['pday'] = t0.day - 1
calcDualKey(t0, 'device_ip', None, 'day', 'pday', 'click', 10, None, True, True)
t0['cnt_diff_device_ip_day_pday'] = t0.cnt_device_ip_day.values  - t0.cnt_device_ip_pday.values
t0['hour1_web'] = t0.hour1.values
t0.ix[t0.app_or_web.values==0, 'hour1_web'] = -1
t0['app_cnt_by_dev_ip'] = my_grp_cnt(t0.device_ip.values.astype('string'), t0.app_id.values.astype('string'))


t0['hour1'] = np.round(t0.hour.values % 100)
t0['cnt_diff_device_ip_day_pday'] = t0.cnt_device_ip_day.values  - t0.cnt_device_ip_pday.values

t0['rank_dev_ip'] = my_grp_idx(t0.device_ip.values.astype('string'), t0.id.values.astype('string'))
t0['rank_day_dev_ip'] = my_grp_idx(np.add(t0.device_ip.values, t0.day.astype('string').values).astype('string'), t0.id.values.astype('string'))
t0['rank_app_dev_ip'] = my_grp_idx(np.add(t0.device_ip.values, t0.app_id.values).astype('string'), t0.id.values.astype('string'))


t0['cnt_dev_ip'] = get_agg(t0.device_ip.values, t0.id, np.size)
t0['cnt_dev_id'] = get_agg(t0.device_id.values, t0.id, np.size)

t0['dev_id_cnt2'] = np.minimum(t0.cnt_dev_id.astype('int32').values, 300)
t0['dev_ip_cnt2'] = np.minimum(t0.cnt_dev_ip.astype('int32').values, 300)

t0['dev_id2plus'] = t0.device_id.values
t0.ix[t0.cnt_dev_id.values == 1, 'dev_id2plus'] = '___only1'
t0['dev_ip2plus'] = t0.device_ip.values
t0.ix[t0.cnt_dev_ip.values == 1, 'dev_ip2plus'] = '___only1'

t0['diff_cnt_dev_ip_hour_phour_aw2_prev'] = (t0.cnt_device_ip_day_hour.values - t0.cnt_device_ip_day_hour_prev.values) * ((t0.app_or_web * 2 - 1)) 
t0['diff_cnt_dev_ip_hour_phour_aw2_next'] = (t0.cnt_device_ip_day_hour.values - t0.cnt_device_ip_day_hour_next.values) * ((t0.app_or_web * 2 - 1)) 


print "to save t0 ..."

dump(t0, tmp_data_path + 't0.joblib_dat')


print "to generate t0tv_mx .. "
app_or_web = None
_start_day = 22
list_param = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'banner_pos', 'device_type', 'device_conn_type']
feature_list_dict = {}

feature_list_name = 'tvexp3'
feature_list_dict[feature_list_name] = list_param + \
                            ['exptv_' + vn for vn in ['app_site_id', 'as_domain', 
                             'C14','C17', 'C21', 'device_model', 'device_ip', 'device_id', 'dev_ip_aw', 
                             'dev_id_ip', 'C14_aw', 'C17_aw', 'C21_aw']] + \
                            ['cnt_diff_device_ip_day_pday', 
                             'app_cnt_by_dev_ip', 'cnt_device_ip_day_hour', 'app_or_web',
                             'rank_dev_ip', 'rank_day_dev_ip', 'rank_app_dev_ip',
                             'diff_cnt_dev_ip_hour_phour_aw2_prev', 'diff_cnt_dev_ip_hour_phour_aw2_next',
                             'exp2_device_ip', 'exp2_app_site_id', 'exp2_device_model', 'exp2_app_site_model',
                             'exp2_app_site_model_aw', 'exp2_dev_ip_app_site',
                             'cnt_dev_ip', 'cnt_dev_id', 'hour1_web']

filter_tv = np.logical_and(t0.day.values >= _start_day, t0.day.values < 31)
filter_t1 = np.logical_and(t0.day.values < 30, filter_tv)
filter_v1 = np.logical_and(~filter_t1, filter_tv)    
    
print filter_tv.sum()


for vn in feature_list_dict[feature_list_name] :
    if vn not in t0.columns:
        print "="*60 + vn
        
yv = t0.click.values[filter_v1]

t0tv_mx = t0.as_matrix(feature_list_dict[feature_list_name])

print t0tv_mx.shape


print "to save t0tv_mx ..."

t0tv_mx_save = {}
t0tv_mx_save['t0tv_mx'] = t0tv_mx
t0tv_mx_save['click'] = t0.click.values
t0tv_mx_save['day'] = t0.day.values
t0tv_mx_save['site_id'] = t0.site_id.values
dump(t0tv_mx_save, tmp_data_path + 't0tv_mx.joblib_dat')



