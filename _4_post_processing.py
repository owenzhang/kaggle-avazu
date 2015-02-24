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

t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx.joblib_dat')
click_values = t0tv_mx_save['click']
day_values = t0tv_mx_save['day']
site_id_values= t0tv_mx_save['site_id']
print "t0tv_mx loaded"


day_test = 30
if utils.tvh == 'Y':
    day_test = 31

#RandomForest model output
rf_pred = load(utils.tmp_data_path + 'rf_pred_v.joblib_dat')
print "RF prediction loaded with shape", rf_pred.shape

#GBDT (xgboost) model output
xgb_pred = load(utils.tmp_data_path + 'xgb_pred_v.joblib_dat')
print "xgb prediction loaded with shape", xgb_pred.shape

#Vowpal Wabbit model output
ctr = 0
vw_pred = 0
for i in [1, 2, 3, 4]:
    vw_pred += 1 / (1+ np.exp(-pd.read_csv(open(utils.tmp_data_path + 'vwV12__r%d_test.txt_pred.txt'%i, 'r'), header=None).ix[:,0].values))
    ctr += 1
vw_pred /= ctr
print "VW prediction loaded with shape", vw_pred.shape

#factorization machine model output
ctr = 0
fm_pred = 0
for i in [51, 52, 53, 54]:
    fm_pred += pd.read_csv(open(utils.tmp_data_path + 'fm__r%d_v.txt.out'%i, 'r'), header=None).ix[:,0].values
    ctr += 1
fm_pred /= ctr
print "FM prediction loaded with shape", fm_pred.shape


blending_w = {'rf': .075, 'xgb': .175, 'vw': .225, 'fm': .525}

total_w = 0
pred = 0

pred += rf_pred * blending_w['rf']
total_w += blending_w['rf']
pred += xgb_pred * blending_w['xgb']
total_w += blending_w['xgb']
pred += vw_pred * blending_w['vw']
total_w += blending_w['vw']
pred += fm_pred * blending_w['fm']
total_w += blending_w['fm']

pred /= total_w

if utils.tvh == 'Y':
    #create submission
    predh_raw_avg = pred
    site_ids_h = site_id_values[day_values == 31] 
    tmp_f1 = site_ids_h == '17d1b03f'
    predh_raw_avg[tmp_f1] *= .13 / predh_raw_avg[tmp_f1].mean()
    predh_raw_avg *= .161 / predh_raw_avg.mean()

    sub0 = pd.read_csv(open(utils.raw_data_path + 'sampleSubmission', 'r'))
    pred_h_str = ["%.4f" % x for x in predh_raw_avg]
    sub0['click'] = pred_h_str
    fn_sub = utils.tmp_data_path + 'sub_sample' + str(utils.sample_pct) + '.csv.gz'
    import gzip
    sub0.to_csv(gzip.open(fn_sub, 'w'), index=False)
    print "=" * 80
    print "Training complted and submission file " + fn_sub + " created."
    print "=" * 80
else:
    #validate using day30
    print "Training completed!"
    print "=" * 80
    print "logloss of blended prediction:", logloss(pred, click_values[day_values==day_test])
    print "=" * 80
