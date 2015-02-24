import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
import pylab 
import sys
import time
from joblib import dump, load, Parallel, delayed
import utils
from utils import *

sys.path.append(utils.xgb_path)
import xgboost as xgb

t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx3.joblib_dat')
t0tv_mx3 = t0tv_mx_save['t0tv_mx']
click_values = t0tv_mx_save['click']
day_values = t0tv_mx_save['day']
print "t0tv_mx3 loaded with shape", t0tv_mx3.shape


n_trees = utils.xgb_n_trees
day_test = 30
if utils.tvh == 'Y':
    day_test = 31

param = {'max_depth':15, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
         'subsample':1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}

nn = t0tv_mx3.shape[0]
np.random.seed(999)
sample_idx = np.random.random_integers(0, 3, nn)

predv_xgb = 0
ctr = 0
for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(np.logical_and(day_values >= 22, day_values < day_test), np.logical_and(sample_idx== idx , True))
    filter_v1 = day_values == day_test

    xt1 = t0tv_mx3[filter1, :]
    yt1 = click_values[filter1]
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        print xt1.shape, yt1.shape
        raise ValueError('wrong shape!')
    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(t0tv_mx3[filter_v1], label=click_values[filter_v1])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print xt1.shape, yt1.shape

    plst = list(param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist)
    #xgb_pred[rseed] = xgb1.predict(dtv3)
    #xgb_list[rseed] = xgb1
    
    ctr += 1
    predv_xgb += xgb1.predict(dvalid)
    print '-'*30, ctr, logloss(predv_xgb / ctr, click_values[filter_v1])

print "to save validation predictions ..."
dump(predv_xgb / ctr, utils.tmp_data_path + 'xgb_pred_v.joblib_dat')

