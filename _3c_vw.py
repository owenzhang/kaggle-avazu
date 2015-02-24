import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
import pylab 
import sys
import time
import os
import utils
from utils import *
from joblib import dump, load, Parallel, delayed

sys.path.append(utils.xgb_path)
import xgboost as xgb

rseed = 0
xgb_eta = .3
tvh = utils.tvh
n_passes = 4

i = 1
while i < len(sys.argv):
    if sys.argv[i] == '-rseed':
        i += 1
        rseed = int(sys.argv[i])
    else:
        raise ValueError("unrecognized parameter [" + sys.argv[i] + "]")
    
    i += 1


file_name1 = '_r' + str(rseed)

path1 = utils.tmp_data_path
fn_t = path1 + 'vwV12_' + file_name1 + '_train.txt'
fn_v = path1 + 'vwV12_' + file_name1 + '_test.txt'


def build_data():
    t0tv_mx_save = load(utils.tmp_data_path + 't0tv_mx.joblib_dat')

    t0tv_mx = t0tv_mx_save['t0tv_mx']
    click_values = t0tv_mx_save['click']
    day_values = t0tv_mx_save['day']

    print "t0tv_mx loaded with shape ", t0tv_mx.shape

    test_day = 30
    if tvh == 'Y':
        test_day = 31

    np.random.seed(rseed)
    nn = t0tv_mx.shape[0]
    r1 = np.random.uniform(0, 1, nn)
    filter1 = np.logical_and(np.logical_and(day_values >= 22, day_values < test_day), np.logical_and(r1 < .25, True))
    filter_v1 = day_values == test_day

    xt1 = t0tv_mx[filter1, :]
    yt1 = click_values[filter1]
    if xt1.shape[0] <=0 or xt1.shape[0] != yt1.shape[0]:
        print xt1.shape, yt1.shape
        raise ValueError('wrong shape!')
    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(t0tv_mx[filter_v1], label=click_values[filter_v1])
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print xt1.shape, yt1.shape


    n_trees = 30
    n_parallel_tree = 1

    param = {'max_depth':6, 'eta':xgb_eta, 'objective':'binary:logistic', 'verbose':1,
	     'subsample':1.0, 'min_child_weight':50, 'gamma':0,
	     'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': rseed,
	     'num_parallel_tree': n_parallel_tree}

    plst = list(param.items()) + [('eval_metric', 'logloss')]
    xgb_test_basis_d6 = xgb.train(plst, dtrain, n_trees, watchlist)

    print "to score gbdt ..."

    dtv = xgb.DMatrix(t0tv_mx)
    xgb_leaves = xgb_test_basis_d6.predict(dtv, pred_leaf = True)
        
    t0 = pd.DataFrame({'click': click_values})
    print xgb_leaves.shape
    for i in xrange(n_trees * n_parallel_tree):
        pred2 = xgb_leaves[:, i]
        #print pred2[:10]
        #print pred_raw_diff[:10]
        print i, np.unique(pred2).size
        t0['xgb_basis'+str(i)] = pred2


    t3a_save = load(utils.tmp_data_path + 't3a.joblib_dat')

    t3a = t3a_save['t3a']
    idx_base = 0
    for vn in ['xgb_basis' + str(i) for i in xrange(30 * n_parallel_tree)]:
        _cat = np.asarray(t0[vn].astype('category').values.codes, dtype='int32')
        _cat1 = _cat + idx_base
        print vn, idx_base, _cat1.min(), _cat1.max(), np.unique(_cat).size
        t3a[vn] = _cat1
        idx_base += _cat.max() + 1

    t3a['click1'] = t3a.click.values * 2 - 1
    t3a['ns_C']='|C'
    t3a['ns_D']='|D'
    t3a['ns_M']='|M'
    t3a['ns_S']='|S'
    t3a['ns_W']='|W'
    t3a['ns_N']='|N'
    t3a['ns_X']='|X'
    t3a['ns_Y']='|Y'
    t3a['ns_Z']='|Z'

    field_list = ['click1']
    field_list += ['ns_C', 'banner_pos', 'C1'] + ['C' + str(x) for x in xrange(14, 22)]
    field_list += ['ns_D', 'dev_ip2plus', 'dev_id2plus']
    field_list += ['ns_M', 'device_model', 'device_type', 'device_cnn_type']
    field_list += ['ns_S', 'app_site_id', 'as_domain', 'as_category']
    field_list += ['ns_W', 'app_or_web']
    field_list += ['ns_N', 'cnt_device_ip_day_hour', 'cnt_device_ip_pday', 
	           'cnt_diff_device_ip_day_pday', 'dev_id_cnt2', 'dev_ip_cnt2',
	           'cnt_device_ip_day_hour_prev', 'cnt_device_ip_day_hour_next']
    field_list += ['ns_X'] + ['xgb_basis'+str(i) for i in xrange(0, 10)]
    field_list += ['ns_Y'] + ['xgb_basis'+str(i) for i in xrange(10, 20)]
    field_list += ['ns_Z'] + ['xgb_basis'+str(i) for i in xrange(20, 30)]


    if tvh == 'Y':
        row_idx = np.logical_and(day_values >= 22, day_values <= 30)
        print row_idx.shape, row_idx.sum()
    else:
        row_idx = np.zeros(t3a.shape[0])

        pre_t_lmt = (day_values < 22).sum()
        t_lmt = (day_values < 30).sum()
        v_lmt = (day_values < 31).sum()

        t_cnt = t_lmt - pre_t_lmt
        v_cnt = v_lmt - t_lmt

        t_idx = np.random.permutation(t_cnt) + pre_t_lmt
        v_idx = np.random.permutation(v_cnt) + t_lmt


        i = 0
        i_t = 0
        i_v = 0
        while True:
	    if i % 7 == 6:
	        row_idx[i] = v_idx[i_v]
	        i_v += 1
	        if i_v >= v_cnt:
	            i_v = 0
	    else:
	        #training
	        row_idx[i] = t_idx[i_t]
	        i_t += 1
	        if i_t >= t_cnt:
	            break
	    i+= 1

        row_idx = row_idx[:i]
        print t3a.shape, t_cnt, v_cnt, row_idx.shape

        t3a['idx'] = np.arange(t3a.shape[0])
        t3a.set_index('idx', inplace=True)

    print "to write training file, this may take a long time"
    import gzip
    t3a.ix[row_idx, field_list].to_csv(open(fn_t, 'w'), sep=' ', header=False, index=False)

    os.system("gzip -f "+fn_t)

    print "to write test file, this shouldn't take too long"
    if tvh == 'Y':
        t3a.ix[day_values==31, field_list].to_csv(open(fn_v, 'w'), sep=' ', header=False, index=False)
    else:
        t3a.ix[day_values==30, field_list].to_csv(open(fn_v, 'w'), sep=' ', header=False, index=False)

    os.system("gzip -f "+fn_v)


build_data()

if tvh == 'Y':
    holdout_str = " --holdout_off "
else:
    holdout_str = " --holdout_period 7 "
    
mdl_name = 'vw' + file_name1 + ".mdl"
vw_cmd_str = utils.vw_path + fn_t + ".gz --random_seed " + str(rseed) + " " + \
"--passes " + str(n_passes) + " -c --progress 1000000 --loss_function logistic -b 25 " +  holdout_str + \
"--l2 1e-7 -q CS -q CM -q MS -l .1 --power_t .5 -q NM -q NS --decay_learning_rate .75 --hash all " + \
" -q SX -q MX -q SY -q MY -q SZ -q MZ -q NV -q MV -q VX -q VY -q VZ" + \
" --ignore H -f " + mdl_name + " -k --compressed"
print vw_cmd_str
os.system(vw_cmd_str)

vw_cmd_str = utils.vw_path + fn_v + ".gz --hash all " + \
    "-i " + mdl_name + " -p " + fn_v + "_pred.txt -t --loss_function logistic --progress 200000"
print vw_cmd_str
os.system(vw_cmd_str)

