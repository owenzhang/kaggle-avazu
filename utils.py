import numpy as np
import pandas as pd
from sklearn.utils import check_random_state 
import time
import sys
from joblib import dump, load

sample_pct = .05
tvh = 'N'
xgb_n_trees = 300


#Please set following path accordingly

#where we can find training, test, and sampleSubmission.csv
raw_data_path = '/home/fast/2014_mobilectr/'
#where we store results -- require about 130GB
tmp_data_path = './tmp_data/'

#path to external binaries. Please see dependencies in the .pdf document
fm_path = ' ~/Downloads/guestwalk/kaggle-2014-criteo/fm'
xgb_path = '/home/zzhang/Downloads/xgboost/wrapper'
vw_path = '~/vowpal_wabbit/vowpalwabbit/vw '


try:
    params=load(tmp_data_path + '_params.joblib_dat')
    sample_pct = params['pct']
    tvh = params['tvh']
except:
    pass


def print_help():
    print "usage: python utils -set_params [tvh=Y|N], [sample_pct]"
    print "for example: python utils -set_params N 0.05"

def main():
    if sys.argv[1] == '-set_params' and len(sys.argv) == 4:
        try:
            tvh = sys.argv[2]
            sample_pct = float(sys.argv[3])
            dump({'pct': sample_pct, 'tvh':tvh}, tmp_data_path + '_params.joblib_dat')
        except:
            print_help()
    else:
        print_help()

if __name__ == "__main__":
    main()

def get_agg(group_by, value, func):
    g1 = pd.Series(value).groupby(group_by)
    agg1  = g1.aggregate(func)
    #print agg1
    r1 = agg1[group_by].values
    return r1

def calcLeaveOneOut2(df, vn, vn_y, cred_k, r_k, power, mean0=None, add_count=False):
    if mean0 is None:
        mean0 = df_yt[vn_y].mean() * np.ones(df.shape[0])
    _key_codes = df[vn].values.codes
    grp1 = df[vn_y].groupby(_key_codes)
    grp_mean = pd.Series(mean0).groupby(_key_codes)
    mean1 = grp_mean.aggregate(np.mean)
    sum1 = grp1.aggregate(np.sum)
    cnt1 = grp1.aggregate(np.size)
    
    #print sum1
    #print cnt1
    vn_sum = 'sum_' + vn
    vn_cnt = 'cnt_' + vn
    _sum = sum1[_key_codes].values
    _cnt = cnt1[_key_codes].values
    _mean = mean1[_key_codes].values
    #print _sum[:10]
    #print _cnt[:10]
    #print _mean[:10]
    #print _cnt[:10]
    _mean[np.isnan(_sum)] = mean0.mean()
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    #print _cnt[:10]
    _sum -= df[vn_y].values
    _cnt -= 1
    #print _cnt[:10]
    vn_yexp = 'exp2_'+vn
#    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    diff = np.power((_sum + cred_k * _mean)/(_cnt + cred_k) / _mean, power)
    if vn_yexp in df.columns:
        df[vn_yexp] *= diff
    else:
        df[vn_yexp] = diff 
    if r_k > 0:
        df[vn_yexp] *= np.exp((np.random.rand(np.sum(filter_train))-.5) * r_k)
    if add_count:
        df[vn_cnt] = _cnt
    return diff


def my_lift(order_by, p, y, w, n_rank, dual_axis=False, random_state=0, dither=1e-5, fig_size=None):
    gen = check_random_state(random_state)
    if w is None:
        w = np.ones(order_by.shape[0])
    if p is None:
        p = order_by
    ord_idx = np.argsort(order_by + dither*np.random.uniform(-1.0, 1.0, order_by.size))
    p2 = p[ord_idx]
    y2 = y[ord_idx]
    w2 = w[ord_idx]

    cumm_w = np.cumsum(w2)
    total_w = cumm_w[-1]
    r1 = np.minimum(n_rank, np.maximum(1, 
                    np.round(cumm_w * n_rank / total_w + .4999999)))
    
    df1 = pd.DataFrame({'r': r1, 'pw': p2 * w2, 'yw': y2 * w2, 'w': w2})
    grp1 = df1.groupby('r')
    
    sum_w = grp1['w'].aggregate(np.sum)
    avg_p = grp1['pw'].aggregate(np.sum) / sum_w 
    avg_y = grp1['yw'].aggregate(np.sum) / sum_w
    
    xs = xrange(1, n_rank+1)
    
    fig, ax1 = plt.subplots()
    if fig_size is None:
        fig.set_size_inches(20, 15)
    else:
        fig.set_size_inches(fig_size)
    ax1.plot(xs, avg_p, 'b--')
    if dual_axis:
        ax2 = ax1.twinx()
        ax2.plot(xs, avg_y, 'r')
    else:
        ax1.plot(xs, avg_y, 'r')
    
    #print "logloss: ", logloss(p, y, w)
    
    return gini_norm(order_by, y, w)

def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)

def gini_norm(pred, y, weight=None):

    #equal weight by default
    if weight == None:
        weight = np.ones(y.size)

    #sort actual by prediction
    ord = np.argsort(pred)
    y2 = y[ord]
    w2 = weight[ord]
    #gini by pred
    cumm_y = np.cumsum(y2)
    total_y = cumm_y[-1]
    total_w = np.sum(w2)
    g1 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    #sort actual by actual
    ord = np.argsort(y)
    y2 = y[ord]
    w2 = weight[ord]
    #gini by actual
    cumm_y = np.cumsum(y2)
    g0 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    return g1/g0

def mergeLeaveOneOut2(df, dfv, vn):
    _key_codes = df[vn].values.codes
    vn_yexp = 'exp2_'+vn
    grp1 = df[vn_yexp].groupby(_key_codes)
    _mean1 = grp1.aggregate(np.mean)
    
    _mean = _mean1[dfv[vn].values.codes].values
    
    _mean[np.isnan(_mean)] = _mean1.mean()

    return _mean
    
    
def calcTVTransform(df, vn, vn_y, cred_k, filter_train, mean0=None):
    if mean0 is None:
        mean0 = df.ix[filter_train, vn_y].mean()
        print "mean0:", mean0
    else:
        mean0 = mean0[~filter_train]
        
    df['_key1'] = df[vn].astype('category').values.codes
    df_yt = df.ix[filter_train, ['_key1', vn_y]]
    #df_y.set_index([')key1'])
    grp1 = df_yt.groupby(['_key1'])
    sum1 = grp1[vn_y].aggregate(np.sum)
    cnt1 = grp1[vn_y].aggregate(np.size)
    vn_sum = 'sum_' + vn
    vn_cnt = 'cnt_' + vn
    v_codes = df.ix[~filter_train, '_key1']
    _sum = sum1[v_codes].values
    _cnt = cnt1[v_codes].values
    _cnt[np.isnan(_sum)] = 0    
    _sum[np.isnan(_sum)] = 0
    
    r = {}
    r['exp'] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    r['cnt'] = _cnt
    return r

def cntDualKey(df, vn, vn2, key_src, key_tgt, fill_na=False):
    
    print "build src key"
    _key_src = np.add(df[key_src].astype('string').values, df[vn].astype('string').values)
    print "build tgt key"
    _key_tgt = np.add(df[key_tgt].astype('string').values, df[vn].astype('string').values)
    
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('string').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('string').values)

    print "aggreate by src key"
    grp1 = df.groupby(_key_src)
    cnt1 = grp1[vn].aggregate(np.size)
    
    print "map to tgt key"
    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    _cnt = cnt1[_key_tgt].values

    if fill_na is not None:
        print "fill in na"
        _cnt[np.isnan(_cnt)] = fill_na    

    vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
    if vn2 is not None:
        vn_cnt_tgt += '_' + vn2
    df[vn_cnt_tgt] = _cnt

def my_grp_cnt(group_by, count_by):
    _ts = time.time()
    _ord = np.lexsort((count_by, group_by))
    print time.time() - _ts
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    print time.time() - _ts
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in xrange(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            if count_by[_ord[i-1]] != count_by[i0]: 
                running_cnt += 1
        else:
            running_cnt = 1
            _prev_grp = group_by[i0]
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            j = i
            while True:
                j0 = _ord[j]
                _cs1[j0] = running_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
            
    print time.time() - _ts
    if True:
        return _cs1
    else:
        _ts = time.time()    

        org_idx = np.zeros(group_by.size, dtype=np.int)
        print time.time() - _ts
        _ts = time.time()    
        org_idx[_ord] = np.asarray(xrange(group_by.size))
        print time.time() - _ts
        _ts = time.time()    

        return _cs1[org_idx]
    
def my_cnt(group_by):
    _ts = time.time()
    _ord = np.argsort(group_by)
    print time.time() - _ts
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    runnting_cnt = 0
    for i in xrange(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            running_cnt += 1
        else:
            running_cnt = 1
            _prev_grp = group_by[i0]
        if i == group_by.size - 1 or group_by[i0] != group_by[_ord[i+1]]:
            j = i
            while True:
                j0 = _ord[j]
                _cs1[j0] = running_cnt
                if j == 0 or group_by[_ord[j-1]] != group_by[j0]:
                    break
                j -= 1
            
    print time.time() - _ts
    return _cs1

def my_grp_value_diff(group_by, order_by, value):
    _ts = time.time()
    _ord = np.lexsort((order_by, group_by))
    print time.time() - _ts
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    print time.time() - _ts
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    for i in xrange(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            _cs1[i0] = value[_ord[i]] - value[_ord[i-1]]
        else:
            _cs1[i0] = 1e7
            _prev_grp = group_by[i0]
    print time.time() - _ts
    
    return np.minimum(_cs1, 1e7)

def my_grp_idx(group_by, order_by):
    _ts = time.time()
    _ord = np.lexsort((order_by, group_by))
    print time.time() - _ts
    _ts = time.time()    
    _ones = pd.Series(np.ones(group_by.size))
    print time.time() - _ts
    _ts = time.time()    
    #_cs1 = _ones.groupby(group_by[_ord]).cumsum().values
    _cs1 = np.zeros(group_by.size)
    _prev_grp = '___'
    for i in xrange(1, group_by.size):
        i0 = _ord[i]
        if _prev_grp == group_by[i0]:
            _cs1[i] = _cs1[i - 1] + 1
        else:
            _cs1[i] = 1
            _prev_grp = group_by[i0]
    print time.time() - _ts
    _ts = time.time()    
    
    org_idx = np.zeros(group_by.size, dtype=np.int)
    print time.time() - _ts
    _ts = time.time()    
    org_idx[_ord] = np.asarray(xrange(group_by.size))
    print time.time() - _ts
    _ts = time.time()    

    return _cs1[org_idx]

def calcDualKey(df, vn, vn2, key_src, key_tgt, vn_y, cred_k, mean0=None, add_count=False, fill_na=False):
    if mean0 is None:
        mean0 = df[vn_y].mean()
    
    print "build src key"
    _key_src = np.add(df[key_src].astype('string').values, df[vn].astype('string').values)
    print "build tgt key"
    _key_tgt = np.add(df[key_tgt].astype('string').values, df[vn].astype('string').values)
    
    if vn2 is not None:
        _key_src = np.add(_key_src, df[vn2].astype('string').values)
        _key_tgt = np.add(_key_tgt, df[vn2].astype('string').values)

    print "aggreate by src key"
    grp1 = df.groupby(_key_src)
    sum1 = grp1[vn_y].aggregate(np.sum)
    cnt1 = grp1[vn_y].aggregate(np.size)
    
    print "map to tgt key"
    vn_sum = 'sum_' + vn + '_' + key_src + '_' + key_tgt
    _sum = sum1[_key_tgt].values
    _cnt = cnt1[_key_tgt].values

    if fill_na:
        print "fill in na"
        _cnt[np.isnan(_sum)] = 0    
        _sum[np.isnan(_sum)] = 0

    print "calc exp"
    if vn2 is not None:
        vn_yexp = 'exp_' + vn + '_' + vn2 + '_' + key_src + '_' + key_tgt
    else:
        vn_yexp = 'exp_' + vn + '_' + key_src + '_' + key_tgt
    df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)

    if add_count:
        print "add counts"
        vn_cnt_src = 'cnt_' + vn + '_' + key_src
        df[vn_cnt_src] = _cnt
        grp2 = df.groupby(_key_tgt)
        cnt2 = grp2[vn_y].aggregate(np.size)
        _cnt2 = cnt2[_key_tgt].values
        vn_cnt_tgt = 'cnt_' + vn + '_' + key_tgt
        df[vn_cnt_tgt] = _cnt2

def get_set_diff(df, vn, f1, f2):
    #print(df[vn].values.sum())
    set1 = set(np.unique(df[vn].values[f1]))
    set2 = set(np.unique(df[vn].values[f2]))
    set2_1 = set2 - set1
    print vn, '\t', len(set1), '\t', len(set2), '\t', len(set2_1)
    return len(set2_1) * 1.0 / len(set2)


def calc_exptv(t0, vn_list, last_day_only=False, add_count=False):
    t0a = t0.ix[:, ['day', 'click']].copy()
    day_exps = {}

    for vn in vn_list:
        if vn == 'dev_id_ip':
            t0a[vn] = pd.Series(np.add(t0.device_id.values , t0.device_ip.values)).astype('category').values.codes
        elif vn == 'dev_ip_aw':
            t0a[vn] = pd.Series(np.add(t0.device_ip.values , t0.app_or_web.astype('string').values)).astype('category').values.codes
        elif vn == 'C14_aw':
            t0a[vn] = pd.Series(np.add(t0.C14.astype('string').values , t0.app_or_web.astype('string').values)).astype('category').values.codes
        elif vn == 'C17_aw':
            t0a[vn] = pd.Series(np.add(t0.C17.astype('string').values , t0.app_or_web.astype('string').values)).astype('category').values.codes
        elif vn == 'C21_aw':
            t0a[vn] = pd.Series(np.add(t0.C21.astype('string').values , t0.app_or_web.astype('string').values)).astype('category').values.codes
        elif vn == 'as_domain':
            t0a[vn] = pd.Series(np.add(t0.app_domain.values , t0.site_domain.values)).astype('category').values.codes
        elif vn == 'site_app_id':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.app_id.values)).astype('category').values.codes
        elif vn == 'app_model':
            t0a[vn] = pd.Series(np.add(t0.app_id.values , t0.device_model.values)).astype('category').values.codes
        elif vn == 'app_site_model':
            t0a[vn] = pd.Series(np.add(t0.app_id.values , np.add(t0.site_id.values , t0.device_model.values))).astype('category').values.codes
        elif vn == 'site_model':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.device_model.values)).astype('category').values.codes
        elif vn == 'app_site':
            t0a[vn] = pd.Series(np.add(t0.app_id.values , t0.site_id.values)).astype('category').values.codes
        elif vn == 'site_ip':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.device_ip.values)).astype('category').values.codes
        elif vn == 'app_ip':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.device_ip.values)).astype('category').values.codes
        elif vn == 'site_id_domain':
            t0a[vn] = pd.Series(np.add(t0.site_id.values , t0.site_domain.values)).astype('category').values.codes
        elif vn == 'site_hour':
            t0a[vn] = pd.Series(np.add(t0.site_domain.values , (t0.hour.values % 100).astype('string'))).astype('category').values.codes
        else:
            t0a[vn] = t0[vn]

        for day_v in xrange(22, 32):
            cred_k = 10
            if day_v not in day_exps:
                day_exps[day_v] = {}

            vn_key = vn

            import time
            _tstart = time.time()

            day1 = 20
            if last_day_only:
                day1 = day_v - 2
            filter_t = np.logical_and(t0.day.values > day1, t0.day.values <= day_v)
            vn_key = vn
            t1 = t0a.ix[filter_t, :].copy()
            filter_t2 = np.logical_and(t1.day.values != day_v, t1.day.values < 31)
            
            if vn == 'app_or_web':
                day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'click', cred_k, filter_t2)
            else:
                if last_day_only:
                    day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'click', cred_k, filter_t2, mean0=t0.expld_app_or_web.values)
                else:
                    day_exps[day_v][vn_key] = calcTVTransform(t1, vn, 'click', cred_k, filter_t2, mean0=t0.exptv_app_or_web.values)
            
            print vn, vn_key, " ", day_v, " done in ", time.time() - _tstart
        t0a.drop(vn, inplace=True, axis=1)
        
    for vn in vn_list:
        vn_key = vn
            
        vn_exp = 'exptv_'+vn_key
        if last_day_only:
            vn_exp='expld_'+vn_key
            
        t0[vn_exp] = np.zeros(t0.shape[0])
        if add_count:
            t0['cnttv_'+vn_key] = np.zeros(t0.shape[0])
        for day_v in xrange(22, 32):
            print vn, vn_key, day_v, t0.ix[t0.day.values == day_v, vn_exp].values.size, day_exps[day_v][vn_key]['exp'].size
            t0.loc[t0.day.values == day_v, vn_exp]=day_exps[day_v][vn_key]['exp']
            if add_count:
                t0.loc[t0.day.values == day_v, 'cnttv_'+vn_key]=day_exps[day_v][vn_key]['cnt']
        
