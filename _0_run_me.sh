#small test run using day 30 as validation
python utils.py -set_params Y 1.0
python _1_encode_cat_features.py
python _2b_generate_dataset_for_vw_fm.py
python _2c_generate_fm_features.py
python _3a_rf.py
python _3b_gbdt.py
python _3c_vw.py -rseed 1
python _3c_vw.py -rseed 2
python _3c_vw.py -rseed 3
python _3c_vw.py -rseed 4
python _3d_fm.py -rseed 51
python _3d_fm.py -rseed 52
python _3d_fm.py -rseed 53
python _3d_fm.py -rseed 54
#should generate logloss ~= 0.3937
python _4_post_processing.py

exit
#try a quick submission using small sample data
python utils -set_params Y 0.05
python _3a_rf.py
python _3b_gbdt.py
python _3c_vw.py -rseed 1
python _3c_vw.py -rseed 2
python _3c_vw.py -rseed 3
python _3c_vw.py -rseed 4
python _3d_fm.py -rseed 51
python _3d_fm.py -rseed 52
python _3d_fm.py -rseed 53
python _3d_fm.py -rseed 54
#should generate a submission with score (Public LB) .3936: (Private LB): .3917
python _4_post_processing.py

#run the whole thing, will take about 2 days
python utils -set_params Y 1.0
python _1_encode_cat_features.py
python _2b_generate_dataset_for_vw_fm.py
python _2c_generate_fm_features.py
python _3a_rf.py
python _3b_gbdt.py
python _3c_vw.py -rseed 1
python _3c_vw.py -rseed 2
python _3c_vw.py -rseed 3
python _3c_vw.py -rseed 4
python _3d_fm.py -rseed 51
python _3d_fm.py -rseed 52
python _3d_fm.py -rseed 53
python _3d_fm.py -rseed 54
#should generate a submission with score (Private LB) ~ .3805
python _4_post_processing.py


