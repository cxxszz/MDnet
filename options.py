"""
all parameters must remain the same, otherwise various versions of mdnet cannot compare with each other
"""
#data_generation.py
crop_size=(107,107)
motion_sigma=(10,10)
frames=1
#annotated sequence attributes
attr={}
attr["iv"]=0#illumination variation
attr["sv"]=1#scale variation
attr["occ"]=2#occlusion
attr["def"]=3#deformation
attr["mb"]=4#motion blur
attr["fm"]=5#fast motion
attr["ipr"]=6#in-plane rotation
attr["opr"]=7#out-of-plane rotation
attr["ov"]=8#out of view
attr["bc"]=9#background clutters
attr["lr"]=10#low resolution
K=44
#train
vot_folder_path = ".."

corner_sigma=1.0/23
center_sigma=0.56


pos_thre=0.7
neg_thre=0.3

success_thr=-0.3567#ln(0.7)
# trans_f
pos_trans_f=0.1
neg_trans_f=1.0
next_trans_f,trans_f_expand=0.6,1.5
# scale_f
pos_scale_f=1.2
neg_scale_f=1.2
next_scale_f=1.05
# aspect_f
pos_aspect_f=1.1
neg_aspect_f=1.1
sp_aspect_f=1.1
# nb of samples are defined below. with too many samples, the program will run out of memory
estimate_sps=256
reg_sps=1000
frame1_pos_sps=480
frame1_neg_sps=1100
max_pos_sps_online,max_neg_sps_online=400,1100# slightly smaller than frame1_pos_sps and frame1_neg_sps respectively
pos_sps=32
neg_sps=96
pos_update_sps,neg_update_sps=50,200
n_frame_long,n_frame_short=100,20
pos_all_sps,neg_all_sps=n_frame_long*pos_update_sps,n_frame_short*neg_update_sps
online_train_interval=10
serializations=max(pos_all_sps//max_pos_sps_online,neg_all_sps//max_neg_sps_online)# int
#(serilizations+1)*max_pos_sps_online>pos_regions_all>=serilizations*max_pos_sps_online.so are negatives