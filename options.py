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
data_path="vot2016.pkl"
n_frame_long,n_frame_short=100,20
corner_sigma=1.0/23
center_sigma=0.56
pos_sps=32
neg_sps=96
reg_sps=1000
pos_thre=0.7
neg_thre=0.3
frame1_pos_sps=500
frame1_neg_sps=500
pos_trans_f=0.1
pos_scale_f=1.2
pos_aspect_f=1.1
neg_trans_f=1.0
neg_scale_f=1.2
neg_aspect_f=1.1
next_trans_f,trans_f_expand=0.6,1.5
next_scale_f=1.05
estimate_sps=256
success_thr=0.79
"""30.0/(32+96)=0.23,30 for now
0.23=-LnP
hence, P=0.79"""
pos_update_sps,neg_update_sps=50,200
long_term_interval=10