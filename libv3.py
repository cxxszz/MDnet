import torch
import torch.nn as nn
import torch.nn.functional as F
import options
import pickle
import os
import numpy as np
import cv2
from collections import OrderedDict
from torch.autograd import Variable
from sklearn.linear_model import Ridge
"""
boundingbox modified to speed up the region extraction: force every example generated to satisfy the
IoU requirement, rather than generating them and checking
updated Binary Loss
changed bounding box fetch from bgr_img[r1:r2+1,c1:c2+1,:] to bgr_img[r1:r2,c1:c2,:] 
that is consistent with area() computation: the nb of pixels a bb contains
"""
class mdnet(nn.Module):
    def __init__(self):
        super(mdnet,self).__init__()
        self.conv1=nn.Conv2d(
            in_channels=3,out_channels=96,kernel_size=(7,7),stride=(2,2)
        )
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))
        self.conv2=nn.Conv2d(
            in_channels=96,out_channels=256,kernel_size=(5,5),stride=(2,2)
        )
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))
        self.conv3=nn.Conv2d(
            in_channels=256,out_channels=512,kernel_size=(3,3),stride=(1,1)
        )
        self.relu3=nn.ReLU()
        self.drop4=nn.Dropout2d(p=0.5)
        self.fc4=nn.Linear(in_features=512*3*3,out_features=512)
        self.relu4=nn.ReLU()

        self.drop5=nn.Dropout2d(p=0.5)
        self.fc5=nn.Linear(in_features=512,out_features=512)
        self.relu5=nn.ReLU()

        self.drop6=nn.Dropout2d(p=0.5)
        self.branches=nn.ModuleList([nn.Linear(in_features=512,out_features=2)]*(K+1))
        self.ls=nn.LogSoftmax()
    def forward(self,inputs,branch_id):
        x=self.conv1(inputs)
        x=self.relu1(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)

        x=self.conv3(x)
        x=self.relu3(x)

        x=x.view(-1,512*3*3)
        x=self.drop4(x)
        x=self.fc4(x)
        x=self.relu4(x)

        x=self.drop5(x)
        x=self.fc5(x)
        x=self.relu5(x)

        x=self.drop6(x)
        x=self.branches[branch_id](x)
        return self.ls(x)

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        # pos_loss = -F.log_softmax(pos_score)[:, 0]
        # neg_loss = -F.log_softmax(neg_score)[:, 1]
        #
        # loss = pos_loss.sum() + neg_loss.sum()
        # return loss
        pos_loss=pos_score[:,0].sum()
        neg_loss=neg_score[:,1].sum()
        return -(pos_loss+neg_loss)
# def fetch_needed_region_with_a_bb_imgpath_pair(bb,bgrimgpath):
#     tmp=bb.fetch_given_path(bgrimgpath)
#     return cv2.resize(tmp,dsize=crop_size)
def fetch_needed_region_with_a_bb_img_pair(bb,bgr_img):
    tmp=bb.fetch(bgr_img)
    # assert tmp.shape[0]>=1 and tmp.shape[1]>=1,"tmp shape={},bb={}".format(tmp.shape,bb)
    return cv2.resize(tmp, dsize=crop_size)
# accepts a string that is "vot2016/****" probably from a text file
# returns two variables: a list that has image paths, which can be used to load images directly
# and a 2d array containing groundtruch bounding boxes in the corresponding images,
# each of which is a tuple (xmin,ymin,xmax,ymax)
def paths_gts(seqname):
    vot_folder_path = options.vot_folder_path
    imgnamelist=sorted(
            [imgname for imgname in os.listdir(os.path.join(vot_folder_path,seqname))
             if os.path.splitext(imgname)[1]==".jpg"]#split the path text with the last period
    )
    gt2dary_tmp=np.loadtxt(
            os.path.join(vot_folder_path,seqname,"groundtruth.txt"),delimiter=","
    )
    if gt2dary_tmp.shape[1]==8:
        # x_mins = np.min(gt2dary_tmp[:, [0, 2, 4, 6]], axis=1)[:, None]
        # y_mins = np.min(gt2dary_tmp[:, [1, 3, 5, 7]], axis=1)[:, None]
        # x_maxs = np.max(gt2dary_tmp[:, [0, 2, 4, 6]], axis=1)[:, None]
        # y_maxs = np.max(gt2dary_tmp[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_mins = np.min(gt2dary_tmp[:, [0, 2, 4, 6]], axis=1)
        y_mins = np.min(gt2dary_tmp[:, [1, 3, 5, 7]], axis=1)
        x_maxs = np.max(gt2dary_tmp[:, [0, 2, 4, 6]], axis=1)
        y_maxs = np.max(gt2dary_tmp[:, [1, 3, 5, 7]], axis=1)
    assert len(imgnamelist)==len(x_mins)
    imgpathlist=[None]*len(imgnamelist)
    gtary=np.zeros(shape=(len(imgpathlist),4))
    gtary[:,0]=x_mins
    gtary[:,1]=y_mins
    gtary[:,2]=x_maxs
    gtary[:,3]=y_maxs
    for i in range(len(imgnamelist)):
        imgpathlist[i]=os.path.join(vot_folder_path,seqname,imgnamelist[i])
    # for test purpose
    # img=cv2.imread(imgpathlist[0])
    # t=gtary[0,:]
    # tl=(int(t[0]),int(t[1]))
    # br=(int(t[2]),int(t[3]))
    # cv2.rectangle(img,tl,br,(0,255,0))
    # cv2.imshow("img",img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return imgpathlist,gtary.astype(np.int)
# def clip(x):
#     return np.clip(x,0,int(1e5))
def xml_read(xml_path):
    """

    :param xml_path:
    :return: a (xmin,ymin,xmax,ymax) tuple
    """
class boundingbox(object):
    # 0.262 s needed to construct 5000 bounding boxes using list(map()) but 0.260 s used in list
    corner_sigma=options.corner_sigma
    center_sigma=options.center_sigma
    def __init__(self,tp):
        """
        speed matters,and comment assertions when you are finally competing
        :param tp:
        """
        assert type(tp)==tuple or type(tp)==np.ndarray,"can't initialize a bounding box with a {}".format(type(tp))
        assert len(tp)==4,"{} is not a valid boundingbox".format(tp)
        # self.xmin,self.ymin,self.xmax,self.ymax=int(tp[0]+0.5),int(tp[1]+0.5),int(tp[2]+0.5),int(tp[3]+0.5)
        # self.xmin=clip(self.xmin)
        # self.ymin=clip(self.ymin)
        # self.xmax=clip(self.xmax)
        # self.ymax=clip(self.ymax)
        self.xmin,self.ymin,self.xmax,self.ymax=np.clip(tp,a_min=0,a_max=1e5).astype(int)
        # before assertion,int and clip
        assert self.xmin<=self.xmax and self.ymin<=self.ymax,"{} is not a valid bounding box".format(tp)
        self.hei=self.ymax-self.ymin
        self.wid=self.xmax-self.xmin
    def __repr__(self):
        return ",".join((
            str(self.xmin),str(self.ymin),str(self.xmax),str(self.ymax)
        ))
    def area(self):
        # returns a float
        return 1.0*(self.xmax-self.xmin)*(self.ymax-self.ymin)# the nb of pixels the bb contains
    def fetch(self,bgr_img):
        # opencv manages axes as (rows,cols,colours)
        # a key issue of speed
        r1,r2,c1,c2=np.clip((self.ymin,self.ymax,self.xmin,self.xmax),
                            np.array([0,1,0,1]),
                            np.array([bgr_img.shape[0]-1,bgr_img.shape[0],bgr_img.shape[1]-1,bgr_img.shape[1]]))
        assert r1<r2 and c1<c2,"r1={},r2={},img hei={},c1={},c2={},img_wid={}".format(
            r1,r2,bgr_img.shape[0],c1,c2,bgr_img.shape[1]
        )
        return bgr_img[r1:r2,c1:c2,:]
    def fetch_given_path(self,bgr_img_path):
        # speed bottleneck, don't use this!
        bgr_img=cv2.imread(bgr_img_path)
        assert isinstance(bgr_img,type(None))!=True,\
        "{} doesn't exist".format(bgr_img_path)
        return self.fetch(bgr_img)
    def IoU(self,bb1):
        # a key issue of speed
        Idx=min(self.xmax,bb1.xmax)-max(self.xmin,bb1.xmin)
        Idy=min(self.ymax,bb1.ymax)-max(self.ymin,bb1.ymin)
        Intersection=Idx*Idy
        Union=self.area()+bb1.area()-Intersection
        return 1.0*Intersection/Union
    def IoUs(self,bbs):
        # a key issue of speed
        # but don't implement multi processes version now
        # list of floats
        return list(map(self.IoU,bbs)) # how about tuple(map())?
    # a list containing options.pos_sps positive bounding boxes
    # and a list having options.neg_sps negative bounding boxes
    # the 2 lists forms one list to retun
    def off_sampling(self,constant_length=True):
        pos_sps=list(map(self.one_pos_sp,[None]*options.pos_sps))
        neg_sps=list(map(self.one_neg_sp,[None]*options.neg_sps))
        # keeping pos_sps and neg_sps always have the same size helps to analyze loss
        return [pos_sps,neg_sps]
    def on_sampling(self):
        pass
    # to generate a positive sample, we only need to disturb (xmin,ymin) and (xmax,ymax) a bit
    def one_pos_sp(self,ctrl=None):
        pxmin,pymin,pxmax,pymax=np.clip(
            np.random.random_sample(size=4),a_min=0.5-self.corner_sigma,a_max=0.5+self.corner_sigma
        )# should be faster since it call numpy only once
        xmin = self.xmin+self.wid*(-0.5+pxmin)
        ymin = self.ymin+self.hei*(-0.5+pymin)
        xmax = self.xmax+self.wid*(-0.5+pxmax)
        ymax = self.ymax+self.hei*(-0.5+pymax)
        # before assertion, int and clip
        xmin,ymin,xmax,ymax=clip(xmin),clip(ymin),clip(xmax),clip(ymax)
        # done in boundingbox initialization
        return boundingbox((xmin,ymin,xmax,ymax))
    # to generate a negative sample, we need to disturb the center of the bounding box as well as
    # those corners
    def one_neg_sp(self,ctrl=None):
        left_right=(-1,1)
        signhori,signvert=np.random.randint(low=0,high=2,size=2)
        signhori=left_right[int(signhori)]
        signvert=left_right[int(signvert)]
        phori,pvert=np.clip(
            np.random.random_sample(size=2),a_min=0.56,a_max=1
        )
        phori=signhori*phori
        pvert=signvert*pvert
        hori_delta=phori*self.wid
        vert_delta=pvert*self.hei
        pxmin, pymin, pxmax, pymax = np.clip(
            np.random.random_sample(size=4), a_min=0.5 - self.corner_sigma, a_max=0.5 + self.corner_sigma
        )  # should be faster since it call numpy only once
        xmin = self.xmin + self.wid * (-0.5 + pxmin)+hori_delta
        ymin = self.ymin + self.hei * (-0.5 + pymin)+vert_delta
        xmax = self.xmax + self.wid * (-0.5 + pxmax)+hori_delta
        ymax = self.ymax + self.hei * (-0.5 + pymax)+vert_delta
        # xmin, ymin, xmax, ymax = clip(xmin), clip(ymin), clip(xmax), clip(ymax)
        # done in boundingbox initialization
        return boundingbox((xmin,ymin,xmax,ymax))
    def to_tuple(self):
        return (self.xmin,self.ymin,self.xmax,self.ymax)
    def to_tl_range_tuple(self):
        return (self.xmin,self.ymin,self.wid,self.hei)
class SampleGenerator():
    def __init__(self, type,trans_f=1.0, scale_f=1.0, aspect_f=None, valid=False):
        self.type = type
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid
    def __call__(self, bb, n,img_size=None):
        # img_size = np.array(img_size) box adjustment is handled in boundingbox class
        # bb-> target array (xmin,ymin,wid,hei)
        target=np.array([
            bb.xmin,bb.ymin,bb.wid,bb.hei
        ])
        # (center_x, center_y, wid, hei)
        sample = np.array([
            (bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, bb.xmax - bb.xmin, bb.ymax - bb.ymin
        ], dtype=np.float32)
        samples = np.tile(sample, (n, 1))
        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n, 1) * 2 - 1
            samples[:, 2:] *= self.aspect_f ** np.concatenate([ratio, -ratio], axis=1)
        if self.type == 'gaussian':
            samples[:, :2] += self.trans_f * np.mean(target[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            samples[:, 2:] *= self.scale_f ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)
        elif self.type == 'uniform':
            samples[:, :2] += self.trans_f * np.mean(target[2:]) * (np.random.rand(n, 2) * 2 - 1)
            samples[:, 2:] *= self.scale_f ** (np.random.rand(n, 1) * 2 - 1)
        else:
            assert "unknown type %s"%(str(type(type)))
        # elif self.type == 'whole':
        #     m = int(2 * np.sqrt(n))
        #     xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
        #     xy = np.random.permutation(xy)[:n]
        #     samples[:, :2] = target[2:] / 2 + xy * (img_size - target[2:] / 2 - 1)
        #     # samples[:,:2] = bb[2:]/2 + np.random.rand(n,2) * (self.img_size-bb[2:]/2-1)
        #     samples[:, 2:] *= self.scale_f ** (np.random.rand(n, 1) * 2 - 1)
        #
        # # adjust bbox range
        # samples[:, 2:] = np.clip(samples[:, 2:], 1, img_size - 1)
        # if self.valid:
        #     samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, img_size - samples[:, 2:] / 2 - 1)
        # else:
        #     samples[:, :2] = np.clip(samples[:, :2], 0, img_size)
        # samples (center_x, center_y, wid, hei)s->(xmin, ymin, wid, hei)s
        samples[:, :2] -= samples[:, 2:] / 2
        samples[:, 2:] += samples[:, :2]
        # (xmin, ymin, wid, hei)s->(xmin,ymin,xmax,ymax)s
        return list(map(boundingbox,samples))#currently not in parallel

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f
class BBRegressor():
    def __init__(self, alpha=1000, overlap=(0.6, 1), scale=(1, 2)):
        # self.img_size = img_size
        self.alpha = alpha
        self.overlap_range = overlap
        self.scale_range = scale
        self.model = Ridge(alpha=self.alpha)

    """
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    self.train() method is the only place we really calculate IoU
    negative regions calculation wastes time but doesn't matter since process it only in the first frame
    """
    def train(self, feature_tensor, bblist, bbgt):
        """
            doesn't worry too much about time usage
        """

        r=bbgt.IoUs(bblist)
        s=[bb.area()/bbgt.area() for bb in bblist]# class generator doesn't define __getitem__
        feature_array = feature_tensor.cpu().data.numpy()
        indices=[i for i in range(len(bblist))
             if r[i]>=self.overlap_range[0] and
             r[i]<=self.overlap_range[1]and
             s[i]>=self.scale_range[0] and
             s[i]<=self.scale_range[1]]
        if len(indices)==0:
            bblist=[bbgt]
        feature_array=np.array([feature_array[i] for i in indices])
        bblist=[bblist[i] for i in indices]
        Y = self.get_examples(bblist, bbgt)
        self.model.fit(feature_array, Y)

    def predict(self, feature_tensor, bblist):
        """
        speed matters
        :param feature_tensor:
        :param bblist: list a small amount of bbs input
        :return:
        """
        assert isinstance(bblist,list),"type(bblist) is {}".format(type(bblist))
        feature_array = feature_tensor.cpu().data.numpy()
        bbox = [(bb.xmin, bb.ymin, bb.wid, bb.hei) for bb in bblist]
        # slightly faster than [[bb.xmin,bb.ymin,bb.wid,bb.hei] for bb in bblist]
        bbox = np.array(bbox)
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:] / 2
        bbox_ = np.copy(bbox)
        Y = self.model.predict(feature_array)

        bbox_[:, :2] = Y[:, :2] * bbox_[:, 2:] + bbox_[:, :2]
        bbox_[:, 2:] = np.exp(Y[:, 2:]) * bbox_[:, 2:]
        bbox_[:, :2] = bbox_[:, :2] - bbox_[:, 2:] / 2# now it is of (xmin,ymin,wid,hei) form
        bbox_[:, 2:] = bbox_[:, :2] + bbox_[:, 2:] # now it is of (xmin,ymin,xmax,ymax) standard bounding box form
        bblist_=list(map(boundingbox,bbox_))
        # r = overlap_ratio(bbox, bbox_)
        # s = np.prod(bbox[:,2:], axis=1) / np.prod(bbox_[:,2:], axis=1)

        r=[u[0].IoU(u[1]) for u in zip(bblist,bblist_)]
        s=[u[0].area()/u[1].area() for u in zip(bblist,bblist_)]
        # assert len(r)>0,"len(r)={}".format(len(r))
        # assert len(s)==len(bblist),"len(s)={},but len(bblist)={}".format(len(s),len(bblist))
        # assert len(self.scale_range)==2 and len(self.overlap_range)==2,"check scale_range or overlap_range"
        indices = [i for i in range(len(bblist))
               if r[i] >= self.overlap_range[0] and
               r[i] <= self.overlap_range[1] and
               s[i] >= self.scale_range[0] and
               s[i] <= self.scale_range[1]]
        # bbox_[idx] = bbox[idx]
        bblist_=[bblist[i] for i in indices]
        # bbox_[:, :2] = np.maximum(bbox_[:, :2], 0)
        # bbox_[:, 2:] = np.minimum(bbox_[:, 2:], self.img_size - bbox[:, :2])

        return bblist_
    def get_examples(self, bblist, bbgt):
        """
        bblist contains bbs that are not too far away from the bbgt and are slightly bigger than bbgt
        but why are slightly smaller bbs also abandoned?
        :param bblist:
        :param bbgt:
        :return:
        """
        bbox=[(bb.xmin,bb.ymin,bb.wid,bb.hei) for bb in bblist]

        bbox=np.array(bbox).astype(float)
        gt = np.array((bbgt.xmin, bbgt.ymin, bbgt.wid, bbgt.hei)).astype(float)
        gt=np.tile(gt,(1,1))
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:] / 2
        gt[:, :2]   = gt[  :, :2] + gt[  :, 2:] / 2

        dst_xy = (gt[:, :2] - bbox[:, :2]) / bbox[:, 2:]
        dst_wh = np.log(gt[:, 2:] / bbox[:, 2:])

        Y = np.concatenate((dst_xy, dst_wh), axis=1)
        return Y
class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]
class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)

        return prec.data[0]
K=options.K
crop_size=options.crop_size
pos_generator=SampleGenerator('gaussian',options.pos_trans_f,options.pos_scale_f,options.pos_aspect_f,True)
neg_generator=SampleGenerator('uniform', options.neg_trans_f,options.neg_scale_f,options.neg_aspect_f,True)
sp_generator=SampleGenerator("gaussian",options.next_trans_f,options.next_scale_f,valid=True)
criterion = BinaryLoss()
def boundingbox_mean(bbs):
    assert len(bbs)>0,"bbs is empty"
    if len(bbs)==1:
        return bbs[0]
    bbs_array_tmp=[(bb.xmin,bb.ymin,bb.xmax,bb.ymax) for
                   bb in bbs]# numpy cannot convert a generator instance to a meaningful array
    bbs_array=np.array(bbs_array_tmp)
    # assert bbs_array.ndim==2,"bbs_array={}".format(bbs_array)
    bb_mean_array=np.mean(bbs_array,axis=0)
    return boundingbox(bb_mean_array)
def online_set_requires_grad(net):
    # determine train() or eval() when you actually train or eval
    if torch.cuda.is_available():
        net=net.cuda()
    net.conv1.requires_grad = False
    net.conv2.requires_grad = False
    net.conv3.requires_grad = False
    net.fc4.requires_grad = True
    net.fc5.requires_grad = True
    net.branches[K].requires_grad = True
    return net
def offline_train_set_requires_grad(net,branch_id):
    net.train()
    if torch.cuda.is_available():
        net=net.cuda()
    net.conv1.requires_grad=True
    net.conv2.requires_grad=True
    net.conv3.requires_grad=True
    net.fc4.requires_grad=True
    net.fc5.requires_grad=True
    net.branches[branch_id].requires_grad=True
    for k in range(K):
        if k!=branch_id:
            net.branches[branch_id].requires_grad = False
    return net
# def extract_regions_in_a_frame(imgpath, bb):
#     pos_bbs, neg_bbs = bb.off_sampling()
#     img = cv2.imread(imgpath)
#     pos_regions = list(map(
#         fetch_needed_region_with_a_bb_img_pair, pos_bbs, [img] * len(pos_bbs)
#     ))
#     neg_regions = list(map(
#         fetch_needed_region_with_a_bb_img_pair, neg_bbs, [img] * len(neg_bbs)
#     ))
#     pos_regions = np.array(pos_regions).transpose(0, 3, 1, 2)
#     neg_regions = np.array(neg_regions).transpose(0, 3, 1, 2)
#     return [pos_regions, neg_regions]
# def extract_regions(idx, imgpaths, bbs, frames=options.frames):
#     tmp = list(map(
#         extract_regions_in_a_frame, imgpaths[idx:idx + frames], bbs[idx:idx + frames]
#     ))  # [[pos_regions,neg_regions],[pos_regions,neg_regions],...[pos_regions,neg_regions]]
#     pos_regions = tmp[0][0]
#     neg_regions = tmp[0][1]
#     for i in range(1, frames):
#         pos_regions = np.concatenate([pos_regions, tmp[i][0]], axis=0)
#         neg_regions = np.concatenate([neg_regions, tmp[i][1]], axis=0)
#     return pos_regions, neg_regions

def extract_regions_with_SampleGenerator_in_a_frame(img, bb, pos_sps=options.pos_sps, neg_sps=options.neg_sps):

    pos_bbs=pos_generator(bb,pos_sps)
    neg_bbs=neg_generator(bb,neg_sps)
    pos_regions = list(map(
        fetch_needed_region_with_a_bb_img_pair, pos_bbs, [img] * len(pos_bbs)
    ))
    neg_regions = list(map(
        fetch_needed_region_with_a_bb_img_pair, neg_bbs, [img] * len(neg_bbs)
    ))# can converting to lists be avoided?
    # but is the time mainly used during map or during converting to lists
    pos_regions = np.array(pos_regions).transpose(0, 3, 1, 2)
    neg_regions = np.array(neg_regions).transpose(0, 3, 1, 2)
    return pos_regions, neg_regions# a bit faster than the one above

def mdnet_offline_one_frame_train(net,branch_id,optimizer,img,bb,
                                  pos_sps=options.pos_sps,neg_sps=options.neg_sps):
    # net=offline_train_set_requires_grad(net,branch_id)
    # don't do it every frame. it wastes time
    # don't need to set train() after offline_train_set_requires_grad()
    pos_regions, neg_regions = extract_regions_with_SampleGenerator_in_a_frame(img, bb, pos_sps, neg_sps)
    pos_regions = Variable(torch.from_numpy(pos_regions).float())
    neg_regions = Variable(torch.from_numpy(neg_regions).float())
    if torch.cuda.is_available():
        pos_regions = pos_regions.cuda()
        neg_regions = neg_regions.cuda()
    pos_scores = net(pos_regions, branch_id)
    neg_scores = net(neg_regions, branch_id)
    loss = criterion(pos_scores, neg_scores)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(net.parameters(), 10)
    optimizer.step()
    return loss
def mdnet_online_one_frame_train(net,branch_id,optimizer,img,bb,
                                  pos_sps=options.pos_sps,neg_sps=options.neg_sps,
                                 pos_regions=0,neg_regions=0):
    # train and eval switches
    net.train()
    if isinstance(pos_regions,int):
        pos_regions, neg_regions = extract_regions_with_SampleGenerator_in_a_frame(img, bb,pos_sps,neg_sps)
        pos_regions = Variable(torch.from_numpy(pos_regions).float())
        neg_regions = Variable(torch.from_numpy(neg_regions).float())
    if torch.cuda.is_available():
        pos_regions = pos_regions.cuda()
        neg_regions = neg_regions.cuda()
    pos_scores = net(pos_regions, branch_id)
    neg_scores = net(neg_regions, branch_id)
    loss = criterion(pos_scores, neg_scores)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(net.parameters(), 10)
    optimizer.step()
    return loss,pos_regions,neg_regions


def regions_estimate_with_SampleGenerator_in_a_frame(img,cur_bb,net,
                                                     branch_id=options.K,estimate_sps=options.estimate_sps):
    net.eval()# train and eval switches
    sample_bbs = sp_generator(cur_bb, estimate_sps)
    sample_regions=list(map(
        fetch_needed_region_with_a_bb_img_pair, sample_bbs, [img]*len(sample_bbs)
    ))#better be a list as bbreg_sample_regions is also a list. an array is ok as well
    sample_regions=np.array(sample_regions).transpose(0,3,1,2)
    inputs=Variable(torch.from_numpy(sample_regions).float()).cuda()

    sample_scores = net(inputs,branch_id)
    return sample_bbs,sample_regions,sample_scores
def mdnet_conv3_output(net,inputs):
    x = net.conv1(inputs)
    x = net.relu1(x)
    x = net.pool1(x)

    x = net.conv2(x)
    x = net.relu2(x)
    x = net.pool2(x)

    x = net.conv3(x)
    x = net.relu3(x)

    x = x.view(-1, 512 * 3 * 3)
    return x
def boundingbox_regressor_train(net,imgpath,bbgt):
    img = cv2.imread(imgpath)
    reg_generator = SampleGenerator("uniform", 0.3, 1.5, 1.1)
    # only use it once, so put it inside the function to reduce memory usage
    reg_bbs=reg_generator(bbgt,options.reg_sps)

    reg_regions=list(map(
        fetch_needed_region_with_a_bb_img_pair, reg_bbs, [img] * len(reg_bbs)#resized
    ))
    reg_regions=np.array(reg_regions).transpose(0,3,1,2)
    reg_regions=Variable(torch.from_numpy(reg_regions).float())
    if torch.cuda.is_available():
        reg_regions=reg_regions.cuda()
    features=mdnet_conv3_output(net,reg_regions)
    # bbr=BBRegressor(img.shape)
    bbr=BBRegressor()
    bbr.train(features,reg_bbs,bbgt)
    return bbr
def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou
    
if __name__ == '__main__':
    # paths_gts("vot2016/marching")
    bb1=boundingbox((55,55,205,205))
    pos_sps,neg_sps=bb1.off_sampling(constant_length=False)
    print(len(pos_sps),len(neg_sps))
