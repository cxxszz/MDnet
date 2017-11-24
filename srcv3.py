import options
import time
import torch
import cv2
import os
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
from libv3 import mdnet,boundingbox,paths_gts,extract_regions_with_SampleGenerator_in_a_frame,\
    online_set_requires_grad,offline_train_set_requires_grad,mdnet_conv3_output,\
    boundingbox_regressor_train,sp_generator,mdnet_offline_one_frame_train,\
    mdnet_online_one_frame_train,regions_estimate_with_SampleGenerator_in_a_frame,\
    boundingbox_mean

mdnetpath="mdnetv3.pt"
trained_net_path="trained_nets/mdnetv3.pt"
logs_directory="log3s"
results_directory="results"
def tracking_result(seqname,gtbb,cur_bb,img,img_nb,fps):
    """

    :param seqname: like "marching"
    :param gtbb:
    :param cur_bb:
    :param img:
    :param img_nb: frame_nb+1
    :param fps:
    :return:
    """
    if not os.path.exists(results_directory):
        os.system("mkdir {}".format(results_directory))
    the_result_directory=os.path.join(results_directory,seqname)#"
    if not os.path.exists(the_result_directory):
        os.system("mkdir {}".format(the_result_directory))
    imgs_directory=os.path.join(the_result_directory,"imgs")
    if not os.path.exists(imgs_directory):
        os.system("mkdir {}".format(imgs_directory))
    result_path=os.path.join(the_result_directory,"{}.txt".format(seqname))
    img_path=os.path.join(imgs_directory,"{}.jpg".format(img_nb))
    cv2.rectangle(img,(int(gtbb.xmin),int(gtbb.ymin)),(int(gtbb.xmax),int(gtbb.ymax)),(255,0,0))
    cv2.rectangle(img,(int(cur_bb.xmin),int(cur_bb.ymin)),(int(cur_bb.xmax),int(cur_bb.ymax)),(0,255,0))
    cv2.imwrite(img_path,img)
    with open(result_path,"a") as f:
        s=" ".join((
            str(gtbb),str(cur_bb),str(gtbb.IoU(cur_bb)),str(fps),"\n"
        ))
        print(s)
        f.write(s)
def offline_train(epoches,seqnames):
    for epoch in range(epoches):
        log_path = os.path.join(logs_directory, "log{}.txt".format(epoch))
        if os.path.exists(mdnetpath):
            print("loaded a model")
            net = torch.load(mdnetpath)
        else:
            net = mdnet()
        seqnameorder=np.random.permutation(len(seqnames))
        for i in seqnameorder:
            imgpathlist,gtary=paths_gts(seqnames[i])
            bbs=list(map(boundingbox,gtary))# bbs contains the whole video's groundtruth bounding box
            # bbs must be a list because every img corresponds to a bb
            branch_id=i%options.K
            video=seqnames[i]
            # set branch_id, layer and train() in every video
            net=offline_train_set_requires_grad(net,branch_id)
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
            # net changes when videos change so the optimizer changes too
            # optimizer=optim.SGD([
            #     {"params":net.conv1.parameters(),"lr":0.0001},
            #     {"params":net.conv2.parameters(),"lr":0.0001},
            #     {"params":net.conv3.parameters(),"lr":0.0001},
            #     {"params":net.fc4.parameters(),"lr":0.001},
            #     {"params":net.fc5.parameters(),"lr":0.001},
            #     {"params":net.branches[branch_id].parameters(),"lr":0.001}
            # ],momentum=0.9,weight_decay=0.0005)

            # print(len(imgpathlist),len(bbs))
            # bbs is ordered as the time order
            # and imgpathlist is also in chronological order
            # fetch patches that later will be converted to a pytorch tensor patch
            # roughly 300 regions every frame times about 300 frames
            # each region is 107*107*3byte(bgr three channels),
            # so 2.88GB(107*107*3*300*300 byte) might be needed at worst
            # therefore you don't need to write a map for all frames in a video sequence
            # but 9.8MB(107*107*3*300 byte) is possible
            # frame_nb_list=range(len(imgpathlist)-options.frames)[0::options.frames]
            # pythonic way to return every nth item in a larger list
            # in this case, simply like the one below
            for frame_nb in range(len(imgpathlist)):
                img=cv2.imread(imgpathlist[frame_nb])
                start = time.time()
                loss=mdnet_offline_one_frame_train(net,branch_id,optimizer,img,bbs[frame_nb])
                with open(log_path,mode="a") as f:
                    s="{},{},".format(
                            video,frame_nb+1)+\
                        "fps=%0.2f,loss=%0.7f\n"%(1.0/(time.time()-start),loss.cpu().data.numpy())
                    # in vot2016,images begin with ***1.jpg
                    print(s)
                    f.write(
                        s
                    )

        torch.save(net,mdnetpath)
# if a function is created as an instance, pos_sps and neg_sps should only be assigned once

def online_tracking(seqname):
    result=[]
    net=torch.load(trained_net_path)
    net=online_set_requires_grad(net)
    imgpathlist,gtary=paths_gts(seqname)
    bbs = list(map(boundingbox, gtary))  # bbs contains the whole video's groundtruth bounding box
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    branch_id=options.K
    # bounding box regressor train in the first frame
    bbr=boundingbox_regressor_train(net,imgpathlist[0],bbs[0])
    # branch trained with shared layers being locked
    frame_nb=0
    img=cv2.imread(imgpathlist[frame_nb])
    loss,pos_regions,neg_regions=mdnet_online_one_frame_train(net,branch_id,optimizer,img,bbs[frame_nb],
                              options.frame1_pos_sps,options.frame1_neg_sps)# both regions are Variables
    pos_regions_all,neg_regions_all=pos_regions[:options.pos_update_sps],neg_regions[:options.neg_update_sps]
    cur_bb=bbs[frame_nb]
    result.append(cur_bb)
    for frame_nb in range(1, len(imgpathlist) - options.frames, options.frames):
        img = cv2.imread(imgpathlist[frame_nb])
        start = time.time()
        # net = mdnet_online_one_frame_train(net, branch_id, optimizer, imgpathlist[frame_nb], bbs[frame_nb],
        #                             options.pos_sps, options.neg_sps)
        # don't train at every step
        # estimate target bounding boxes
        sample_bbs,sample_regions,sample_scores=regions_estimate_with_SampleGenerator_in_a_frame(
            img,cur_bb,net,branch_id)
        top_scores,top_indices=sample_scores[:,0].topk(5)
        target_score=top_scores.mean()
        success=(target_score>options.success_thr)
        top_bbs=[sample_bbs[i.data[0]] for i in top_indices]
        assert len(top_bbs)>0,"top_bbs is empty"
        target_bb=boundingbox_mean(top_bbs)
        if success:
            print("successful")
            sp_generator.set_trans_f(options.next_trans_f)
            bbreg_sample_bbs=[sample_bbs[i.data[0]] for i in top_indices]
            bbreg_sample_regions=[sample_regions[i.data[0]] for i in top_indices]# sample_regions are arrays
            bbreg_sample_regions=np.array(bbreg_sample_regions)
            bbreg_sample_regions=Variable(torch.from_numpy(bbreg_sample_regions).float())
            if torch.cuda.is_available():
                bbreg_sample_regions=bbreg_sample_regions.cuda()
            bbreg_feature_tensor=mdnet_conv3_output(net,bbreg_sample_regions)
            bbreg_sample_bbs=bbr.predict(bbreg_feature_tensor,bbreg_sample_bbs)
            if len(bbreg_sample_regions)>0:
                target_bb=boundingbox_mean(bbreg_sample_bbs)
            cur_bb = target_bb
            result.append(cur_bb)
            # data collection
            pos_example_regions,neg_example_regions=extract_regions_with_SampleGenerator_in_a_frame(
                img,cur_bb,options.pos_update_sps,options.neg_update_sps
            )# both regions are Variables
            pos_example_regions,neg_example_regions=\
                Variable(torch.from_numpy(pos_example_regions).float()),\
                Variable(torch.from_numpy(neg_example_regions).float())
            if torch.cuda.is_available():
                pos_example_regions,neg_example_regions=pos_example_regions.cuda(),neg_example_regions.cuda()
            assert isinstance(pos_example_regions,Variable) and isinstance(neg_example_regions,Variable) and\
                isinstance(pos_regions_all,Variable) and isinstance(neg_regions_all,Variable),\
                "pos_example_regions is {}, neg_example_regions is {}, pos_regions_all is {}, neg_regions_all is {}".format(
                    type(pos_example_regions),type(neg_example_regions),type(pos_regions_all),type(neg_regions_all)
                )
            pos_regions_all=torch.cat([pos_regions_all,pos_example_regions],dim=0)
            neg_regions_all=torch.cat((neg_regions_all,neg_example_regions),dim=0)
            l_pos,l_neg=pos_regions_all.size()[0],neg_regions_all.size()[0]
            # if l_pos>options.n_frame_long*options.pos_update_sps:
            #     pos_regions_all=pos_regions_all[l_pos-options.n_frame_long*options.pos_update_sps:]
            # if l_neg>options.n_frame_short*options.neg_update_sps:
            #     neg_regions_all=neg_regions_all[l_neg-options.n_frame_short*options.neg_update_sps:]
            if l_pos>options.frame1_pos_sps:
                pos_regions_all=pos_regions_all[-options.frame1_pos_sps:]
            if l_neg>options.frame1_neg_sps:
                neg_regions_all=neg_regions_all[-options.frame1_neg_sps:]
        else:
            print("failed")
            cur_bb=result[-1]
            result.append(target_bb)
            sp_generator.set_trans_f(options.trans_f_expand)
            # short-term update updates
            mdnet_online_one_frame_train(net, branch_id, optimizer, img, cur_bb,
                                        pos_regions=pos_regions_all, neg_regions=neg_regions_all)
        # result
        # long-term update updates fc4, fc5 and the branch using
        if frame_nb%options.long_term_interval==0:

            mdnet_online_one_frame_train(net,branch_id,optimizer,img,cur_bb,
                                        pos_regions=pos_regions_all,neg_regions=neg_regions_all)

        fps=1.0/(time.time()-start)
        tracking_result(seqname[8:],bbs[frame_nb],cur_bb,img,frame_nb+1,fps)
def main0():
    """
    1)some bounding boxes are partly outside of the image,
    like ../vot2016/birds2/00000366.jpg whose size is(wid=960,hei=540),
    with bounding box [648 456 754 570].And in vot2016/butterfly,
    the butterfly is sometimes outside of the image window.
    and sometimes when the butterfly is outside of the image window,
    the boundingbox is instead the image window.
    images could be cropped. So while training, image padding is necessary

    2)motion blur can be caused by targets fast motion and camara parameters modification
    3)some bounding boxes can only contain part of the "target", whose quotes mean the "target" is
    what we human beings consider as a target. For example, in ../vot2016/birds2/00000514.jpg, the legs
    are not contained. And only the upper body of the baby is contained in image ../vot2016/blanket/00000001.jpg
    through ../vot2016/blanket/00000030.jpg, while only the head of the baby is contained in image ../vot2016/blanket/00000037.jpg
    When the baby's head is partly occluded by the blanket, only part of the head is contained in ../vot2016/blanket/00000060.jpg
    When the head is totally occluded by the blanket,only the abdomen of the baby is contained in ../vot2016/blanket/00000061.jpg
    The arms of the baby are sometimes included but sometimes not
    4)Whether the bike is included in vot2016/bmx is not specified.
    Ghost can be a more accurate term of the motion blur
    5)rotation information is discarded in vot2016/book
    6)appearance changes must be handled in vot2016/book and vot2016/butterfly
    7)MDnet didn't exploit temporal information during offline training
    8)random samples may be redundant and not sufficient at the same time
    9)the foreground and the background are very similar in vot2016/fish1 and vot2016/rabbit, namely,
    protective coloration, even challenging for human eyes
    10)when very similar objects appear in the image,tracking other targets makes sense like in vot2016/fish1,
    but makes the result evaluation complex. This is also one of the reasons whether holistic detection is needed
    11)the boundingbox can be very narrow or very short, which makes int operation and then bilinear
    interpolation unreliable
    12)what is similar? have similar colors? have similar shapes?
    13)prediction is not really important. but when you want to generate artificial data,
    it would be super important
    14)crazy number of similar targets in vot2016/leaves
    15)does trajectory calculation make sense when you don't know the camera's position
    16)tracking transparent objects
    17)
    :return:
    """
    with open("vot2016.txt","r") as f:
        seqnames_with_back_slashes=f.readlines()
    seqnames=[seqname_with_back_slash[:-1] for seqname_with_back_slash in
              seqnames_with_back_slashes]
    print(seqnames[:-1])
    offline_train(100*options.K,seqnames[:-1])
    # seqnames=["vot2016/bag",
    #           "vot2016/girl",
    #           "vot2016/marching",
    #           "vot2016/book",
    #           "vot2016/ball1",
    #           "vot2016/ball2"]
    # pretrain(100*len(seqnames),seqnames)

def main1():
    offline_train(2,["vot2016/singer3"])
def main2():
    online_tracking("vot2016/marching")
if __name__ == '__main__':
    # main2()
    parser=argparse.ArgumentParser()
    parser.add_argument("part",help="the part of mdnet you run")
    args=parser.parse_args()
    if args.part=="offline0":
        offline_train(2, ["vot2016/singer3"])
    elif args.part=="offline1":
        with open("vot2016.txt", "r") as f:
            seqnames_with_back_slashes = f.readlines()
        seqnames = [seqname_with_back_slash[:-1] for seqname_with_back_slash in
                    seqnames_with_back_slashes]
        print(seqnames[:-1])
        offline_train(100 * options.K, seqnames[:-1])
    elif args.part=="online0":
        # the whole MDnet
        video=input("the video you test: ")
        seqname=os.path.join("vot2016",video)
        online_tracking(seqname)
