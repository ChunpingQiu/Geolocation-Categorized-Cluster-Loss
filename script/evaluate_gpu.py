import scipy.io
import torch
import numpy as np
#import time
import os
from utils.gpu_select import Auto_Select_GPU
from config import DefaultConfig,SaveConfig,LoadConfig
Auto_Select_GPU()
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#######################################################################
# Evaluate
def evaluate(qf,ql,gf,gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index
    #print(good_index)
    #print(index[0:10])
    junk_index = np.argwhere(gl==-1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def Cal_CMA(result):
    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    print("query:", query_feature.shape)
    print("gallery:", gallery_feature.shape)
    # print(gallery_feature[0,:])
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    # print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], gallery_feature, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print(round(len(gallery_label) * 0.01))
    print('Recall@1:%.2f  Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100, ap / len(query_label) * 100))
    print(CMC[0].item() * 100, CMC[4].item() * 100, CMC[9].item() * 100,
          CMC[round(len(gallery_label) * 0.01)].item() * 100, ap / len(query_label) * 100)
    print(len(CMC))
def write_results(full_path,CMC):
    np.savetxt(full_path, CMC, fmt='%0.8f')
    # file.close()
######################################################################
if __name__ == '__main__':

    projectname='05221926-osnet_ain_x0_75_geo-Pretrain_2views'
    # query_name='drone'
    # gallery_name='satellite'
    query_name = 'satellite'
    gallery_name = 'drone'
    basepath="/home/qcp/00E/SHS/Light-osnet/output_result"
    logdir=os.path.join(basepath,projectname,'results',query_name+'-'+gallery_name+'.mat')

    print('Evalue:',logdir)
    result = scipy.io.loadmat(logdir)

    query_feature = torch.FloatTensor(result['query_f'])
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    print("query:",query_feature.shape)
    print("gallery:",gallery_feature.shape)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],gallery_feature,gallery_label)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp


    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print(round(len(gallery_label)*0.01))
    print('Recall@1:%.2f  Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f'%(CMC[0]*100,CMC[4]*100,CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100))

    print(CMC[0].item()*100,CMC[4].item()*100,CMC[9].item()*100, CMC[round(len(gallery_label)*0.01)].item()*100, ap/len(query_label)*100)
    print(len(CMC))
    print(CMC[0:int(len(CMC)*0.1)])
    print(CMC[0:50])
    resultspath=os.path.join(basepath,projectname,'results',query_name+'-'+gallery_name+'.txt')
    # write_results(resultspath,CMC.numpy())
    f = open(resultspath, "a+")
    f.write('%s > %s '%(query_name,gallery_name))
    f.write('\nRecall@1:%.2f  Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f'%(CMC[0]*100,CMC[4]*100,CMC[9]*100, CMC[round(len(gallery_label)*0.01)]*100, ap/len(query_label)*100))
