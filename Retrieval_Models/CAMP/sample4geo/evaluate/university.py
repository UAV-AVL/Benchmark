import torch
import numpy as np
from tqdm import tqdm
import time
import gc
from ..trainer import predict

def find_bad_res(query_name, gallery_list, tolerance=10):
    if_bad = True

    query_label = query_name.split("/")[-2]
    for i in range(len(gallery_list)):
        res = gallery_list[i].split("/")[-2]
        if res == query_label and i < tolerance:
            if_bad = False
            break

    return if_bad

def find_mediocre_res(query_name, gallery_list, best_threshold=1, worst_threshold=5):
    if_mediocre = False
    query_label = query_name.split("/")[-2]
    for i in range(len(gallery_list)):
        res = gallery_list[i].split("/")[-2]
        if res == query_label and i == best_threshold-1:
            if_mediocre = False
            return if_mediocre
        elif res == query_label and i >= best_threshold-1 and i <= worst_threshold-1:
            if_mediocre = True
            return if_mediocre
        elif i > worst_threshold-1:
            if_mediocre = False
            break

    return if_mediocre



def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True):
    
    
    print("Extract Features:")
    img_features_query, ids_query, path_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery, path_gallery = predict(config, model, gallery_loader)
    
    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    with open("./mediocre_results_xpw.txt", 'w') as f:

        for i in tqdm(range(len(ids_query))):
            ap_tmp, CMC_tmp, index_rank = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)

            # -- write list
            query_name = path_query[i]
            ref_names = []
            for j in range(10):
                ref_names.append(path_gallery[index_rank[j]])
            # if find_bad_res(query_name, ref_names, tolerance=10):
            #     f.write(f"{query_name} {ref_names}\n")

            if find_mediocre_res(query_name, ref_names, best_threshold=1, worst_threshold=5):
                f.write(f"{query_name} {ref_names}\n")


            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
    
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_gallery)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))             
        
    print(' - '.join(string)) 
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        #torch.cuda.empty_cache()
    
    return CMC[0]


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp + (index, )


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap, cmc

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




