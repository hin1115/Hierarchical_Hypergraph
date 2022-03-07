from scipy import sparse
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

def sps2torchsparse(H):
    # sparse.csr -> torch.sparse_coo_matrix
    H = sparse.coo_matrix(H)
    values = H.data
    indices = np.vstack((H.row, H.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = H.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def add_high_order(H, set_to_zero = True):
    # H : torch.sparse_coo_tensor -> H||H@(H^T@H) : torch.sparse_coo_tensor
    H = H.float()
    high_order = torch.sparse.mm(H.t(), H)
    # if set_to_zero:
    #     high_order = torch.sparse_coo_tensor(high_order._indices(), torch.ones_like(high_order._values()), high_order.shape)
    high_order = torch.sparse.mm(H, high_order)
    row = torch.cat((H._indices()[0], high_order._indices()[0]))
    col = torch.cat((H._indices()[1], high_order._indices()[1] + H.shape[1]))
    value = torch.cat((H._values(), high_order._values()))
    if set_to_zero:
        value = torch.ones_like(value)
    # value = torch.cat((H._values(), high_order._values()))
    shape = (H.shape[0], H.shape[1] + high_order.shape[1])
    H = torch.sparse_coo_tensor(torch.stack((row, col)), value, shape)
    return H

def Generate_G_from_H(H): # sparse matrix version
    # H는 torch.sparse.FloatTensor 타입
    H = H.float()
    n_node = H.shape[0]
    n_edge = H.shape[1]
    HT = H.t()

    W = torch.ones(n_edge)
    ind = torch.Tensor(np.arange(n_edge))
    i = torch.stack((ind, ind))
    W = torch.sparse_coo_tensor(i, W, torch.Size([n_edge,n_edge]))

    DV = torch.sparse.sum(H, dim=1).float()
    DE = torch.sparse.sum(H, dim=0).float()

    ind = DE.indices().squeeze()
    i = torch.stack((ind, ind))
    v = torch.float_power(DE.values(), -1)
    invDE = torch.sparse_coo_tensor(i, v, torch.Size([n_edge,n_edge])).float()

    ind = DV.indices().squeeze()
    i = torch.stack((ind, ind))
    v = torch.float_power(DV.values(), -0.5)
    DV2 = torch.sparse_coo_tensor(i, v, torch.Size([n_node,n_node])).float()

    G = torch.sparse.mm(DV2,H)
    G = torch.sparse.mm(G,W)
    G = torch.sparse.mm(G,invDE)
    G = torch.sparse.mm(G,HT)
    G = torch.sparse.mm(G,DV2)
    # DV2H = torch.sparse.mm(DV2,H)
    # invDEHT = torch.sparse.mm(invDE,HT)
    # invDEHTDV2 = torch.sparse.mm(invDEHT,DV2)

    return G
    # return DV2H, W, invDEHTDV2

def Generate_norm_Adj(R, category_data, category_rate = 1):
    A_shape = (R.shape[0] + R.shape[1], R.shape[0] + R.shape[1])
    R = torch.sparse_coo_tensor(torch.stack((R._indices()[0], R._indices()[1] + R.shape[0])), R._values(), size = (R.shape[0], R.shape[0]+R.shape[1]))
    ii = torch.concat((R._indices(), R.t()._indices()), dim = 1)
    vv = torch.concat((R._values(), R.t()._values()))
    A = torch.sparse_coo_tensor(ii, vv, size = A_shape)

    if torch.sum(category_data._values()) != 0:
        itembyitem = torch.sparse.mm(category_data, category_data.t())
        # i = torch.sparse.sum(category_data, dim = 1)._indices()[0]
        # ii = torch.stack((i, i))
        # itembyitem -= torch.sparse_coo_tensor(ii, torch.sparse.sum(category_data, dim = 1)._values())
        DE = torch.sparse.sum(itembyitem, dim = 1)
        DE2 = torch.float_power(itembyitem, -0.5)
        DE2 = torch.sparse_coo_tensor(torch.stack((DE2._indices()[0], DE2._indices()[0])), DE2._values(), size = (category_data.shape[0], category_data.shape[0])).float()
        norm_diag = torch.sparse.mm(DE2, itembyitem)
        norm_diag = torch.sparse.mm(norm_diag, DE2)

        A += torch.sparse_coo_tensor(torch.stack( (norm_diag._indices()[0] + R.shape[0], norm_diag._indices()[1] + R.shape[0])), norm_diag._values() * category_rate, size = A_shape)

    DE = torch.sparse.sum(A, dim = 1)
    DE2 = torch.float_power(DE, -0.5)
    DE2 = torch.sparse_coo_tensor(torch.stack((DE2._indices()[0], DE2._indices()[0])), DE2._values(), size = A_shape).float()
    norm_Adj = torch.sparse.mm(DE2, A)
    norm_Adj = torch.sparse.mm(norm_Adj, DE2)
   
    return norm_Adj

def split_per_one_user(sparse_matrix, test_size_ = 0.9, use_ranking = False):
    # 랭킹 정보를 사용하지 않고, 전부 1로 바꿔서 사용할 경우 : use_ranking = False로 사용
    if use_ranking == False:
        sparse_matrix.data = np.ones_like(sparse_matrix.data)    
    
    # 각 유저마다 indices, data를 저장. one_users.shape = (num_user, 2, num_item_per_one_user)
    one_users = []
    start_index = 0
    for idx in sparse_matrix.indptr[1:]:
        one_users.append([sparse_matrix.indices[start_index : idx], sparse_matrix.data[start_index : idx]])
        start_index = idx

    train_row, train_col, train_data = [], [], []
    test_row, test_col, test_data = [], [], []
    
    # 한명씩 1:9로 나눠서 train, test에 저장
    for idx, i in enumerate(one_users):
        user_num = idx
        index_per_user = i[0]
        ranking_per_user = i[1]
        index_train, index_test, ranking_train, ranking_test = train_test_split(index_per_user, ranking_per_user, test_size = test_size_, shuffle = True) 

        train_row += [user_num] * len(index_train)
        train_col += index_train.tolist()
        train_data += ranking_train.tolist()

        test_row += [user_num] * len(index_test)
        test_col += index_test.tolist()
        test_data += ranking_test.tolist()

    H_u_train = sparse.csr_matrix((train_data, (train_row, train_col)), shape = (sparse_matrix.shape[0], sparse_matrix.shape[1]))
    H_u_test = sparse.csr_matrix((test_data, (test_row, test_col)), shape = (sparse_matrix.shape[0], sparse_matrix.shape[1]))

    return H_u_train, H_u_test

def data_generator(csr_matrix, batch_size = 64): 
    matrix_shape = csr_matrix.shape
    n_batch = matrix_shape[0]//batch_size + 1
    all_items = [i for i in range(matrix_shape[1])]
    random_index = np.random.choice(matrix_shape[0], matrix_shape[0], replace=False)

    for i in range(n_batch):
        users = []
        pos_items = []
        neg_items = []
        if (i+1) != n_batch:
            one_batch_ind = random_index[i * batch_size : (i+1) * batch_size]
        else:
            one_batch_ind = random_index[i * batch_size : ]
        for j in one_batch_ind:
            users.append(j)
            pos_items_per_user = csr_matrix[j].indices
            neg_items_per_user = list(set(all_items) - set(pos_items_per_user))

            pos_items += list(np.random.choice(pos_items_per_user, 1, replace = False))
            neg_items += list(np.random.choice(neg_items_per_user, 1, replace = False))

        yield users, pos_items, neg_items

class mymetric():
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(1000)]

    def _ndcg(self, gt, rec, k = 20):
        rec = np.argsort(-rec,axis=1)
        ndcgs = []
        for j in range(gt.shape[0]):
            dcg = 0.0
            ground_true = gt[j].indices
            for i, r in enumerate(rec[j][:k]):
                if r in ground_true:
                    dcg += 1.0 / np.log(i + 2)
            if len(ground_true)>0:
                ndcgs.append(dcg / self._idcgs[min(len(ground_true), k)])
            else:
                ndcgs.append(1.)
        return np.array(ndcgs)

    def _recall(self, gt, rec, k=20):
        # rec = all_user_emb @ all_item_emb, 학습 데이터에 있는 아이템은 제외한 상태
        # gt = sparse_all_dataset
        rec = np.argsort(-rec,axis=1)
        recalls = []
        for j in range(gt.shape[0]):
            recall = 0.0
            ground_true = gt[j].indices
            # kk = len()
            for i, r in enumerate(rec[j][:k]):
                if r in ground_true:
                    recall += 1.0 / len(ground_true)
            if len(ground_true)>0:
                recalls.append(recall)
            else:
                recalls.append(1.)
        return np.array(recalls)

    def _precision(self, gt, rec, k=20):
        rec = np.argsort(-rec,axis=1)
        precisions = []
        for j in range(gt.shape[0]):
            precision = 0.0
            ground_true = gt[j].indices
            for i, r in enumerate(rec[j][:k]):
                if r in ground_true:
                    precision += 1.0 / k
            if len(ground_true)>0:
                precisions.append(precision)
            else:
                precisions.append(1.)
        return np.array(precisions)

    def _hit_ratio(self, gt, rec, k=20): # not corrected
        if ((gt==0).sum()+(gt==1).sum())==(gt.shape[0]*gt.shape[1]):
            gt = [np.concatenate(np.argwhere(t==1)) if len(np.argwhere(t==1))>0 else [] for t in gt]
        else:
            gt = np.argsort(-gt,axis=1)
        rec = np.argsort(-rec,axis=1)
        hit_ratios = []
        for j in range(len(gt)):
            hit = 0.0
            kk = min(len(rec[j]),k)
            for i, r in enumerate(rec[j][:kk]):
                if r in gt[j]:
                    hit = 1.0
                    break
            if len(gt[j])>0:
                hit_ratios.append(hit)
            else:
                hit_ratios.append(1.)
        return np.array(hit_ratios)

def bpr_loss(users, pos_items, neg_items, decay = 0.00001):
    pos_scores = torch.sum(torch.mul(users, pos_items), axis = 1)
    neg_scores = torch.sum(torch.mul(users, neg_items), axis = 1)
    maxi = nn.LogSigmoid()(pos_scores - neg_scores)

    mf_loss = -1 * torch.mean(maxi)

    regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2  + torch.norm(neg_items) ** 2) / 2
    # emb_loss = (decay * regularizer) / users.shape[0]
    emb_loss = (decay * regularizer)

    return mf_loss + emb_loss, mf_loss, emb_loss

def first_dropout(H, dropout_rate = 0.1):
    indices = np.random.choice(H._values().shape[0], int(H._values().shape[0] * (1 - dropout_rate)), replace = False)
    ii = torch.stack((H._indices()[0][indices], H._indices()[1][indices]))
    return torch.sparse_coo_tensor(ii, H._values()[indices], H.shape)