from __future__ import print_function
from __future__ import division

import evaluation
import numpy as np
import torch
import logging
import loss
import json
import networks
import time
#import margin_net
import similarity
import os
# __repr__ may contain `\n`, json replaces it by `\\n` + indent
json_dumps = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def load_config(config_name = 'config.json'):
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)
    return config

def predict_batchwise(model, dataloader):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # extract batches (A becomes list of samples)
        for batch in dataloader:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    '''
                    embedding3, embedding4 = model(J)
                    J = torch.cat((embedding3,embedding4),1).cpu()
                    '''
                    embedding2, embedding3,embedding4 = model(J)
                    J = torch.cat((embedding2,embedding3,embedding4),1).cpu()
                if i == 3:
                    A[3].extend(J)
                else:
                    for j in J:
                        A[i].append(j)

    model.train()
    model.train(model_is_training)  # revert to previous training state
    list1 = [torch.stack(A[i]) for i in range(len(A)) if i != 3]
    list1.append(A[3])
    return list1

def predict_batchwise_inshop(model, dataloader):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # use tqdm when the dataset is large (SOProducts)
        is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in dataloader:#, desc='predict', disable=not is_verbose:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)

                    m3, m4 = model(J)
                    J = torch.cat((m3,m4), 1).cpu().numpy()
                    # predict model output for image
                    #J = model(J).data.cpu().numpy()
                    # take only subset of resulting embedding w.r.t dataset1
                if i == 3:
                    A[3].extend(J)
                else:
                    for j in J:
                        A[i].append(np.asarray(j))
        result = [np.stack(A[i]) for i in range(len(A)) if i!=3]
    model.train()
    model.train(model_is_training) # revert to previous training state
    return result


def evaluate(model, dataloader, recall_list=[1,2,4,8]):
    eval_time = time.time()

    # calculate embeddings with model and get targets
    X, T, indexs,image_paths = predict_batchwise(model, dataloader)

    #eval_time = time.time() - eval_time
    #logging.info('Eval time: %.2f' % eval_time)

    # get predictions by assigning nearest 8 neighbors with euclidian
    max_dist = max(recall_list)
    recall = []
    if max_dist == 8:
        Y = evaluation.assign_by_euclidian_at_k(X, T, max_dist)
        Y = torch.from_numpy(Y)
        for k in recall_list:
            r_at_k = evaluation.calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
    else:
        start = 0
        cal_batch = 4000
        true_num = [0, 0, 0, 0]
        while(len(X[start:]) >= cal_batch):
            true_num1 = evaluation.assign_by_euclidian_at_k_sop(X, T, recall_list, start, cal_batch)
            start = start + cal_batch
            for i in range(4):
                true_num[i] = true_num[i] + true_num1[i]
        true_num1 = evaluation.assign_by_euclidian_at_k(X, T, recall_list, start)
        for i in range(4):
            true_num[i] = true_num[i] + true_num1[i]
        for k, num in zip(recall_list, true_num):
            recall.append(num / (1. * len(T)))
            logging.info("R@{} : {:.3f}".format(k, 100 * num / (1. * len(T))))


    eval_time = time.time() - eval_time
    logging.info('Eval time: %.2f' % eval_time)
    return recall

def evaluate_inshop(model, dl_query, dl_gallery,
        K = [1, 10, 20, 30, 40, 50]):

    # calculate embeddings with model and get targets
    X_query, T_query, _, image_paths = predict_batchwise_inshop(
        model, dl_query)
    X_gallery, T_gallery, _, image_paths = predict_batchwise_inshop(
        model, dl_gallery)

    nb_classes = dl_query.dataset.nb_classes()
    assert nb_classes == len(set(T_query))


# when no error: out of memory.
    X_eval = torch.cat(
        [torch.from_numpy(X_query), torch.from_numpy(X_gallery)])
    D = similarity.pairwise_distance(X_eval)[:len(X_query), len(X_query):]

    Y = T_gallery[D.topk(k = max(K), dim = 1, largest = False)[1]]

    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T_query, Y, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall
''' "when the program is killed due to out of memory", run below code 
    cal_batch = 7109   #we calculate the distance in batches
    start = 0
    Num = Total = len(X_query)
    counts = [0 for i in range(len(K))]
    while( Total > cal_batch ):
        count = evaluation.assign_by_euclidian_at_k_inshop(X_query[start : start + cal_batch],X_gallery,T_query[start: start + cal_batch],T_gallery,K)
        for i in range(len(K)):
            counts[i] = counts[i] + count[i]
        start = start + cal_batch
        Total = Total - cal_batch
    else:
        count = evaluation.assign_by_euclidian_at_k_inshop(X_query[start:],X_gallery,T_query[start:],T_gallery,K)
        for i in range(len(K)):
            counts[i] = counts[i] + count[i]
    recall = []
    for k, num in zip(K, counts):
        recall.append(num/(1. * Num))
        logging.info("R@{} : {:.3f}".format(k, 100 * num/(1.* Num)))
    return recall
'''


