import time
import gc
import numpy as np
import sklearn.metrics.pairwise
import torch
import os
import cv2
def assign_by_euclidian_at_k_sop(X, T, recall_list, start, cal_batch=0):
    k = max(recall_list)
    if cal_batch == 0:
        gap = len(X[start:])
    else:
        gap = cal_batch
    distances = sklearn.metrics.pairwise.pairwise_distances(X[start:start+gap],X)
    indices = np.argsort(distances, axis=1)[:, 1: k + 1]
    Y = torch.from_numpy(np.array([[T[i] for i in ii] for ii in indices]))


    true_num = []
    for k in recall_list:
        num = sum([1 for t, y in zip(X[start:start+gap], Y) if t in y[:k]])
        true_num.append(num)

    return true_num