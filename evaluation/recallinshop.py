import time
import gc
import numpy as np
import sklearn.metrics.pairwise
import torch
import os
import cv2
def assign_by_euclidian_at_k_inshop(X, Y, X_T, Y_T, recall_list):
    k = max(recall_list)
    distances = sklearn.metrics.pairwise.pairwise_distances(X, Y) #(cal_batch, len(Y))
    indices  = np.argsort(distances, axis = 1)[:, 0 : k ] #(cal_batch, k)
    Y = torch.from_numpy(np.array([[Y_T[i] for i in ii] for ii in indices]))

    counts = []
    for k in recall_list:
        num = sum([1 for t, y in zip(X_T, Y) if t in y[:k]])
        counts.append(num)

    return counts