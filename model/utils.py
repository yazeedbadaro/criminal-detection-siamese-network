import numpy as np
import torch
import torchvision
import torch.nn.functional as F

def evaluate_pair(output1,output2,target,threshold):
    euclidean_distance = F.pairwise_distance(output1, output2)
    cond = euclidean_distance<threshold
    pos_sum = 0
    neg_sum = 0
    pos_acc = 0
    neg_acc = 0

    for i in range(len(cond)):
        if target[i]:
            neg_sum+=1
            if not cond[i]:
                neg_acc+=1
        if not target[i]:
            pos_sum+=1
            if cond[i]:
                pos_acc+=1

    return pos_acc,pos_sum,neg_acc,neg_sum


def initialize_weights(m):
    classname = m.__class__.__name__

    if (classname.find('Linear') != -1):
        m.weight.data.normal_(mean = 0, std = 0.01)
    if (classname.find('Conv') != -1):
        m.weight.data.normal_(mean = 0.5, std = 0.01)
