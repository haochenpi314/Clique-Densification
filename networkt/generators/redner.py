from os import replace
from sys import int_info
from numpy.lib.function_base import append, percentile
from pandas.core import api
import networkt
import sys
import os
import random


def redner_graph(p, n, target_m=None):
    p = float(p)
    n = int(n)
    edges = []
    adjs = {}

    adjs[0] = []
    for i in range(1, n):
        target = random.randint(0, i - 1)

        adjs[i] = [target, ]
        edges.append([i, target])
        for adj in adjs[target]:
            if random.random() < p:
                adjs[i].append(adj)
                adjs[adj].append(i)
                edges.append([i, adj])

        adjs[target].append(i)
        if target_m > 0 and len(edges) > target_m:
            break

    return edges

# from binomial to poisson binomial
def reweight(prob, weights, threshold=1.):
    import numpy as np
    total_prob = len(weights) * prob
    total_weight = sum(weights)
    probs = weights * (total_prob / total_weight)
    
    while np.any(probs > threshold):
        overflow = np.sum(probs[probs > threshold] - threshold)
        # make those higher vals equal to threshold
        probs[probs > threshold] = threshold
        # assign the extra to the rest
        # probs[probs < threshold] += overflow / len(probs[probs < threshold])
        probs[probs < threshold] += probs[probs < threshold] * \
            overflow / np.sum(probs[probs < threshold])

    return probs


def padme_graph(p, n, target_m=0):
    import numpy as np
    p = float(p)
    n = int(n)
    edges = []
    adjs = {}
    adjs[0] = []
    for i in range(1, n):
        target = random.randint(0, i - 1)
        adjs[i] = [target, ]
        if len(edges) > 0 and len(adjs[target]) > 0:
            weights = np.array([(len(adjs[adj])) for adj in adjs[target]])
            probs = reweight(p, weights)
            for j, node in enumerate(adjs[target]):
                if probs[j] < random.random():
                    continue
                adjs[node].append(i)
                adjs[i].append(node)
                edges.append([i, node])
                # edges.append([node, i])
        adjs[target].append(i)
        edges.append([i, target])
        if target_m > 0 and len(edges) > target_m:
            break

    return edges


def threshold_padme_graph(p, r, n, target_m=None):
    import numpy as np
    p = float(p)
    r = float(r)
    n = int(n)
    edges = []
    adjs = {}
    adjs[0] = []
    for i in range(1, n):
        target = random.randint(0, i - 1)
        adjs[i] = [target, ]
        if len(edges) > 0 and len(adjs[target]) > 0:
            weights = np.array([(len(adjs[adj])) for adj in adjs[target]])
            probs = reweight(p, weights, p + (1 - p) * r)
            for j, node in enumerate(adjs[target]):
                if probs[j] < random.random():
                    continue
                adjs[node].append(i)
                adjs[i].append(node)
                edges.append([i, node])
                # edges.append([node, i])
        adjs[target].append(i)
        edges.append([i, target])

        if target_m > 0 and len(edges) > target_m:
            break

    return edges

