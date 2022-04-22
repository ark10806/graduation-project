import numpy as np
from typing import *
from torch import normal
from tqdm import tqdm
from random import shuffle
from numpy.linalg import norm
from collections import Counter

def normalize(vec: np.array):
    return vec / norm(vec, axis=-1, keepdims=True)

class Node:
    def __init__(self, fname: str, info: dict):
        self.fname = fname
        self.feature = info['feature']


class Cluster:
    def __init__(self, center_pt: np.array, elmts: list):
        self.group_id = None
        self.center_pt = center_pt
        self.elmts = elmts
        self.n_elmts = 0

    def merge(self, cluster):
        self.center_pt = self.center_pt * self.n_elmts + cluster.center_pt * cluster.n_elmts
        self.center_pt = normalize(self.center_pt)
        #! elmts를 set으로 구성한 경우 remove -> amortized O(1)
        #!         list로 구성한 경우 remove -> O(N)
        self.elmts.extend(cluster.elmts)
        self.n_elmts = self.n_elmts + cluster.n_elmts
    
    def discharge(self, eta: int):
        discharged : List[Node] = []
        n_discharged = 0
        for child in self.elmts:
            if self.center_pt @ child.feature.T < eta:
                discharged.append(child)
                n_discharged += 1
                self.center_pt = self.center_pt * (self.n_elmts - n_discharged) - child.feature
                self.center_pt = normalize(self.center_pt)
        
        for child in discharged:
            self.elmts.remove(child)

        return [Cluster(child.feature, [child]) for child in discharged]


class VecScan:
    def __init__(self):
        self.epochs = epochs
        self.eta = eta
        self.minPts = minPts
        
    
    def bigbang(self):
        """ Initialize each of features into "Cluster" instance whose len(element) is 1"""
        universe = []