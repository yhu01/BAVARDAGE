import collections
import pickle
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
from torch.distributions.dirichlet import Dirichlet
from scipy.stats import norm, chi2
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import FSLTask
import vb_equi_iso as lib
import importlib
import time
import sys
last_tick = 0

def display(s, force = False):
    global last_tick
    if time.time() - last_tick > 0.5 or force:
        sys.stderr.write(s)
        last_tick = time.time()

use_gpu = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser(description= 'BAVARDAGE')
    parser.add_argument('--dataset', default='cub', help='dataset: cub')
    parser.add_argument('--model', default='RN18', help='model: RN18') 
    parser.add_argument('--shot', default=1, type=int, help='1/5')
    parser.add_argument('--run', default=10000, type=int, help='600/1000/10000')
    parser.add_argument('--way', default=5, type=int)
    parser.add_argument('--query', default=15, type=int)
    parser.add_argument('--balanced', default=False, action='store_true')
    parser.add_argument('--nIter_vb', default=5, type=int)
    parser.add_argument('--nIter_km', default=5, type=int)
    parser.add_argument('--T_vb', default=4, type=int)
    parser.add_argument('--T_km', default=10, type=int)
    parser.add_argument('--clp', default=5, type=float)
    parser.add_argument('--preprocess', default='MESV')
    return parser.parse_args()

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_trainset(dir_base_data):
    base_output = load_pickle(dir_base_data)
    base_data = []
    num_cls = []
    for i in list(base_output.keys()):
        base_data_cls = torch.tensor(np.array(base_output[i]))
        num_cls.append(base_data_cls.shape[0])                             
        base_data.append(base_data_cls)
                                     
    base_data = torch.cat(base_data, dim=0) # X_base en taille [64, 600, 640]

    return base_data, num_cls
    
def computeSharedCov(base_data, num_cls, postRescale=False):
    # compute scov over all classes
    count = 0
    covs = []
    for i in range(len(num_cls)):
        n_cls = num_cls[i]
        base_data_c = base_data[count:count+n_cls].unsqueeze(0)
        count += n_cls
        res = base_data_c - base_data_c.mean(dim=1, keepdim=True)
        cov = torch.bmm(res.permute(0,2,1), res) # [64, 640, 640]
        covs.append(cov)
    
    covs = torch.cat(covs, dim=0)
    norms = covs.norm(dim=(1,2), keepdim=True)
    scov = (covs/norms).mean(0)
    sncov = norms.mean(0)
   
    if postRescale:
        return scov*sncov
    else:
        return scov

def scaleEachUnitaryDatas(data):
    
    return data/data.norm(dim=2, keepdim=True)

class DataSet:
    data: None
    labels: None
        
    def __init__(self, data=None, labels=None, n_shots=1, n_ways=5, n_queries=15):
        self.data = data
        self.labels = labels
        self.n_shots = n_shots
        self.n_ways = n_ways
        self.n_lsamples = n_ways*n_shots
        self.n_queries = n_queries
        self.n_usamples = n_ways*n_queries
        if self.data is not None:
            self.n_runs = data.size(0)
            self.n_samples = data.size(1)
            self.n_feat = data.size(2)
            
            if self.n_samples != self.n_lsamples + self.n_usamples:
                print("Invalid settings: queries incorrect wrt size")
                self.exit()
                
    def cuda(self):
        self.data = self.data.cuda()
        self.labels = self.labels.cuda()
    
    def cpu(self):
        self.data = self.data.cpu()
        self.labels = self.labels.cpu()        
    
    def printState(self):
        print("DataSet: {}-shot, {}-ways, {}-queries, {}-runs, {}-feats".format( \
             self.n_shots, self.n_ways, self.n_queries, self.n_runs, self.n_feat))
        print("\t {}-labelled {}-unlabelled {}-tot".format( \
              self.n_lsamples, self.n_usamples, self.n_samples))
        
def getRunSet(n_shots, n_ways, n_queries, n_runs, balanced = False, preprocess='ME', dataset='mini', model='WRN'):
    cfg = {'shot':n_shots, 'ways':n_ways, 'queries':n_queries, 'balanced':balanced, 'runs':n_runs}
    
    load = dataset + '_' + model   
    FSLTask.loadDataSet(load)
    data, labels = FSLTask.GenerateRunSet(cfg=cfg)

    ds = DataSet(data, labels, n_shots=n_shots, n_ways=n_ways, n_queries=n_queries)
    save_dir = './checkpoints/{}/{}'.format(dataset, model)
    ds.dir_base_data = save_dir + '/base_{}.plk'.format(dataset)
    ds.base_data, ds.num_cls = load_trainset(ds.dir_base_data)
    
    for p in preprocess:
        if p == "M":
            print("--- preprocess: Mean subtraction")
            base_mu = ds.base_data.mean(dim=0, keepdim=True)
            ds.data = ds.data - base_mu
            ds.base_data = ds.base_data - base_mu
                
        elif p == "V":
            print("--- preprocess: Rotation using base data")
            Sw_base = computeSharedCov(ds.base_data, ds.num_cls, True)
            Sw_base += torch.eye(Sw_base.shape[-1]) * 1e-6
            
            Sw_base_U = (ds.U.permute(0,2,1).matmul(Sw_base.inverse()).matmul(ds.U)).inverse()
            e_Swb_Udata, v_Swb_Udata = torch.linalg.eigh(Sw_base_U)
            
            e_Swb_Udata, idx = e_Swb_Udata.sort(dim=1, descending=True)
            idx = idx.unsqueeze(1).expand(-1, ds.data.shape[2], -1)
            v_Swb_Udata = v_Swb_Udata.gather(dim=2, index=idx)
            
            ds.e_Swb_Udata = e_Swb_Udata
            ds.rdata_b = ds.data.matmul(v_Swb_Udata)
        
        elif p == "S":
            print("--- preprocess: Data projection on U")
            ds.U, ds.S, _ = torch.linalg.svd(ds.data.permute(0,2,1), full_matrices=False) # U:[n_runs, 640, N], S:[N], Vh = [N, N]
            ds.data = torch.bmm(ds.data, ds.U)
        
        elif p == "E":
            print("--- preprocess: Euclidean normalization")
            ds.data = scaleEachUnitaryDatas(ds.data)
            ds.base_data = ds.base_data / ds.base_data.norm(dim=1, keepdim=True)
            
        else:
            print("unknown preprocessing!!")
            pass
            
    return ds

# =========================================
#    Class to define samples mask for Centroid computations
# =========================================

class SimpleWMask:
    """ class that selects which samples to be used for centroid computatoin 
    Default implementation use probas as wmask
    """
    def __init__(self, ds):
        self.ds = ds
        
    def ncm(self, X):
        mus = X[:,:self.ds.n_lsamples]\
            .reshape(ds.n_runs, ds.n_shots, ds.n_ways, -1)\
            .mean(1)
        dist2 = (X.unsqueeze(2) - mus.unsqueeze(1)).norm(dim=3).pow(2)
        probas = F.softmax(-dist2, dim=2)
        
        return probas
    
    def soft_km(self, X, T, nIter=20):
        mus = X[:,:self.ds.n_lsamples]\
            .reshape(ds.n_runs, ds.n_shots, ds.n_ways, -1)\
            .mean(1)
        for i in range(nIter):
            dist2 = (X.unsqueeze(2) - mus.unsqueeze(1)).norm(dim=3).pow(2)
            probas = F.softmax(-dist2*T, dim=2)
            train_labels = self.ds.labels[:,:self.ds.n_lsamples]
            probas[:,:self.ds.n_lsamples] = 0
            probas[:,:self.ds.n_lsamples].scatter_(2, train_labels.unsqueeze(2), 1)
            mus = probas.permute(0,2,1).matmul(X).div(probas.sum(dim=1).unsqueeze(2))
            
        return probas
    
# =========================================
#    Optimization routines
# =========================================

class Optimizer:
    def __init__(self, ds):
        self.ds = ds
        
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = self.ds.labels.eq(olabels).float()
        acc_test = matches[:,self.ds.n_lsamples:].mean(1)  

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(acc_test.size(0))
        return m, pm
    
    def getAccuracies(self, probas):
        olabels = probas.argmax(dim=2)
        matches = self.ds.labels.eq(olabels).float()
        acc_test = matches[:,self.ds.n_lsamples:].mean(1)  

        #m = acc_test.mean().item()
        #pm = acc_test.std().item() * 1.96 / math.sqrt(acc_test.size(0))
        return acc_test

if __name__ == '__main__':
    # ---- data loading
    params = parse_args()
    n_shots = params.shot
    n_runs = params.run
    n_ways = params.way
    n_queries = params.query
    balanced = params.balanced

    # for unbalanced, support set labels: [0,1,2,3,4,0,1,2,3,4,...,0,1,2,3,4], query set labels: [0,0,0,...,0,1,1,...1,...,4,4,4,...4]
    # for balalanced, support set labels: [0,1,2,3,4,0,1,2,3,4,...,0,1,2,3,4], query set labels: [0,1,2,3,4,0,1,2,3,4,...,0,1,2,3,4]
    ds = getRunSet(n_shots=n_shots, n_ways=n_ways, n_queries=n_queries, n_runs=n_runs, balanced=balanced, preprocess=params.preprocess, dataset=params.dataset, model=params.model)
    ds.printState()
    if use_gpu:
        ds.cuda()

    wmasker = SimpleWMask(ds)
    optim = Optimizer(ds)

    probas_ncm = wmasker.ncm(ds.data)
    res_ncm = optim.getAccuracy(probas_ncm)
    print('NCM accuracy:', res_ncm)

    probas_bl = wmasker.soft_km(ds.data, T=params.T_km, nIter=params.nIter_km)
    res_bl = optim.getAccuracy(probas_bl)
    print('Soft-kmeans accuracy:', res_bl)

    bs = 1000
    if n_runs < bs:
        bs = n_runs
    
    ds.n_runs = bs
    labels = ds.labels
    accuracy = []
    for b in range(0, n_runs, bs):
        print('batch:', b)

        X = ds.data[b:b+bs]
        ds.labels = labels[b:b+bs]

        wmasker = SimpleWMask(ds)
        optim = Optimizer(ds)
        probas_init = wmasker.soft_km(X, T=params.T_km, nIter=params.nIter_km)
        
        if use_gpu:
            X_rotated = ds.rdata_b.cuda()[b:b+bs] 
            scalingValues = ds.e_Swb_Udata.cuda()[b:b+bs] 
        else:
            X_rotated = ds.rdata_b[b:b+bs] 
            scalingValues = ds.e_Swb_Udata[b:b+bs]
            
        scalingValues = scalingValues.pow(-0.5).clamp(max=params.clp)

        importlib.reload(lib)
        sigma2 = 1./params.T_vb
        vbClusterModel = lib.VB_IsoGM(ds, scalingValues, sigma2)
        vbMixtureModel = lib.VB_SymDirMM(ds, fixedModel=balanced)
        vbModel = lib.VB_EM(ds, mixtureModel=vbMixtureModel, clusterModel=vbClusterModel)
        probas_vb = vbModel.loop(X_rotated, probas_init, nIter=params.nIter_vb)

        batch_acc = optim.getAccuracies(probas_vb)
        accuracy.append(batch_acc)

    accuracy = torch.cat(accuracy)
    m = accuracy.mean().item()
    pm = accuracy.std().item() * 1.96 / math.sqrt(accuracy.size(0))
    print('VB accuracy:', m, pm)