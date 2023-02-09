import os
import pickle
import numpy as np
import torch
import math
# from tqdm import tqdm

# ========================================================
#   Usefull paths
_datasetFeaturesFiles = {
                         "cub_RN18": "./checkpoints/cub/RN18/cub.plk",
                         }
_cacheDir = "./"
_maxRuns = 10000
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None


def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None

def convert_prob_to_samples(prob, q_shot):
    prob = prob * q_shot
    for i in range(len(prob)):
        if sum(np.round(prob[i])) > q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.floor(prob[i, idx])
            prob[i] = np.round(prob[i])
        elif sum(np.round(prob[i])) < q_shot:
            while sum(np.round(prob[i])) != q_shot:
                idx = 0
                for j in range(len(prob[i])):
                    frac, whole = math.modf(prob[i, j])
                    if j == 0:
                        frac_clos = abs(frac - 0.5)
                    else:
                        if abs(frac - 0.5) < frac_clos:
                            idx = j
                            frac_clos = abs(frac - 0.5)
                prob[i, idx] = np.ceil(prob[i, idx])
            prob[i] = np.round(prob[i])
        else:
            prob[i] = np.round(prob[i])
    return prob.astype(int)


def get_dirichlet_query_dist(alpha, n_tasks, n_ways, q_shots):
    alpha = np.full(n_ways, alpha)
    prob_dist = np.random.dirichlet(alpha, n_tasks)
    return convert_prob_to_samples(prob=prob_dist, q_shot=q_shots)


def loadDataSet(dsname):
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labels, _randStates, _rsCfg, _min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    dataset = _load_pickle(_datasetFeaturesFiles[dsname])

    # Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))
    
def GenerateRun(iRun, cfg, regenRState=False):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    indices = np.arange(_min_examples)
    
    support = []
    support_label = []
    query = []
    query_label = []
    n_feat = data.shape[-1]
    n_samples_per_cls = data.shape[1]
    
    if cfg['balanced'] is True:
        for i in range(cfg['ways']):
            shuffle_indices = np.random.permutation(indices)
            samples = data[classes[i], shuffle_indices, :][:cfg['shot']+cfg['queries']]
            support.append(samples[:cfg['shot']])
            support_label += [i] * cfg['shot']
            query.append(samples[cfg['shot']:])
            query_label += [i] * cfg['queries']
    else:
        alpha = 2
        num_query_samples = get_dirichlet_query_dist(alpha, 1, cfg['ways'], cfg['ways'] * cfg['queries'])[0]
        while not ((cfg['shot']+num_query_samples)<n_samples_per_cls).all():
            num_query_samples = get_dirichlet_query_dist(alpha, 1, cfg['ways'], cfg['ways'] * cfg['queries'])[0]
            
        for i in range(cfg['ways']):
            shuffle_indices = np.random.permutation(indices)
            samples = data[classes[i], shuffle_indices, :][:cfg['shot']+num_query_samples[i]]
            support.append(samples[:cfg['shot']])
            support_label += [i] * cfg['shot']
            query.append(samples[cfg['shot']:])
            query_label += [i] * num_query_samples[i]
            
    support = torch.cat(support).view(cfg['ways'], cfg['shot'], -1).permute(1, 0, 2).reshape(-1, n_feat)
    query = torch.cat(query)
    shuffle_ind = torch.randperm(query.shape[0])
    query = query[shuffle_ind]
    query_label = torch.tensor(query_label)[shuffle_ind].tolist()
    
    dataset = torch.cat([support, query], dim=0)
    label = support_label + query_label
    label = torch.tensor(label)
    label[:cfg['shot']*cfg['ways']] = label[:cfg['shot']*cfg['ways']].view(cfg['ways'], cfg['shot']).permute(1,0).reshape(-1)
    
    return dataset, label


def ClassesInRun(iRun, n_ways):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:n_ways]
    return classes


def setRandomStates(cfg):
    global _randStates, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}_r{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways'], cfg['runs']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        #np.random.seed(0)
        _randStates = []
        for iRun in range(cfg['runs']):
            np.random.seed(iRun)
            _randStates.append(np.random.get_state())
            #GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(cfg=None):
    global dataset, label, _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15, "runs":_maxRuns}
        
    start = 0
    end = cfg['runs']

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))
    
    dataset = torch.zeros((end-start, cfg['ways'] * (cfg['shot']+cfg['queries']), data.shape[2]))
    label = torch.zeros((end-start, cfg['ways'] * (cfg['shot']+cfg['queries']))).type(torch.int64)
    
    for iRun in range(end-start):
        dataset[iRun], label[iRun] = GenerateRun(iRun, cfg)

    return dataset, label


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15, "runs": 10}
    setRandomStates(cfg)

    run10, label = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])
    print(ds.size())
