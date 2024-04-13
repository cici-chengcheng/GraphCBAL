import numpy as np
import scipy.sparse as sp
import warnings
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support, accuracy_score
import torch
import math



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features

def column_normalize(tens):
    ret = tens - tens.mean(axis=0)
    return ret

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_add_diag=adj + sp.eye(adj.shape[0])  # \hat{A} = A + I
    adj_normalized = normalize_adj(adj_add_diag)
    return adj_normalized.astype(np.float32) #sp.coo_matrix(adj_unnorm)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

##=========================================================================

def statMetrics(y_pred, label, classnum):
    if len(label.size())==1:
        y_pred_prob = torch.exp(y_pred).cpu().detach().numpy()
        y_pred = y_pred.max(1)[1].type_as(label)
        y_pred = y_pred.cpu().detach().numpy()
        label = label.cpu().numpy()
    elif len(label.size())==2:
        y_pred=(y_pred > 0.).cpu().detach().numpy()
        label=label.cpu().numpy()

    _, recalls, _, _ = precision_recall_fscore_support(label, y_pred, labels=np.arange(classnum), zero_division=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mic, mac = f1_score(label, y_pred, average="micro"), f1_score(label, y_pred, average="macro")
        auc = 0 #roc_auc_score(label, y_pred_prob, average="macro", multi_class='ovr')
        f1_scores = f1_score(label, y_pred, average=None)
        f1_std = np.std(f1_scores)

    return mic, mac, auc, f1_std, f1_scores, recalls


def mean_std(L):
    if type(L)==np.ndarray:
        L=L.tolist()
    elif type(L)==torch.Tensor:
        L=L.cpu().detach().numpy().tolist()
    m=sum(L)/float(len(L))
    bias=[(x-m)**2 for x in L]
    std=math.sqrt(sum(bias)/float(len(L)-1))
    return [float(m), float(std)]

##==========================================================================

def entropy(tens, multilabel=False):
    if multilabel:
        reverse=1-tens
        ent_1= -torch.log(torch.clamp(tens, min=1e-7)) * tens
        ent_2= -torch.log(torch.clamp(reverse,min=1e-7))*reverse
        ent=ent_1+ent_2
        entropy=torch.mean(ent,dim=1)
    else:
        assert type(tens)==torch.Tensor and len(tens.size())==3,"calculating entropy of wrong size"
        entropy = - torch.log(torch.clamp(tens, min=1e-7)) * tens
        entropy = torch.sum(entropy, dim=2)
    return entropy


##==========================================================================


class AverageMeter(object):
    def __init__(self, name='', ave_step=10, S=10, criterion="greater"):
        self.name = name
        self.ave_step = ave_step
        self.history = []
        self.history_extrem = None
        self.S = S
        self.criterion = criterion

    def update(self, data):
        if data is not None:
            self.history.append(data)

    def __call__(self):
        if len(self.history) == 0:
            value =  None
        else:
            cal = self.history[-self.ave_step:]
            value = sum(cal)/float(len(cal))
        return value

    def should_save(self):
        if self.criterion == "greater":
            if len(self.history) > self.S*2 and sum(self.history[-self.S:])/float(self.S) > sum(self.history[-self.S*2:])/float(self.S*2):
                if self.history_extrem is None :
                    self.history_extrem = sum(self.history[-self.S:])/float(self.S)
                    return False
                else:
                    if self.history_extrem < sum(self.history[-self.S:])/float(self.S):
                        self.history_extrem = sum(self.history[-self.S:])/float(self.S)
                        return True
                    else:
                        return False
            else:
                return False
        else:  # less
            if len(self.history) > self.S*2 and sum(self.history[-self.S:])/float(self.S) < sum(self.history[-self.S*2:])/float(self.S*2):
                if self.history_extrem is None :
                    self.history_extrem = sum(self.history[-self.S:])/float(self.S)
                    return False
                else:
                    if self.history_extrem > sum(self.history[-self.S:])/float(self.S):
                        self.history_extrem = sum(self.history[-self.S:])/float(self.S)
                        return True
                    else:
                        return False
            else:
                return False


#===========================================================

def inspect_grad(model):
    name_grad = [(x[0], x[1].grad) for x in model.named_parameters() if x[1].grad is not None]
    name, grad = zip(*name_grad)
    assert not len(grad) == 0, "no layer requires grad"
    mean_grad = [torch.mean(x) for x in grad]
    max_grad = [torch.max(x) for x in grad]
    min_grad = [torch.min(x) for x in grad]
    logger.info("name {}, mean_max min {}".format(name,list(zip(mean_grad, max_grad, min_grad))))

def inspect_weight(model):
    name_weight = [x[1] for x in model.named_parameters() if x[1].grad is not None]
    print("network_weight:{}".format(name_weight))


#==============================================================

def common_rate(counts,prediction,seq):
    summation = counts.sum(dim=1, keepdim=True)
    squaresum = (counts ** 2).sum(dim=1, keepdim=True)
    ret = (summation ** 2 - squaresum) / (summation * (summation - 1)+1)

    equal_rate=counts[seq,prediction].reshape(-1,1)/(summation+1)

    return ret,equal_rate

