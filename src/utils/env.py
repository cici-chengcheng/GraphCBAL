import  torch.multiprocessing as mp
import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np

from src.utils.player import Player
from src.utils.utils import *


def logprob2Prob(logprobs, multilabel=False):
    if multilabel:
        probs = torch.sigmoid(logprobs)
    else:
        probs = F.softmax(logprobs, dim=2)
    return probs

def normalizeEntropy(entro,classnum):
    maxentro = np.log(float(classnum))
    entro = entro/maxentro  
    return entro

def prob2Logprob(probs,multilabel=False):
    if multilabel:
        raise NotImplementedError("multilabel for prob2Logprob is not implemented")
    else:
        logprobs = torch.log(probs)
    return logprobs

def perc(input):
    # the biger valueis the biger result is
    numnode = input.size(-2)
    res = torch.argsort(torch.argsort(input, dim=-2), dim=-2) / float(numnode)
    return res

def degprocess(deg):
    # deg = torch.log(1+deg)
    #return deg/20.
    return torch.clamp_max(deg / 20., 1.)

def localdiversity(probs, adj, deg):
    indices = adj.coalesce().indices()
    N = adj.size()[0]
    edgeprobs = probs[:,indices.transpose(0,1),:]
    headprobs = edgeprobs[:,:,0,:]
    tailprobs = edgeprobs[:,:,1,:]
    kl_ht = (torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*tailprobs,dim=-1) - \
             torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*tailprobs,dim=-1)).transpose(0,1)
    kl_th = (torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*headprobs,dim=-1) - \
             torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*headprobs,dim=-1)).transpose(0,1)

    sparse_output_kl_ht = torch.sparse.FloatTensor(indices,kl_ht,size=torch.Size([N,N,kl_ht.size(-1)]))
    sparse_output_kl_th = torch.sparse.FloatTensor(indices,kl_th,size=torch.Size([N,N,kl_th.size(-1)]))

    sum_kl_ht = torch.sparse.sum(sparse_output_kl_ht,dim=1).to_dense().transpose(0,1)
    sum_kl_th = torch.sparse.sum(sparse_output_kl_th,dim=1).to_dense().transpose(0,1)

    mean_kl_ht = sum_kl_ht/(deg+1e-10)
    mean_kl_th = sum_kl_th/(deg+1e-10)
    # normalize
    mean_kl_ht = mean_kl_ht / mean_kl_ht.max(dim=1, keepdim=True).values
    mean_kl_th = mean_kl_th / mean_kl_th.max(dim=1, keepdim=True).values
    return mean_kl_ht,mean_kl_th

def diversity(state, trainmask):
    batchsize, featurenum = state.shape[0], state.shape[2]
    selected_node_features = torch.masked_select(state, trainmask.unsqueeze(-1).expand(state.size()).bool()).reshape(batchsize,-1,featurenum)
    dist = torch.cdist(state, selected_node_features)
    if dist.shape[-1] > 0:
        min_dist, _ = torch.min(dist, dim=-1, keepdim=True)
        zeros = torch.zeros_like(min_dist)
        low_mask = min_dist < 5e-4
        min_dist = torch.where(low_mask, zeros, min_dist)
    else:
        min_dist = torch.zeros([dist.shape[0], dist.shape[1], 1]).cuda()
    return min_dist

def pagerank_centrality(G, batchsize):
    pr = nx.pagerank(G.G, alpha=0.85)
    pr = torch.tensor(list(pr.values())).cuda()
    return pr.expand([batchsize]+list(pr.size()))

def class_diversity(fulllabel, trainmask, selected_label_num, output):

    base = torch.where(selected_label_num == 0., torch.tensor(1).float(), 1/selected_label_num.float()).cuda()  # 若该类别尚未选择过节点 设为1
    selected_class_idx = (fulllabel * trainmask).long()
    cd_selected = torch.gather(base, dim=1, index=selected_class_idx)

    output = torch.exp(output).transpose(1,2)
    base_broadcast = base.unsqueeze(1).expand_as(output)
    cd_not_selected = torch.sum(output * base_broadcast, dim=-1)

    cd = torch.where(trainmask>0.1, cd_selected, cd_not_selected)
    return cd

def class_diff_diversity(fulllabel, trainmask, selected_label_num, output):

    # 考虑所有节点的多样性
    if torch.sum(selected_label_num) > 0:
        selected_class_idx = (fulllabel * trainmask).long().unsqueeze(dim=-1)  # (batchsize, 节点数, 1)
        selected_label_num = selected_label_num.unsqueeze(dim=1)  # (batchsize, 1, 类别数)
        selected_label_num = selected_label_num.expand([-1, selected_class_idx.shape[1], -1]).cuda()  # (batchsize, 节点数, 类别数)
        selected_base = torch.gather(selected_label_num, dim=2, index=selected_class_idx).squeeze(dim=-1)
        cdd_slected = 1 - selected_base / torch.sum(selected_label_num, dim=-1)

        not_selected_base = 1 - selected_label_num / torch.sum(selected_label_num, dim=-1, keepdim=True)  # (batchsize, 节点数, 类别数)
        output = torch.exp(output).transpose(1,2)  # (batchsize, 节点数, 类别数)
        cdd_not_selected = torch.sum(not_selected_base * output, dim=-1)

        cdd = torch.where(trainmask > 0.1, cdd_slected, cdd_not_selected)
    else:
        cdd = torch.ones_like(trainmask).float().cuda()
    return cdd

def major_class(fulllabel, trainmask, selected_label_num, maxpernum, output):
    major_judge = torch.where(selected_label_num <= maxpernum, torch.tensor(0).float(), torch.tensor(1).float()).cuda()  # 小于等于选择上限 则为小类别
    mc_selected = torch.gather(major_judge, dim=1, index=fulllabel)

    output = torch.exp(output).transpose(1,2)  # (batchsize, 节点数, 类别数)
    major_judge_broadcast = major_judge.unsqueeze(1).expand_as(output)  # (batchsize, 节点数, 类别数)
    mc_not_selected = torch.sum(output * major_judge_broadcast, dim=-1)  # 未选择的节点按softmax在各类别上的概率，依据是否为大类别乘以1或0，再在类别上求和
    # mc = torch.where(trainmask>0.1, all_mc, torch.tensor(0).float().cuda())
    mc = torch.where(trainmask>0.1, mc_selected, mc_not_selected)
    return mc


class Env(object):
    ## an environment for multiple players testing the policy at the same time
    def __init__(self, players, args):
        '''
        players: a list containing main player (many task) (or only one task
        '''
        self.players = players
        self.args = args
        self.nplayer = len(self.players)
        self.graphs = [p.G for p in self.players]
        featdim =-1
        if args.use_centrality:
            self.prs = []
            for playersid in range(len(self.players)):
                self.prs.append(pagerank_centrality(self.graphs[playersid], self.args.batchsize))

        self.statedim = self.getState(0).size(featdim)


    def step(self, actions, playerid=0):
        p = self.players[playerid]
        p.query(actions)
        p.trainOnce()
        reward = p.validation(test=False, rerun=False)
        return reward


    def getState(self, playerid=0):
        p = self.players[playerid]
        output = logprob2Prob(p.allnodes_output.transpose(1,2), multilabel=p.G.stat["multilabel"])
        state = self.makeState(output, p.trainmask, p.G.deg, p.G, playerid)
        return state


    def reset(self,playerid=0):
        self.players[playerid].reset(fix_test=self.args.fix_test)

    
    def makeState(self, probs, trainmask, deg, G, playerid, multilabel=False):
        entro = entropy(probs, multilabel=multilabel)
        entro = normalizeEntropy(entro, probs.size(-1))
        deg = degprocess(deg.expand([probs.size(0)]+list(deg.size())))
        p = self.players[playerid]

        features = []
        if self.args.use_entropy:
            features.append(entro)
        if self.args.use_select:
            features.append(trainmask)
        if self.args.use_centrality:
            features.append(self.prs[playerid])
        if self.args.use_class_diversity:
            cd = class_diversity(p.fulllabel, p.trainmask, p.selected_label_num, p.allnodes_output)
            features.append(cd)
        if self.args.use_major_class:
            mc = major_class(p.fulllabel, p.trainmask, p.selected_label_num, self.args.max_pernum, p.allnodes_output)
            features.append(mc)

        state = torch.stack(features, dim=-1)  # 将各个特征张量(batchsize,节点数)组成的列表,新创建最后一维,拼接为(batchsize, 节点数, 特征数)


        if self.args.use_diversity:
            min_dist = diversity(state, trainmask)
            state = torch.cat([state, min_dist], dim=-1)

        return state