
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.classificationnet import GCN
from src.utils.utils import *

class Player(nn.Module):

    def __init__(self, G, args, fixed_test=True, rank=0):

        super(Player,self).__init__()
        self.G = G
        self.args = args
        self.rank = rank
        self.batchsize = args.batchsize

        # classification net
        if self.G.stat['multilabel']:
            self.net = GCN(self.G.stat['nfeat'],args.nhid,self.G.stat['nclass'],args.batchsize,args.dropout,False,bias=True).cuda()
            self.loss_func=F.binary_cross_entropy_with_logits
        else:
            self.net = GCN(self.G.stat['nfeat'], args.nhid, self.G.stat['nclass'], args.batchsize, args.dropout, True).cuda()
            self.loss_func = F.nll_loss

        
        self.fulllabel = self.G.Y.expand([self.batchsize]+list(self.G.Y.size()))
        self.reset(fix_test=fixed_test)
        self.count = 0


    def makeValTestMask(self, fix_test=True):

        trainmask = torch.zeros((self.batchsize, self.G.stat['nnode'])).to(torch.float).cuda()
        valmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()
        testmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()

        valid = []
        testid = []
        vallabel = []
        testlabel = []
        self.selected_label_num = torch.zeros((self.batchsize, self.G.stat["nclass"]))

        for i in range(self.batchsize):
            base = np.array([x for x in range(self.G.stat["nnode"])])
            if fix_test:
                testid_ = [x for x in range(self.G.stat["nnode"] - self.args.ntest, self.G.stat["nnode"])]
            else:
                testid_ = np.sort(np.random.choice(base, size=self.args.ntest, replace=False)).tolist()

            s = set(testid_)
            base = np.array([x for x in range(self.G.stat["nnode"]) if x not in s])

            valid_ = np.sort(np.random.choice(base, size=self.args.nval, replace=False)).tolist()

            testmask[i, testid_] = 1.
            testid.append(testid_)
            testlabel.append(self.G.Y[testid_])
            valmask[i,valid_]=1.
            valid.append(valid_)
            vallabel.append(self.G.Y[valid_])

        self.trainid = torch.tensor([[]] * self.batchsize).cuda()
        self.valid = torch.tensor(valid).cuda()
        self.testid = torch.tensor(testid).cuda()
        self.vallabel = torch.stack(vallabel).cuda()
        self.testlabel = torch.stack(testlabel).cuda()
        self.trainmask = trainmask
        self.valmask = valmask
        self.testmask = testmask


    def lossWeighting(self,epoch):
        return min(epoch,10.)/10.


    def query(self, nodes):
        self.trainmask[[x for x in range(self.batchsize)], nodes] = 1.
        self.trainid = torch.cat([self.trainid, nodes.unsqueeze(-1)], dim=-1)
        

    def getPool(self, reduce=True):
        mask = self.testmask + self.valmask + self.trainmask
        row, col = torch.where(mask<0.1)
        if reduce:
            row, col = row.cpu().numpy(), col.cpu().numpy()
            pool = []
            for i in range(self.batchsize):
                pool.append(col[row==i])
            return pool
        else:
            return row, col
    

    def updateSelectedLabelNum(self, action):

        class_num = len(self.selected_label_num[0])

        best_candidate_stds = []
        for i in range(self.batchsize):
            candidate_stds = []
            for j in range(class_num):
                candidate = self.selected_label_num[i].clone()
                candidate[j] += 1
                candidate_stds.append(candidate.std())
            best_candidate_stds.append(min(candidate_stds))

        selected_label = self.G.Y[action]
        punish_mask = []
        for i in range(self.batchsize):
            old_selected_num = self.selected_label_num[i, selected_label[i]]
            if old_selected_num + 1 > self.args.max_pernum:
                punish_mask.append(1)
            else:
                punish_mask.append(0)

        for i in range(self.batchsize):
            self.selected_label_num[i, selected_label[i]] += 1

        return  punish_mask


    def trainOnce(self, log=False):
        nlabeled = torch.sum(self.trainmask)/self.batchsize
        if nlabeled == 0:
            raise ValueError("If the node loss function is not selected temporarily, it will be invalid!!")
        self.net.train()
        self.opt.zero_grad()
        output, x_embed = self.net(self.G.X, self.G.normadj)

        if self.G.stat["multilabel"]:
            output_trans = output.transpose(1,2)
            losses = self.loss_func(output_trans,self.fulllabel,reduction="none").sum(dim=2)
        else:
            losses = self.loss_func(output, self.fulllabel, reduction="none")
        loss = torch.sum(losses*self.trainmask)/nlabeled * self.lossWeighting(float(nlabeled.cpu()))  # 只考虑已选择的节点计算损失
        loss.backward()
        self.opt.step()
        self.allnodes_output = output.detach()
        self.x_embed = x_embed.detach()
        return output


    def validation(self, test=False, rerun=True):
        if test:
            mask = self.testmask
            labels= self.testlabel
            index = self.testid
        else:
            mask = self.valmask
            labels = self.vallabel
            index = self.valid
        if rerun:
            self.net.eval()
            output, x_embed = self.net(self.G.X, self.G.normadj)
        else:
            output = self.allnodes_output
            x_embed = self.x_embed

        acc = []
        trainmask_label = torch.gather(self.fulllabel, dim=-1, index=self.trainid.long())
        
        for i in range(self.batchsize):
            pred_val = (output[i][:,index[i]]).transpose(0,1)
            mic, mac, auc, f1_std, f1_scores, recalls = statMetrics(pred_val, labels[i], self.G.stat['nclass'])
            label_std = 0.
            if trainmask_label.shape[1] > 0: 
                label_std = np.bincount(trainmask_label[i].detach().cpu().numpy(), minlength=pred_val.shape[1]).std()

            acc.append((mic, mac, auc, f1_std, label_std, f1_scores, recalls))
        return list(zip(*acc))

    def trainRemain(self):
        for _ in range(self.args.remain_epoch):
            self.trainOnce()


    def reset(self, resplit=True, fix_test=True):
        if resplit:
            self.makeValTestMask(fix_test=fix_test)
        
        self.net.reset()
        self.opt = torch.optim.Adam(self.net.parameters(),lr=self.args.lr,weight_decay=5e-4)
        allnodes_output, x_embed = self.net(self.G.X, self.G.normadj)
        self.allnodes_output = allnodes_output.detach()
        self.x_embed = x_embed.detach()


