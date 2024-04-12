# individual player who takes the action and evaluates the effect
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from src.utils.common import *
from src.utils.classificationnet import GCN,MLP_NORM
from src.utils.utils import *
from src.utils.data_utils import load_fixed_splits

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
            if self.args.classification_type == "GCN":
                self.net = GCN(self.G.stat['nfeat'],args.nhid,self.G.stat['nclass'],args.batchsize,args.dropout,True).cuda()
                self.loss_func=F.nll_loss
            else:
                self.net = MLP_NORM(nnodes=self.G.stat['nnode'],nfeat=self.G.stat['nfeat'],
                                    nhid=args.nhid,nclass=self.G.stat['nclass'],
                                    dropout=args.dropout,alpha=0,beta=1,gamma=0,delta=0,
                                    norm_func_id=1,norm_layers=2,orders=1,orders_func_id=2,cuda=self.args.gpu)
                self.loss_func = F.nll_loss
                device = torch.device("cuda:"+str(self.args.gpu))
                self.net = self.net.to(device)

        
        self.fulllabel = self.G.Y.expand([self.batchsize]+list(self.G.Y.size()))  # 原本Y的形状是(节点数,)，扩展后通过复制变为(batchsize,节点数)
        self.reset(fix_test=fixed_test, split_type=args.split_type)  # initialize
        self.count = 0


    def makeValTestMask(self, fix_test=True):

        # 将验证和测试的mask全部初始化为形状(batchsize,节点数)的0矩阵
        trainmask = torch.zeros((self.batchsize, self.G.stat['nnode'])).to(torch.float).cuda()
        valmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()
        testmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()
        #uselessmask = torch.zeros((self.batchsize,self.G.stat['nnode'])).to(torch.float).cuda()

        valid = []
        testid = []
        vallabel = []
        testlabel = []
        self.selected_label_num = torch.zeros((self.batchsize, self.G.stat["nclass"]))

        for i in range(self.batchsize):

            if self.args.fixTestAndValid:
                if self.G.stat['name'] == "reddit":
                    #prefix = os.path.join(root, name, name)
                    path = "./data/reddit/"
                    testid_ = pkl.load(open(path + "reddit.test.pkl", 'rb'))
                    valid_ = pkl.load(open(path + "reddit.val.pkl", 'rb'))
                else:
                    split_idx_lst = load_fixed_splits(self.G.stat["name"], '')
                    testid_ = split_idx_lst[0]['test'].tolist()
                    valid_ = split_idx_lst[0]['valid'].tolist()

                #base = split_idx_lst[0]['train'].tolist()
                #uselessid_ = torch.where(self.G.Y[base] == -1)[0].tolist()
                #uselessmask[i, uselessid_] = 1.

            else:

                base = np.array([x for x in range(self.G.stat["nnode"])])
                if fix_test:
                    testid_=[x for x in range(self.G.stat["nnode"]-self.args.ntest, self.G.stat["nnode"])]  # 每次都取末尾的ntest个样本用于测试
                else:
                    testid_ = np.sort(np.random.choice(base, size=self.args.ntest, replace=False)).tolist()  # 每次都随机(不重复)选择ntest个样本用于测试


                s = set(testid_)
                base = np.array([x for x in range(self.G.stat["nnode"]) if x not in s ])  # 排除训练集和测试集样本id

                if self.args.valid_type == "random":
                    valid_ = np.sort(np.random.choice(base, size=self.args.nval, replace=False)).tolist()  # 每次都随机(不重复)选择nval个样本用于验证
                else:
                    if self.args.valid_type == "balance":
                        validsize = [71, 71, 71, 71, 72, 72, 72]
                    elif self.args.valid_type == "miss":
                        validsize = [83, 83, 83, 83, 84, 84, 0]
                    valid_ = []
                    for classid in range(self.G.stat["nclass"]):
                        class_idxs = torch.where(self.G.Y[base] == classid)[0].tolist()
                        select_valid_class = np.sort(np.random.choice(class_idxs, size=validsize[classid], replace=False)).tolist()
                        valid_.extend(select_valid_class)

            testmask[i, testid_] = 1.
            testid.append(testid_)
            testlabel.append(self.G.Y[testid_])

            #print("len(valid_):",len(valid_))
            valmask[i,valid_]=1.
            valid.append(valid_)
            vallabel.append(self.G.Y[valid_])




        self.trainid = torch.tensor([[]] * self.batchsize).cuda()
        self.valid = torch.tensor(valid).cuda()
        self.testid = torch.tensor(testid).cuda()
        self.vallabel = torch.stack(vallabel).cuda()  # vallabel由batchsize个列表构成，每个列表中的元素是torch格式的标签
        self.testlabel = torch.stack(testlabel).cuda()
        self.trainmask = trainmask
        self.valmask = valmask
        self.testmask = testmask
        #self.uselessmask = uselessmask

    def makeImbalancedTrainValTestMask(self,  fix_test= True):
        # 将训练、验证、测试的mask全部初始化为形状(batchsize,节点数)的0张量
        trainmask = torch.zeros((self.batchsize, self.G.stat['nnode'])).to(torch.float).cuda()
        valmask = torch.zeros((self.batchsize, self.G.stat['nnode'])).to(torch.float).cuda()
        testmask = torch.zeros((self.batchsize, self.G.stat['nnode'])).to(torch.float).cuda()

        if self.G.stat["name"] == "cora":
            imbalance_class_num = 3
        elif self.G.stat["name"] == "citeseer":
            imbalance_class_num = 3
        elif self.G.stat["name"] == "pubmed":
            imbalance_class_num = 1
        elif self.G.stat["name"] == "coauthor_cs":
            imbalance_class_num = 7
        elif self.G.stat["name"] == "coauthor_phy":
            imbalance_class_num = 2
        elif self.G.stat["name"] == "reddit":
            imbalance_class_num = 5

        major_init_num = self.args.major_pernum
        minor_init_num = int(major_init_num * self.args.imbalance_ratio)
        selected_label_num = torch.zeros((self.batchsize, self.G.stat["nclass"]))

        #确定各batchsize的trainid
        for i in range(self.G.stat["nclass"]):
            class_idxs = torch.where(self.G.Y == i)[0].tolist()  # 筛出属于当前标签i的节点id
            #print("class:",i," class_idxs_len:",len(class_idxs))
            for batch in range(self.batchsize):
                selecte_nodes = torch.where(trainmask[batch, :] == 1.)[0].tolist()
                #print("selecte_nodes_len:",len(selecte_nodes))
                unselected_nodes = list(set(class_idxs) - set(selecte_nodes))
                #print("unselecte_nodes_len:",len(unselected_nodes))
                if not i > self.G.stat["nclass"] - 1 - imbalance_class_num:
                    sample_idxs = np.sort(np.random.choice(unselected_nodes, size=major_init_num, replace=False)).tolist()
                else:
                    sample_idxs = np.sort(np.random.choice(unselected_nodes, size=minor_init_num, replace=False)).tolist()
                trainmask[batch, sample_idxs] = 1.
                selected_label_num[batch,i] = len(sample_idxs)
                #print("selected_label_num len:", len(sample_idxs))
        # 确定各batchsize的vailid和testid
        trainid = []
        valid = []
        testid = []
        vallabel = []
        testlabel = []
        trainlabel = []
        for i in range(self.batchsize):
            base = np.array([x for x in range(self.G.stat["nnode"])])
            selected_trained_nodes = torch.where(trainmask[batch, :] == 1.)[0].tolist()
            #print("batch:",i,"  train_idx len:",len(selected_trained_nodes))
            trainid.append(selected_trained_nodes)
            trainlabel.append(self.G.Y[selected_trained_nodes])
            unselected_nodes = list(set(base) - set(selected_trained_nodes))
            if fix_test:
                testid_ = [x for x in range(self.G.stat["nnode"] - self.args.ntest, self.G.stat["nnode"])]
            else:
                testid_ = np.sort(np.random.choice(unselected_nodes, size=self.args.ntest, replace=False)).tolist()

            unselected_nodes = list((set(unselected_nodes) - set(testid_)))
            valid_ = np.sort(np.random.choice(unselected_nodes, size=self.args.nval, replace=False)).tolist()

            testmask[i, testid_] = 1.
            testid.append(testid_)
            testlabel.append(self.G.Y[testid_])
            valmask[i, valid_] = 1.
            valid.append(valid_)
            vallabel.append(self.G.Y[valid_])

        self.trainid = torch.tensor(trainid).cuda()
        self.valid = torch.tensor(valid).cuda()
        self.testid = torch.tensor(testid).cuda()
        self.vallabel = torch.stack(vallabel).cuda()
        self.testlabel = torch.stack(testlabel).cuda()
        self.trainmask = trainmask
        self.valmask = valmask
        self.testmask = testmask
        self.selected_label_num = selected_label_num
        # print("selected training node numbers when initialized", selected_label_num)



    def lossWeighting(self,epoch):
        return min(epoch,10.)/10.


    def query(self, nodes):
        self.trainmask[[x for x in range(self.batchsize)], nodes] = 1.  # 每个batch的对应node变为1
        self.trainid = torch.cat([self.trainid, nodes.unsqueeze(-1)], dim=-1)  # 追加每个batch用于训练的节点id
        # print(self.trainid)
        # print(torch.gather(self.fulllabel, dim=-1, index=self.trainid))
        

    def getPool(self, reduce=True):
        mask = self.testmask + self.valmask + self.trainmask
        # print(torch.sum(mask,dim=-1,keepdim=True))
        # 返回的是为零的元素"坐标" 表示图中剩余无标签的节点id
        row, col = torch.where(mask<0.1)  # row中最大元素是batchsize col中最大元素是节点数 长度均为batchsize*节点数
        if reduce:
            row, col = row.cpu().numpy(), col.cpu().numpy()
            pool = []
            for i in range(self.batchsize):
                pool.append(col[row==i])
            return pool
        else:
            return row, col
    

    def updateSelectedLabelNum(self, action):
        #----------------------更新前------------------------#
        # print("更新前\n", self.selected_label_num)
        class_num = len(self.selected_label_num[0])

        best_candidate_stds = []
        for i in range(self.batchsize):
            candidate_stds = []
            for j in range(class_num):
                candidate = self.selected_label_num[i].clone()
                candidate[j] += 1
                candidate_stds.append(candidate.std())
            # print(candidate_stds)
            best_candidate_stds.append(min(candidate_stds))
        # print("each batch best std", best_candidate_stds)

        selected_label = self.G.Y[action]
        punish_mask = []
        for i in range(self.batchsize):
            old_selected_num = self.selected_label_num[i, selected_label[i]]
            if old_selected_num + 1 > self.args.max_pernum:
                punish_mask.append(1)
            else:
                punish_mask.append(0)
        # punish_mask = torch.tensor(punish_mask)

        #----------------------更新后------------------------#

        for i in range(self.batchsize):
            self.selected_label_num[i, selected_label[i]] += 1
        # print("更新后\n", self.selected_label_num)
        
        num_regret = []
        for i in range(self.batchsize):
            # print("each batch std", self.selected_label_num[i].std())
            num_regret.append(best_candidate_stds[i] - self.selected_label_num[i].std())
        # num_regret = torch.tensor(num_regret)
        # print("each batch num_regret", num_regret)

        return num_regret, punish_mask


    def trainOnce(self, log=False):
        nlabeled = torch.sum(self.trainmask)/self.batchsize  # 每个batch已经选择的节点个数
        if nlabeled == 0:
            raise ValueError("暂未选择节点 损失函数将无效!!")
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
        #if log:
            #logger.info("nnodes selected:{},loss:{}".format(nlabeled,loss.detach().cpu().numpy()))
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
        if rerun:  # 用于评估classification net
            self.net.eval()
            output, x_embed = self.net(self.G.X, self.G.normadj)  # (batchsize, 类别数, 节点数)
        else:
            output = self.allnodes_output
            x_embed = self.x_embed

        # 【计算验证/测试集损失】
        # if self.G.stat["multilabel"]:
        #     # logger.info("output of classification {}".format(output))
        #     output_trans = output.transpose(1,2)
        #     losses_val = self.loss_func(output_trans,self.fulllabel,reduction="none").mean(dim=2)
        # else:
        #     losses_val = self.loss_func(output, self.fulllabel, reduction="none")  # (batchsize, 节点数)
        # loss_val = torch.sum(losses_val*mask, dim=1, keepdim=True)/torch.sum(mask, dim=1, keepdim=True)

        # 【计算指标】
        acc = []
        trainmask_label = torch.gather(self.fulllabel, dim=-1, index=self.trainid.long())  # 已选节点的标签
        
        for i in range(self.batchsize):
            pred_val = (output[i][:,index[i]]).transpose(0,1)  # 选出当前batch的验证集/测试集输出  (节点数, 类别数)
            mic, mac, auc, f1_std, f1_scores, recalls = statMetrics(pred_val, labels[i], self.G.stat['nclass'])
            label_std = 0.
            if trainmask_label.shape[1] > 0: 
                label_std = np.bincount(trainmask_label[i].detach().cpu().numpy(), minlength=pred_val.shape[1]).std()

            # if self.args.reward_rectify == 1:  # 1-类别中数量最少与最多之比 作为权重
            #    trainmask_label_num = torch.bincount(trainmask_label[i]).detach().cpu().numpy()
            #     wgt = 1 - min(trainmask_label_num) / max(trainmask_label_num)

            # elif self.args.reward_rectify == 2:  # 多数类少数类平均节点数差异 作为权重
            #     trainmask_label_num = torch.bincount(trainmask_label[i]).detach().cpu().numpy()
            #     # print(trainmask_label_num)
            #     major_idx = self.major_label_idx
            #     minor_idx = self.minor_label_idx
            #     label_diversity = trainmask_label_num[major_idx].mean() - trainmask_label_num[minor_idx].mean()
            #     wgt = np.abs(label_diversity) / self.min_label_diversity  # 归一化
            # else:
            #     wgt = 0
            # acc.append((mic, mac, auc, f1_std, label_std, f1_scores))
            acc.append((mic, mac, auc, f1_std, label_std, f1_scores, recalls))
        return list(zip(*acc))  # [(mic,mac,auc,f1_std,label_std,f1_scores),...,(mic,mac,auc,f1_std,label_std,f1_scores)] -> [(mic,...,mic),(mac,...,mac)......]


    def validation_noPolicy(self):
        labels= self.testlabel
        index = self.testid
        self.net.eval()
        output, x_embed = self.net(self.G.X, self.G.normadj)  # (batchsize, 类别数, 节点数)

        acc = []
        for i in range(self.batchsize):
            pred_val = (output[i][:,index[i]]).transpose(0,1)  # 选出当前batch的测试集输出  (节点数, 类别数)
            mic, mac, auc, f1_std, f1_scores = statMetrics(pred_val, labels[i])
            acc.append((mic, mac, auc, f1_std, f1_scores))
        return list(zip(*acc))  # [(mic,mac,auc,f1_std,f1_scores),...,(mic,mac,auc,f1_std,f1_scores)] -> [(mic,...,mic),(mac,...,mac)......]


    def trainRemain(self):
        for _ in range(self.args.remain_epoch):
            self.trainOnce()


    def reset(self, resplit=True, fix_test=True, split_type="SR"):
        if split_type == "GPA":
            if resplit:
                self.makeValTestMask(fix_test=fix_test)
        elif split_type == "ENS":
            self.makeImbalancedTrainValTestMask(fix_test=fix_test)
        
        self.net.reset()
        self.opt = torch.optim.Adam(self.net.parameters(),lr=self.args.lr,weight_decay=5e-4)
        allnodes_output, x_embed = self.net(self.G.X, self.G.normadj)  # (batchsize, 类别数, 节点数)
        self.allnodes_output = allnodes_output.detach()
        self.x_embed = x_embed.detach()


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--budget", type=int, default=20, help="budget per class")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--remain_epoch", type=int, default=35, help="continues training $remain_epoch")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    from src.utils.dataloader import GraphLoader
    args = parse_args()
    G = GraphLoader("cora")
    G.process()
    p = Player(G,args)
    p.query([2,3])
    p.query([4,6])

    p.trainOnce()

    print(p.trainmask[:,:10])
    print(p.allnodes_output[0].size())