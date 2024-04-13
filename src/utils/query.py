
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


def choose(p,pool):
    return np.random.choice(pool,size=1,replace=False,p=p)


class RandomQuery(object):
    def __init__(self):
        pass

    def __call__(self,pool):
        ret = []
        for row in pool:
            p = np.ones(len(row))
            p/=p.sum()
            ret .append(choose(p,row))
        ret = np.concatenate(ret)
        return ret


class ProbQuery(object):
    def __init__(self, args, type="soft"):
        self.type = type
        self.args = args

    def __call__(self,probs,pool,p=None):
        if self.type == "soft":
            return self.softquery(probs,pool)
        elif self.type == "hard":
            return self.hardquery(probs,pool)
        elif self.type == "filter_hard":
            return self.filterHardQuery(probs, pool, p)

    def softquery(self,logits,pool):
        batchsize = logits.size(0)
        valid_logits = logits[pool].reshape(batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits  # torch.clamp(valid_logits,max = MAX_EXP)
        valid_probs = F.softmax(valid_logits, dim=1)
        pool = pool[1].reshape(batchsize, -1)
        assert pool.size() == valid_probs.size()
        m = Categorical(valid_probs)
        action_inpool = m.sample()
        action = pool[[x for x in range(batchsize)], action_inpool]
        return action

    def hardquery(self, logits, pool):
        batchsize = logits.size(0)
        valid_logits = logits[pool].reshape(batchsize,-1)  # 剩余无标签的节点被选择的概率 (batchsize, 无标签节点数)
        max_logits = torch.max(valid_logits, dim=1, keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits #torch.clamp(valid_logits,max = MAX_EXP)
        valid_probs = F.softmax(valid_logits, dim=1)
        
        pool = pool[1].reshape(batchsize, -1)
        action_inpool = torch.argmax(valid_probs, dim=1)
        action = pool[[x for x in range(batchsize)], action_inpool]  # (batchsize,)
        return action
    
    def filterHardQuery(self, logits, pool, p):
        output = torch.exp(p.allnodes_output).transpose(1,2)  # (batchsize, 节点数, 类别数)
        major_judge = torch.where(p.selected_label_num <= self.args.max_pernum, torch.tensor(0).float(), torch.tensor(1).float()).cuda()  # 小于等于选择上限 则为小类别
        major_judge_broadcast = major_judge.unsqueeze(1).expand_as(output)  # (batchsize, 节点数, 类别数)
        mc_not_selected = torch.sum(output * major_judge_broadcast, dim=-1)  # 未选择的节点按softmax在各类别上的概率，依据是否为大类别乘以1或0，再在类别上求和
        mc_mask = torch.where(mc_not_selected > self.args.filter_threshold)  # 大概率属于大类别的row和col

        logits[mc_mask] = torch.tensor(0).float().cuda()  # 将大概率属于大类别的节点logits置为0，即最小化被选中的可能性
        valid_logits = logits[pool].reshape(self.args.batchsize, -1)
        max_logits = torch.max(valid_logits, dim=1, keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)

        pool = pool[1].reshape(self.args.batchsize, -1)
        action_inpool = torch.argmax(valid_probs, dim=1)
        action = pool[[x for x in range(self.args.batchsize)], action_inpool]
        return action


def unitTestProbQuery():
    probs = F.softmax(torch.randn(4,7)*3,dim=1)
    mask = torch.zeros_like(probs)
    mask[:,1] = 1
    pool = torch.where(mask==0)
    print(probs,pool)

    q = ProbQuery(type = "soft")
    action = q(probs,pool)
    print(action)

    q.type = "hard"
    action = q(probs,pool)
    print(action)


def selectActions(self,logits,pool):
        valid_logits = logits[pool].reshape(self.args.batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        if self.globel_number %10==0:
            logger.info(max_logits)
        valid_logits = valid_logits - max_logits #torch.clamp(valid_logits,max = MAX_EXP)

        # valid_logprobs = F.log_softmax(valid_logits,dim=1)
        valid_probs = F.softmax(valid_logits, dim=1)
        pool = pool[1].reshape(self.args.batchsize,-1)
        assert pool.size()==valid_probs.size()

        m = Categorical(valid_probs)
        action_inpool = m.sample()                        
        logprob = m.log_prob(action_inpool)
        action = pool[[x for x in range(self.args.batchsize)],action_inpool]
    
        return action,logprob
if __name__=="__main__":
    unitTestProbQuery()
