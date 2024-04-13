from src.utils.player import Player
from src.utils.dataloader import GraphLoader
import random
import numpy as np
import argparse
from functools import reduce
from src.utils.utils import *
from src.utils.policynet import *
from src.utils.query import *



switcher = {'gcn':PolicyNetGCN, 'mlp':PolicyNetMLP}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid",type=int,default=64)
    parser.add_argument("--pnhid",type=str,default='8+8')
    parser.add_argument("--pnout",type=int,default=1)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--lr",type=float,default=1e-2)
    parser.add_argument("--batchsize",type=int,default=1,help="here batchsize means the number of "
                                                               "repeated (independence) experiment of testing")
    parser.add_argument("--budgets",type=str, help="budget for label query")
    #parser.add_argument("--split_type",type=str,default="random")
    parser.add_argument('--max_pernum', type=int, default=20)
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--datasets",type=str,default="cora")
    parser.add_argument("--train_datasets",type=str,default="cora")
    parser.add_argument("--modelname",type=str,default="tmp")
    parser.add_argument("--remain_epoch",type=int,default=100,help="continues training $remain_epoch"
                                                                  " epochs after all the selection")
    parser.add_argument("--method",type=str,default='2',help="1 for random, 2 policy")

    parser.add_argument("--experimentnum",type=int,default=50)
    parser.add_argument("--fix_test",type=int,default=1)
    parser.add_argument("--multigraphindex", type=str, default="")
    
    parser.add_argument('--use_entropy', type=int, default=1)
    parser.add_argument('--use_select', type=int, default=1)
    parser.add_argument('--use_diversity', type=int, default=1)
    parser.add_argument('--use_centrality', type=int, default=1)
    parser.add_argument('--use_class_diversity', type=int, default=1)
    parser.add_argument('--use_major_class', type=int, default=0)


    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--policynet",type=str,default='gcn',help="gcn;mlp;no")
    parser.add_argument('--pg', type=str, default='a2c',help="reinforce; a2c; ppo")

    args = parser.parse_args()
    args.pnhid = [int(n) for n in args.pnhid.split('+')]

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args


def randomQuery(p, args):

    q = RandomQuery()
    G = p.G
    for i in range(args.current_budget):
        pool = p.getPool()
        selected = q(pool)
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce()

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def policyQuery(p, args, policy=None):
    from src.utils.env import Env
    from src.utils.query import ProbQuery
    G = p.G
    e = Env([p],args)
    if policy is None:
        policy = switcher[args.policynet](args, e.statedim, "actor").cuda()
        if args.pg == 'a2c':
            policy.load_state_dict(torch.load("saved_models/{}/{}.pkl".format(args.train_datasets, args.modelname))["actor_state_dict"])
        else:
            policy.load_state_dict(torch.load("saved_models/{}/{}.pkl".format(args.train_datasets, args.modelname)))
        policy.eval()

    q = ProbQuery(args, "hard")


    action_index = np.zeros([args.batchsize, args.current_budget])
    for i in range(args.current_budget):

        pool = p.getPool(reduce=False)
        state = e.getState()
        logits = policy(state, G.normadj)
        selected = q(logits, pool)

        p.updateSelectedLabelNum(selected)
        action_index[:, i] = selected.detach().cpu().numpy()
        p.query(selected)
        p.trainOnce(log=False)

    for i in range(args.remain_epoch):
        p.trainOnce(log=False)

    acc_test = p.validation(test=True)
    return acc_test, p.selected_label_num


def multipleRun(p, args, times=100):
    method = {"1":randomQuery,"2":policyQuery}
    ave = []
    ave_selected_label_num = []


    for time in range(1, times+1):
        p.reset(fix_test=args.fix_test)
        acc_test, selected_label_num = method[args.method](p, args)
        ave.append(acc_test)
        ave_selected_label_num.append(selected_label_num)

        if time % 5 == 0:
            print("第", time, "轮独立测试")

    outstr = "modelname:"+ args.modelname+"\n"
    outstr += "datasets:"+args.datasets+"\n"
    outstr += "query budgets:"+ args.budgets+"\n"
    outstr += " nval:"+str(args.nval) + " ntest:"+ str(args.ntest)+"\n"

    ave = list(zip(*ave))
    ave = [reduce(lambda x,y:x+y, i) for i in ave]
    stat_ave = [mean_std(L) for L in ave[:2]]
    metric_list = ["mic", "mac"]
    for metric_name, metric_result in zip(metric_list, stat_ave):
        outstr += "{}: {:.2f} ± {:.2f}".format(metric_name, metric_result[0] * 100, metric_result[1] * 100) + "\n"

    recalls = np.stack(ave[-1])
    rec_mean = np.mean(recalls, axis=0)
    rec_std = np.std(recalls, axis=0)
    strr = ''
    for m,s in zip(rec_mean, rec_std):
        strr = strr + str(np.round(m, 5)) + " ± " + str(np.round(s, 5)) + ", "
    outstr += "recalls:" + strr[:-2]+ "\n"

    selected_label_num = torch.mean(torch.stack(ave_selected_label_num).squeeze(1), dim=0)
    outstr += "selected_num:" + str(selected_label_num.numpy()) + "\n"

    selected_num = torch.stack(ave_selected_label_num).squeeze(1)

    imb_ratio_mean = torch.mean(torch.min(selected_num, dim=-1)[0] / torch.max(selected_num, dim=-1)[0], dim=-1)
    imb_ratio_std = torch.std(torch.min(selected_num, dim=-1)[0] / torch.max(selected_num, dim=-1)[0], dim=-1)

    outstr += ("imb_ratio_mean:" + str(np.round(imb_ratio_mean.numpy(), 2)) + "±" +
               str(np.round(imb_ratio_std.numpy() , 2)) + "\n")

    print(outstr)
    filename = args.datasets + "_method" + args.method + "_new.txt"
    with open(filename, "a") as f:
        f.write(str(outstr))
        f.write("-------------------------------------------\n")
    f.close()




if __name__=="__main__":

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    datasets = args.datasets.split('+')
    budgets = [int(x) for x in args.budgets.split('+')]

    for i in range(len(datasets)):
        print(datasets[i])
        args.current_budget = budgets[i]
        print('budgets: %d' % (args.current_budget))
        g = GraphLoader(datasets[i], sparse=True, multigraphindex=args.multigraphindex, args=args)
        g.process()
        p = Player(g, args)
        multipleRun(p, args, times=args.experimentnum)


    
