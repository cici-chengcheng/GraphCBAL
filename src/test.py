from src.utils.player import Player
from src.utils.query import RandomQuery
from src.utils.dataloader import GraphLoader
import random
import numpy as np
import argparse
import time
from functools import reduce
from src.utils.common import logger
from src.utils.utils import *
from src.utils.query import ProbQuery
from src.utils.policynet import *
import scipy.stats as stats
import pandas as pd


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
    parser.add_argument("--split_type",type=str,default="GPA", help="SR; ENS; GPA")
    parser.add_argument("--imbalance_ratio",type=float,default=0.3)
    parser.add_argument("--major_pernum",type=int,default=0)
    parser.add_argument("--minor_method",type=str,default="last")
    parser.add_argument('--max_pernum', type=int, default=20)
    parser.add_argument("--init_method",type=str,default="zero", help="zero; imbalance; full; When initialize:"
                                                                      "zero - select zero nodes for each class;"
                                                                      "imbalance - select `major_pernum` for major classes and fewer for minor classes;"
                                                                      "full - select nodes for full budget")
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--datasets",type=str,default="cora")  # 测试集用的数据
    parser.add_argument("--train_datasets",type=str,default="cora")  # 模型训练时用的数据
    parser.add_argument("--modelname",type=str,default="tmp")
    parser.add_argument("--remain_epoch",type=int,default=100,help="continues training $remain_epoch"
                                                                  " epochs after all the selection")
    parser.add_argument("--method",type=str,default='3',help="1 for random, 2 for age, 3 for policy, 4 for entropy, 5 for centrality")

    parser.add_argument("--experimentnum",type=int,default=50)  # 做n次独立实验 即初始化classification net n次
    parser.add_argument("--fix_test",type=int,default=1)
    parser.add_argument("--multigraphindex", type=str, default="")
    
    parser.add_argument('--use_entropy', type=int, default=1)
    parser.add_argument('--use_degree', type=int, default=0)
    parser.add_argument('--use_local_diversity', type=int, default=0)
    parser.add_argument('--use_select', type=int, default=1)
    parser.add_argument('--use_label', type=int, default=0)
    # parser.add_argument('--label_tmp_test', type=int, default=0)
    parser.add_argument('--use_diversity', type=int, default=1)
    parser.add_argument('--use_centrality', type=int, default=1)
    parser.add_argument('--use_class_diversity', type=int, default=1)
    parser.add_argument('--use_class_diff_diversity', type=int, default=0)
    parser.add_argument('--use_major_class', type=int, default=0)

    parser.add_argument('--valid_type', type=str, default='random', help='random; balance; miss')
    parser.add_argument('--fixTestAndValid', type=int, default=0)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--age_basef", type=float, default=0.95)
    parser.add_argument("--same_initialization", type=int, default=0)
    parser.add_argument("--policynet",type=str,default='gcn',help="gcn;mlp;no")
    parser.add_argument('--pg', type=str, default='a2c',help="reinforce; a2c; ppo")
    parser.add_argument('--filter', type=int, default=0)
    parser.add_argument('--filter_threshold', type=float, default=0, help='the threshold that filters out unlabeld nodes w.r.t prob of being a major class')

    parser.add_argument("--classification_type",type=str,default="GCN", help="GloGNN; GCN")
    #parser.add_argument("--cuda",type=int,default=1)

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


def ageQuery(p, args):

    from src.baselines.age import AGEQuery
    G = p.G
    q = AGEQuery(G,args.age_basef)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1,2),dim=2).detach().cpu().numpy()
        selected = q(output,pool,i)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce(log=True)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def entropyQuery(p, args):

    from src.baselines.age import EntropyQuery
    G = p.G
    q = EntropyQuery(G,args.age_basef)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1,2),dim=2).detach().cpu().numpy()
        selected = q(output,pool,i)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce(log=True)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def centralityQuery(p, args):

    from src.baselines.age import CentralityQuery
    G = p.G
    q = CentralityQuery(G,args.age_basef)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1,2),dim=2).detach().cpu().numpy()
        selected = q(output,pool,i)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=True)

    for i in range(args.remain_epoch):
        p.trainOnce(log=True)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def coresetQuery(p, args):

    from src.baselines.coreset import CoreSetQuery
    G = p.G
    notrainmask = p.testmask + p.trainmask
    row,col = torch.where(notrainmask<0.1)
    row, col = row.cpu().numpy(),col.cpu().numpy()
    trainsetid = []
    for i in range(args.batchsize):
        trainsetid.append(col[row==i].tolist())

    q = CoreSetQuery(args.batchsize,trainsetid)

    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1, 2), dim=2).detach().cpu().numpy()
        validoutput = output[row, col].reshape((args.batchsize,len(trainsetid[0]),-1))
        selected = q(validoutput)
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=False)

    for i in range(args.remain_epoch):
        p.trainOnce(log=False)

    acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test


def anrmabQuery(p, args):

    from src.baselines.anrmab import AnrmabQuery
    G = p.G
    q = AnrmabQuery(G, args.current_budget,G.stat["nnode"]-args.ntest-args.nval,args.batchsize)

    lastselect = None
    for i in range(args.current_budget):
        pool = p.getPool()
        output = torch.nn.functional.softmax(p.allnodes_output.transpose(1, 2), dim=2).detach().cpu().numpy()

        if lastselect is not None:
            lastselectedoutput = output[range(len(lastselect)), lastselect]
            lastpred = np.argmax(lastselectedoutput,axis=-1)
            truelabel = G.Y[lastselect].cpu().numpy()
            lastselectacc = (truelabel == lastpred).astype(np.float)
        else:
            lastselectacc = [0. for i in range(args.batchsize)]
        selected = q(output, lastselectacc, pool)
        lastselect = selected
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=False)

    for i in range(args.remain_epoch):
        p.trainOnce(log=False)

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
    
    if not args.filter:
        q = ProbQuery(args, "hard")
    else:
        q = ProbQuery(args, "filter_hard")

    action_index = np.zeros([args.batchsize, args.current_budget])
    for i in range(args.current_budget):

        pool = p.getPool(reduce=False)
        state = e.getState()
        logits = policy(state, G.normadj)
        if not args.filter:
            selected = q(logits, pool)
        else:
            selected = q(logits, pool, p)
        p.updateSelectedLabelNum(selected)
        action_index[:, i] = selected.detach().cpu().numpy()
        logger.debug("selected nodes {}".format(selected))
        p.query(selected)
        p.trainOnce(log=False)

    for i in range(args.remain_epoch):
        p.trainOnce(log=False)
        

    # acc_val = p.validation()
    acc_test = p.validation(test=True)
    return acc_test, p.selected_label_num


def multipleRun(p, args, times=100):
    method = {"1":randomQuery,"2":ageQuery,"3":policyQuery,"4":entropyQuery,"5":centralityQuery,"6":coresetQuery,"7":anrmabQuery}
    ave = []
    ave_selected_label_num = []


    for time in range(1, times+1):
        print("第", time, "轮独立测试")
        p.reset(fix_test=args.fix_test, split_type=args.split_type)
        acc_test, selected_label_num = method[args.method](p, args)
        ave.append(acc_test)
        ave_selected_label_num.append(selected_label_num)

        selectednum = [item for sublist in selected_label_num.tolist() for item in sublist]

        mic = acc_test[0][0]
        mac = acc_test[1][0]
        test_class_accuracy = acc_test[6][0]

        #acc_test = mic, mac, auc, f1_std, label_std, f1_scores, recalls


        columns = ['mic', 'mac']
        data_to_append = []
        data_to_append.append(mic)
        data_to_append.append(mac)
        for i in range(len(selectednum)):
            add = 'selec' + str(i)
            columns.append(add)
            data_to_append.append(selectednum[i])

        for i in range(len(test_class_accuracy)):
            add = 'accur' + str(i)
            columns.append(add)
            data_to_append.append(test_class_accuracy[i])

        filename = "result_" + args.datasets+"_"+ args.modelname+ ".xlsx"

        try:
            existing_data = pd.read_excel(filename, engine='openpyxl')
        except FileNotFoundError:
            existing_data = pd.DataFrame()

        # 将新数据追加到现有数据后面
        plus_data = pd.DataFrame([data_to_append], columns=columns)
        new_data = existing_data.append(plus_data, ignore_index=True)
        new_data.to_excel(filename, index=False)
        ''''''

        if time % 5 == 0:
            print("第", time, "轮独立测试")

    outstr = "modelname:"+ args.modelname+"\n"
    outstr += "datasets:"+args.datasets+"\n"
    outstr += "query budgets:"+ args.budgets+"\n"
    outstr += "split_type:"+ args.split_type+ " nval:"+str(args.nval) + " ntest:"+ str(args.ntest)+"\n"

    #print(args.modelname)
    ave = list(zip(*ave))  # [((mic,mic,mic,mic,mic),(mic,mic,mic,mic,mic)), ((.),(.)), ((.),(.)), ...]
    ave = [reduce(lambda x,y:x+y, i) for i in ave]  # [(mic,mic,mic,mic,mic,mic,mic,mic,mic,mic), (..), (..), ...]
    # stat_ave = [mean_std(L) for L in ave[:-1]]  # 最后一个是f1_score需要单独处理
    stat_ave = [mean_std(L) for L in ave[:2]]  # 最后2个是f1_score需要单独处理
    metric_list = ["mic", "mac"]
    for metric_name, metric_result in zip(metric_list, stat_ave):
        #print("{}: {:.5f} ± {:.5f}".format(metric_name, metric_result[0], metric_result[1]))
        outstr += "{}: {:.2f} ± {:.2f}".format(metric_name, metric_result[0] * 100, metric_result[1] * 100) + "\n"

    #final_f1_scores = np.zeros(len(ave[-2][0]))
    #for b in ave[-2]:
    #    final_f1_scores += b
    #final_f1_scores = np.round(final_f1_scores/len(ave[-2]), 4)
    # print("f1_score:", torch.tensor(final_f1_scores))
    #print("f1_score:", list(final_f1_scores))

    recalls = np.stack(ave[-1])
    rec_mean = np.mean(recalls, axis=0)
    rec_std = np.std(recalls, axis=0)
    # recall_std = np.round(np.array(recalls).std(), 5)
    # print("recall_std", torch.tensor(recall_std))
    strr = ''
    for m,s in zip(rec_mean, rec_std):
        strr = strr + str(np.round(m, 5)) + " ± " + str(np.round(s, 5)) + ", "
    #print("recalls:", strr[:-2])
    outstr += "recalls:" + strr[:-2]+ "\n"

    selected_label_num = torch.mean(torch.stack(ave_selected_label_num).squeeze(1), dim=0)
    #print("selected_num:", list(selected_label_num.numpy()))
    outstr += "selected_num:" + str(selected_label_num.numpy()) + "\n"

    selected_num = torch.stack(ave_selected_label_num).squeeze(1)

    imb_ratio_mean = torch.mean(torch.min(selected_num, dim=-1)[0] / torch.max(selected_num, dim=-1)[0], dim=-1)
    imb_ratio_std = torch.std(torch.min(selected_num, dim=-1)[0] / torch.max(selected_num, dim=-1)[0], dim=-1)

    #print("imb_ratio_mean:", np.round(imb_ratio_mean.numpy(), 5))
    #print("imb_ratio_std:", np.round(imb_ratio_std.numpy(), 5))
    outstr += ("imb_ratio_mean:" + str(np.round(imb_ratio_mean.numpy(), 2)) + "±" +
               str(np.round(imb_ratio_std.numpy() , 2)) + "\n")

    print(outstr)
    filename = args.datasets + "_method" + args.method + "_new.txt"
    with open(filename, "a") as f:
        f.write(str(outstr))
        f.write("-------------------------------------------\n")
    f.close()





def noPolicyRun(p, args, times=100):
    ave = []
    for time in range(times):
        p.reset(fix_test=args.fix_test, split_type=args.split_type)
        for _ in range(args.remain_epoch):
            p.trainOnce(log=False)
        
        acc_test = p.validation_noPolicy()
        ave.append(acc_test)
        if time % 5 == 0:
            print("第",time,"轮独立1测试")
    
    ave = list(zip(*ave))  # [((mic,mic,mic,mic,mic),(mic,mic,mic,mic,mic)), ((.),(.)), ((.),(.)), ...]
    ave = [reduce(lambda x,y:x+y, i) for i in ave]  # [(mic,mic,mic,mic,mic,mic,mic,mic,mic,mic), (..), (..), ...]
    stat_ave = [mean_std(L) for L in ave[:-1]]  # 最后一个是f1_score需要单独处理
    metric_list = ["mic", "mac", "auc", "f1_std"]
    for metric_name, metric_result in zip(metric_list, stat_ave):
        print("{}: {:.5f} ± {:.5f}".format(metric_name, metric_result[0], metric_result[1]))

    final_f1_scores = np.zeros(len(ave[-1][0]))
    for b in ave[-1]:
        final_f1_scores += b
    final_f1_scores = np.round(final_f1_scores/len(ave[-1]), 4)
    print("f1_score:", torch.Tensor(final_f1_scores))

'''
def sameIntializationTest(p,args,times=100):
    method = {"1":randomQuery,"2":ageQuery,"3":policyQuery,"4":entropyQuery,"5":centralityQuery}
    ave1, ave2, ave3 = [], [], []
    ave1macro,ave2macro,ave3macro = [],[],[]
    for time in range(times):
        p.reset(fix_test=args.fix_test)
        torch.save(p.net.state_dict(),"tmpfile_net_params.pkl")
        
        acc_test1, _ = method["1"](p, args,test_no_seq=False)
        ave1.extend(list(acc_test1[0]))
        ave1macro.extend(list(acc_test1[1]))

        p.reset(resplit=False)
        p.net.load_state_dict(torch.load("tmpfile_net_params.pkl"))
        acc_test2, _ = method["2"](p, args,test_no_seq = False)
        ave2.extend(list(acc_test2[0]))
        ave2macro.extend(list(acc_test2[1]))

        p.reset(resplit=False)
        p.net.load_state_dict(torch.load("tmpfile_net_params.pkl"))
        acc_test3, _ = method["3"](p, args,test_no_seq = False)
        ave3.extend(list(acc_test3[0]))
        ave3macro.extend(list(acc_test3[1]))
    
    ave1, ave1macro = np.array(ave1), np.array(ave1macro)
    ave2, ave2macro = np.array(ave2), np.array(ave2macro)
    ave3, ave3macro = np.array(ave3), np.array(ave3macro)

    improveby2 = ave2 - ave1
    improveby3 = ave3 - ave1
    
    ratio = improveby3 - improveby2
    
    averatio,stdratio = mean_std(ratio)
    ttestresult = stats.ttest_rel(improveby3,improveby2)
    print("mean difference between AGE and ours {}, std {}".format(averatio,stdratio))
    print("paired ttest p value {}".format(ttestresult.pvalue))    
'''


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
    if args.policynet != 'no':
        budgets = [int(x) for x in args.budgets.split('+')]
    else:
        if args.init_method != "full":
            raise ValueError(" policynet is 'no', but init_method is not 'full' ")
    
    if (datasets[0] == "reddit1401"):
        multigraphindex = args.multigraphindex.split("+")
        for i in range(len(multigraphindex)):
            print (datasets[0], multigraphindex[i])
            args.current_budget = budgets[i]
            print('budgets: %d' % (args.current_budget))
            g = GraphLoader(datasets[0], sparse=True, multigraphindex=multigraphindex[i], args=args)
            g.process()
            p = Player(g, args)
            if args.policynet == "no":
                pass
            else:
                multipleRun(p, args, times=args.experimentnum)
    else:
        for i in range(len(datasets)):
            print(datasets[i])
            if args.policynet != 'no':
                args.current_budget = budgets[i]
                print('budgets: %d' % (args.current_budget))
            else:
                print('budgets is not needed')
            g = GraphLoader(datasets[i], sparse=True, multigraphindex=args.multigraphindex, args=args)
            g.process()
            p = Player(g, args)
            if args.policynet == "no":
                noPolicyRun(p, args, times=args.experimentnum)
            else:
                multipleRun(p, args, times=args.experimentnum)
    
    # python -m src.test --method 3 --modelname pretrain_reddit1+2 --datasets reddit1401+reddit1401+reddit1401 --multigraphindex graph3+graph4+graph5 --budgets 50+50+50
    
    # python -m src.test --method 3 --policynet gcn --pg reinforce --use_label 0 --alpha 1.0 --remain_epoch 150 --experimentnum 20 --datasets cora --init_method zero --budgets 140 --modelname gcn_cora_GPA_TD_best9782
