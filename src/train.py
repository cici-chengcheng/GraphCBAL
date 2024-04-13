
import numpy as np, torch, argparse, datetime
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn
from src.utils.dataloader import GraphLoader
from src.utils.player import Player
from src.utils.env import Env
from src.utils.policynet import *
from src.utils.rewardshaper import RewardShaper
from src.utils.utils import *
import random


switcher = {'gcn':PolicyNetGCN, 'mlp':PolicyNetMLP}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--pnhid', type=str, default='8+8')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pdropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--a_rllr', type=float, default=0.001)  # 原版GPA的reinforce参数为0.01  a2c猜测从0.001开始
    parser.add_argument('--c_rllr', type=float, default=0.001)
    parser.add_argument('--schedule', type=int, default=1)
    parser.add_argument('--entcoef', type=float, default=0)

    parser.add_argument('--datasets', type=str, default='cora')
    parser.add_argument('--multigraphindex', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--budgets', type=str, default='35', help='total budget')
    parser.add_argument('--imbalance_ratio', type=float, default=0.0)
    parser.add_argument('--major_pernum', type=int, default=0)
    parser.add_argument('--max_pernum', type=int, default=5)
    parser.add_argument('--minor_method', type=str, default='last')
    parser.add_argument('--init_method', type=str, default='zero', help='zero; imbalance; full; When initialize:zero - select zero nodes for each class;imbalance - select `major_pernum` for major classes and fewer for minor classes;full - select nodes for full budget')
    parser.add_argument('--ntest', type=int, default=1000)
    parser.add_argument('--nval', type=int, default=500)
    parser.add_argument('--fix_test', type=int, default=0)
    parser.add_argument('--metric', type=str, default='macrof1')
    parser.add_argument('--maxepisode', type=int, default=4000)
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument('--remain_epoch', type=int, default=35, help='continues training $remain_epoch epochs after all the selection')
    
    parser.add_argument('--shaping', type=str, default='1234', help='reward shaping method, 0 for no shaping;1 for add future reward,i.e. R = r+R*gamma;2 for add/only use finalreward;3 for subtract baseline(value of curent state);4 for normalize')
    parser.add_argument('--frweight', type=float, default=0.01)
    parser.add_argument('--punishment', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha*f1_macro_gain + (1-alpha)*(class_diversity)')
    
    parser.add_argument('--policynet', type=str, default='gcn', help='gcn; mlp')
    parser.add_argument('--pg', type=str, default='a2c', help='reinforce; a2c; ppo')
    parser.add_argument('--policyfreq', type=str, default="7", help='-1 matches REINFORCE; positive value matches policy update freq for A2C')
    parser.add_argument('--ppo_epoch', type=int, default=5)
    parser.add_argument('--a2c_gamma', type=float, default=0.9)
    
    parser.add_argument('--use_entropy', type=int, default=1)
    parser.add_argument('--use_select', type=int, default=1)
    parser.add_argument('--use_diversity', type=int, default=1)
    parser.add_argument('--use_centrality', type=int, default=1)
    parser.add_argument('--use_class_diversity', type=int, default=1)
    parser.add_argument('--use_major_class', type=int, default=0)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--logfreq', type=int, default=10)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--savename', type=str, default='tmp')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    args.pnhid = [int(n) for n in args.pnhid.split('+')]

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args


def check_args(args):
    if (args.pg == 'reinforce' and args.a_rllr >= 1.0) or (args.pg == 'a2c' and (args.c_rllr >= 1.0 or args.a_rllr >= 1.0)):
        raise ValueError("Unmodified RL learning rate")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


class SingleTrain(object):
    def __init__(self, args):
        self.args = args
        self.datasets = self.args.datasets.split('+')
        self.budgets = [int(x) for x in self.args.budgets.split('+')]
        self.policyfreq = [int(x) for x in self.args.policyfreq.split('+')]
        self.graphs, self.players, self.rshapers, self.accmeters, self.accmeters_vice = [], [], [], [], []
        for i, dataset in enumerate(self.datasets):
            g = GraphLoader(dataset, sparse=True, args=args, multigraphindex=('graph' + str(i + 1)))
            g.process()
            self.graphs.append(g)
            p = Player(g, args, fixed_test=args.fix_test).cuda()
            self.players.append(p)
            self.rshapers.append(RewardShaper(args, p))
            self.accmeters.append(AverageMeter('accmeter', ave_step=10, S=10, criterion="less"))
            self.accmeters_vice.append(AverageMeter('accmeter_vice', ave_step=10, S=10, criterion="greater"))

        self.env = Env(self.players, args)

        self.policy = switcher[args.policynet](args, self.env.statedim, 'actor').cuda()
        self.a_opt = torch.optim.Adam(self.policy.parameters(), lr=self.args.a_rllr)

        if args.pg == 'a2c':
            milestones = [x*int(self.budgets[0]/self.policyfreq[0]+1) for x in [1000, 2000, 3000]]
            self.a_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.a_opt, milestones, gamma=0.3)
            self.critic = switcher[args.policynet](args, self.env.statedim, 'critic').cuda()
            self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=self.args.c_rllr)
            self.c_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.c_opt, milestones, gamma=0.3)
        else:
            self.a_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.a_opt, [1000, 2000, 3000], gamma=0.3)


    def jointtrain(self, maxepisode, begin_time):
        for episode in range(1, maxepisode + 1):
            for playerid in range(len(self.datasets)):
                if self.args.pg == 'a2c':
                    l_state, l_action, l_pool, l_reward_unshaped,  l_punish_mask, l_cd = self.playPartlyEpisode(playerid, episode)
                    a_loss, c_loss = self.finishEpisodeAC(l_state, l_action, l_pool, l_reward_unshaped, l_punish_mask, l_cd, episode, playerid)
                    if episode % self.args.logfreq == 0:
                        print('The {}th time: Policy loss{:.6f}, Critic loss{:.6f}'.format(episode, a_loss, c_loss))
                    if self.args.save == 1:
                        if (self.accmeters[playerid].should_save() or self.accmeters_vice[playerid].should_save()) and episode > 2000 or episode % 500 == 0:

                            torch.save({'actor_state_dict':self.policy.state_dict(), 'critic_state_dict':self.critic.state_dict}, 
                                    'models/{}.pkl'.format(self.args.datasets+'/'+self.args.policynet+'_'+self.args.savename+'_'+begin_time+'_'+str(episode)))
                else:
                    shapedrewards, logp_actions, p_actions = self.playOneEpisode(episode, playerid)
                    loss = self.finishEpisode(shapedrewards, logp_actions, p_actions)
                    if episode % self.args.logfreq == 0:
                        print('The {}th time: Policy loss{:.6f}'.format(episode, loss))

                    if self.args.save == 1:
                        if (self.accmeters[playerid].should_save() or self.accmeters_vice[playerid].should_save()) and episode > 2000 or episode % 500 == 0:

                            torch.save(self.policy.state_dict(),
                                       'models/{}.pkl'.format(self.args.datasets+'/'+self.args.policynet+'_'+self.args.savename+'_'+begin_time+'_'+str(episode)))


    def playOneEpisode(self, episode, playerid=0):
        self.playerid = playerid
        # WARM UP  
        self.env.reset(playerid)
        for _ in range(self.args.warmup_epoch):
            self.players[playerid].trainOnce()

        rewards, logp_actions, p_actions, punish_masks = [], [], [], []
        self.states, self.pools = [], []
        initialrewards = self.env.players[playerid].validation()
        rewards.append(initialrewards)
        punish_masks.append([0] * self.args.batchsize)
        self.action_index = np.zeros([self.args.batchsize, self.budgets[playerid]])

        for epoch in range(self.budgets[playerid]):
            state = self.env.getState(playerid)
            self.states.append(state)
            pool = self.env.players[playerid].getPool(reduce=False)
            self.pools.append(pool)
            logits = self.policy(state, self.graphs[playerid].normadj)
            action, logp_action, p_action, _ = self.selectActions(logits, pool)
            punish_mask = self.env.players[playerid].updateSelectedLabelNum(action)
            punish_masks.append(punish_mask)
            self.action_index[:, epoch] = action.detach().cpu().numpy()
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            rewards.append(self.env.step(action, playerid))
        
        self.env.players[playerid].trainRemain()
        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        punish_masks = np.stack(punish_masks, axis=0)

        finalrewards = self.env.players[playerid].validation(rerun=True)
        metric_dict = {'microf1': 0, 'macrof1': 1, 'auc': 2, 'f1_std': 3, 'label_std': 4, 'f1_scores': 5}

        final_microf1, _ = mean_std(finalrewards[metric_dict['microf1']])
        final_macrof1, _ = mean_std(finalrewards[metric_dict['macrof1']])
        final_auc, _ = mean_std(finalrewards[metric_dict['auc']])
        final_f1std, _ = mean_std(finalrewards[metric_dict['f1_std']])
        sln = self.env.players[playerid].selected_label_num
        final_imb_ratio = (torch.min(sln, dim=-1)[0]/torch.max(sln, dim=-1)[0]).float().mean().numpy()
        final_labelstd, _ = mean_std(finalrewards[metric_dict['label_std']])


        self.accmeters[playerid].update(final_labelstd)
        self.accmeters_vice[playerid].update(final_imb_ratio)

        shapedrewards = self.rshapers[playerid].reshape(rewards, finalrewards, punish_masks)

        return shapedrewards, logp_actions, p_actions


    def playPartlyEpisode(self, playerid, episode):
        self.playerid = playerid
        # WARM UP
        self.env.reset(playerid)
        for _ in range(self.args.warmup_epoch):
            self.players[playerid].trainOnce()

        logp_actions, rewards, punish_masks, class_divs = [], [], [], []
        self.states, self.pools, self.next_states = [], [], []
        initialrewards = self.env.players[playerid].validation()
        rewards.append(initialrewards)
        punish_masks.append([0] * self.args.batchsize)
        class_divs.append(torch.ones(self.args.batchsize).cuda())

        for epoch in range(self.budgets[playerid]):

            state = self.env.getState(playerid)
            self.states.append(state)
            pool = self.env.players[playerid].getPool(reduce=False)
            self.pools.append(pool)
            logits = self.policy(state, self.graphs[playerid].normadj)
            action, logp_action, _, action_inpool = self.selectActions(logits, pool)
            logp_actions.append(logp_action)
            if epoch == self.budgets[playerid] - 1:
                latest_action_inpool = action_inpool
            punish_mask = self.env.players[playerid].updateSelectedLabelNum(action)

            punish_masks.append(punish_mask)
            rewards.append(self.env.step(action, playerid))
            next_state = self.env.getState(playerid)

            cd = self.calculateCD(action, playerid)
            class_divs.append(cd)
            self.next_states.append(next_state)

            if (epoch + 1) % self.policyfreq[playerid] == 0 or (epoch == self.budgets[playerid]-1):

                punish_masks = np.stack(punish_masks, axis=0)
                class_divs = torch.stack(class_divs)


                if epoch == self.budgets[playerid] - 1:
                    latest_punish_mask = punish_masks[-1]
                    logp_actions = logp_actions[:-1]
                    latest_class_div = class_divs[-1]

                shapedrewards, latest_reward_unshaped = self.rshapers[playerid].reshape_TD(
                    rewards, punish_masks, class_divs, epoch, episode == 1, self.budgets[playerid])
                self.finishPartlyEpisode(self.states, logp_actions, shapedrewards, self.next_states)
                
                if epoch == self.budgets[playerid] - 1:
                    return self.states[-1], latest_action_inpool, self.pools[-1], \
                        latest_reward_unshaped, latest_punish_mask, latest_class_div
                else:
                    self.states, self.next_states = [], []
                    logp_actions, rewards, punish_masks, class_divs = \
                        [], [rewards[-1]], [punish_masks[-1]], [class_divs[-1]]


    def finishPartlyEpisode(self, states, logp_actions, rewards, next_states):
        rewards = torch.from_numpy(rewards).cuda().type(torch.float32)
        steps = rewards.shape[0]

        if steps > 0:
            # update critic
            Vs, next_Vs = [], []
            for i in range(steps):
                Vs.append(self.critic(states[i], self.graphs[self.playerid].normadj))
                next_Vs.append(self.critic(next_states[i], self.graphs[self.playerid].normadj))
            Vs, next_Vs = torch.stack(Vs), torch.stack(next_Vs)
            
            targets = rewards + self.args.a2c_gamma * next_Vs
            critic_losses = F.mse_loss(Vs, targets.detach())
            self.c_opt.zero_grad()
            critic_losses.backward()
            self.c_opt.step()
            if self.args.schedule:
                last_lr = self.c_scheduler.get_last_lr()
                self.c_scheduler.step()
                curr_lr = self.c_scheduler.get_last_lr()
                if last_lr != curr_lr:
                    print("c_lr has changed from {} to {}".format(last_lr, curr_lr))
            
            # update actor
            logp_actions = torch.stack(logp_actions)
            advantages = targets - Vs
            actor_losses = -torch.mean(torch.sum((logp_actions * advantages.detach()), dim=0))
            self.a_opt.zero_grad()
            actor_losses.backward()
            self.a_opt.step()
            if self.args.schedule:
                last_lr = self.a_scheduler.get_last_lr()
                self.a_scheduler.step()
                curr_lr = self.a_scheduler.get_last_lr()
                if last_lr != curr_lr:
                    print("a_lr has changed from {} to {}".format(last_lr, curr_lr))


    def finishEpisode(self, rewards, logp_actions, p_actions):
        rewards = torch.from_numpy(rewards).cuda().type(torch.float32)
        if self.args.pg == 'reinforce':
            losses = logp_actions * rewards
            loss = -torch.mean(torch.sum(losses, dim=0))
            self.a_opt.zero_grad()
            loss.backward()
            self.a_opt.step()
            if self.args.schedule:
                last_lr = self.a_scheduler.get_last_lr()
                self.a_scheduler.step()
                curr_lr = self.a_scheduler.get_last_lr()
                if last_lr != curr_lr:
                    print("a_lr has changed from {} to {}".format(last_lr, curr_lr))

        else:
            if self.args.pg == 'ppo':
                epsilon = 0.2
                p_old = p_actions.detach()
                r_sign = torch.sign(rewards).type(torch.float32)
                for i in range(self.args.ppo_epoch):
                    if i != 0:
                        p_actions = [self.trackActionProb(self.states[i], self.pools[i], self.actions[i]) for i in range(len(self.states))]
                        p_actions = torch.stack(p_actions)
                    ratio = p_actions / p_old
                    losses = torch.min(ratio * rewards, (1 + epsilon * r_sign) * rewards)
                    loss = -torch.mean(losses)
                    self.a_opt.zero_grad()
                    loss.backward()
                    self.a_opt.step()

        return loss.item()


    def finishEpisodeAC(self, latest_state, latest_action_inpool, latest_pool, \
                        latest_reward_unshaped, latest_punish_mask, latest_cd, episode, playerid=0):
        self.env.players[playerid].trainRemain()
        finalreward = self.env.players[playerid].validation(rerun=True)
        shapedreward = self.rshapers[playerid].reshape_TD_final(latest_reward_unshaped, finalreward, \
             latest_punish_mask, latest_cd, episode == 1, self.budgets[playerid])

        v = self.critic(latest_state, self.graphs[self.playerid].normadj)
        target = torch.Tensor(shapedreward).float().cuda()
        critic_loss = F.mse_loss(v, target.detach())
        self.c_opt.zero_grad()
        critic_loss.backward()
        self.c_opt.step()
        if self.args.schedule:
            last_lr = self.c_scheduler.get_last_lr()
            self.c_scheduler.step()
            curr_lr = self.c_scheduler.get_last_lr()
            if last_lr != curr_lr:
                print("c_lr has changed from {} to {}".format(last_lr, curr_lr))

        advantages = (target - v).detach()
        logit = self.policy(latest_state, self.graphs[self.playerid].normadj)
        latest_logp_action = self.selectActions(logit, latest_pool, latest_action_inpool)

        actor_loss = -torch.mean(latest_logp_action * advantages)
        self.a_opt.zero_grad()
        actor_loss.backward()
        self.a_opt.step()
        if self.args.schedule:
            last_lr = self.a_scheduler.get_last_lr()
            self.a_scheduler.step()
            curr_lr = self.a_scheduler.get_last_lr()
            if last_lr != curr_lr:
                print("a_lr has changed from {} to {}".format(last_lr, curr_lr))

        metric_dict = {'microf1': 0, 'macrof1': 1, 'auc': 2, 'f1_std': 3, 'label_std': 4, 'f1_scores': 5}

        final_microf1, _ = mean_std(finalreward[metric_dict['microf1']])
        final_macrof1, _ = mean_std(finalreward[metric_dict['macrof1']])
        final_auc, _ = mean_std(finalreward[metric_dict['auc']])
        final_f1std, _ = mean_std(finalreward[metric_dict['f1_std']])

        sln = self.env.players[playerid].selected_label_num
        final_imb_ratio = (torch.min(sln, dim=-1)[0]/torch.max(sln, dim=-1)[0]).float().mean().numpy()
        # print(final_imb_ratio)
        final_labelstd, _ = mean_std(finalreward[metric_dict['label_std']])

        self.accmeters[playerid].update(final_labelstd)
        self.accmeters_vice[playerid].update(final_imb_ratio)


        return actor_loss, critic_loss


    def trackActionProb(self, state, pool, action):
        logits = self.policy(state, self.graphs[self.playerid].normadj)
        valid_logits = logits[pool].reshape(self.args.batchsize, -1)
        max_logits = torch.max(valid_logits, dim=1, keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)
        prob = valid_probs[(list(range(self.args.batchsize)), action)]
        return prob


    def selectActions(self, logits, pool, given_action_inpool=None):
        valid_logits = logits[pool].reshape(self.args.batchsize, -1)
        max_logits = torch.max(valid_logits, dim=1, keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)
        self.valid_probs = valid_probs
        pool = pool[1].reshape(self.args.batchsize, -1)
        assert pool.size() == valid_probs.size()
        dist = Categorical(valid_probs)

        if given_action_inpool is not None:
            logprob = dist.log_prob(given_action_inpool)
            return logprob
        else:
            action_inpool = dist.sample()
            logprob = dist.log_prob(action_inpool)
            prob = valid_probs[list(range(self.args.batchsize)), action_inpool]
            action = pool[[x for x in range(self.args.batchsize)], action_inpool]
            return action, logprob, prob, action_inpool
        
    

    def calculateCD(self, action, playerid):
        p = self.env.players[playerid]
        base = torch.where(p.selected_label_num == 0., torch.tensor(1).float(), 1/p.selected_label_num.float()).cuda()  # 若该类别尚未选择过节点 设为1
        selected_class_idx = (p.fulllabel * p.trainmask).long()
        cd_selected = torch.gather(base, dim=1, index=selected_class_idx)
        cd = torch.gather(cd_selected, dim=-1, index=action.unsqueeze(-1))


        return cd.squeeze(-1)

if __name__ == '__main__':

    args = parse_args()
    check_args(args)
    setup_seed(args.seed)

    begin_time = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%m%d_%H:%M:%S')

    singletrain = SingleTrain(args)
    singletrain.jointtrain(args.maxepisode, begin_time)




