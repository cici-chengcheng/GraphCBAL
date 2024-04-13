import numpy as np
import torch


class RewardShaper(object):
    def __init__(self, args, p):
        self.args = args
        self.p = p
        self.metric = args.metric
        self.shaping = args.shaping
        self.entcoef = args.entcoef
        self.gamma = 0.9
        self.rate = 0.05
        self.ratio_rate = 0.1
        self.frweight = args.frweight

        if "0" in self.shaping and "1" in self.shaping:
            raise ValueError("arguments invalid")
        self.hashistorymean, self.hashistoryvar = False, False
        if "3" in self.shaping:
            self.historymean = np.zeros((500, args.batchsize))
        if "4" in self.shaping:
            self.histrvar = np.zeros((500, args.batchsize))

        self.alpha = args.alpha
        self.punishment = args.punishment
        
        self.hashistratiomean, self.hashistfinalratiomean = False, False
        self.hist_ratio_mean, self.hist_final_ratio_mean = np.ones((500, args.batchsize)), np.ones(args.batchsize)


    def reshape(self, rewards_all, finalrewards_all, punish_masks, class_divs):

        if type(class_divs) == torch.Tensor:
            class_divs = np.array(class_divs.cpu())
        rewards_sub, finalrewards = self._roughProcess(rewards_all, finalrewards_all, punish_masks, class_divs)

        rewards = np.zeros_like(rewards_sub)
        # 【No reward reshaping】
        if "0" in self.shaping:
            rewards += rewards_sub
        # 【Consider future rewards】
        if "1" in self.shaping:
            for i in range(rewards_sub.shape[0]-1, 0, -1):
                rewards_sub[i-1] += self.gamma*rewards_sub[i]
            rewards += rewards_sub
        # 【The reward after convergence】
        if "2" in self.shaping:
            rewards += finalrewards*self.frweight
        # 【mean】
        if "3" in self.shaping:
            if not self.hashistorymean:
                self.historymean[:rewards_sub.shape[0], :] += rewards.mean(1,keepdims=True)
                self.hashistorymean = True
            else:
                self.historymean[:rewards_sub.shape[0], :] = \
                    self.historymean[:rewards_sub.shape[0], :] * (1 - self.rate) + self.rate * rewards.mean(1,keepdims=True)
            rewards = rewards - self.historymean[:rewards.shape[0], :]
        # 【std】
        if "4" in self.shaping:
            if not self.hashistoryvar:
                self.histrvar[:rewards_sub.shape[0], :] += rewards.std(1, keepdims=True) + np.ones_like(rewards.std(1, keepdims=True), dtype=float)*1e-5
                self.hashistoryvar = True
            else:
                self.histrvar[:rewards_sub.shape[0], :] = \
                    self.histrvar[:rewards.shape[0], :] * (1 - self.rate) + self.rate * rewards.std(1, keepdims=True)
            rewards = rewards/self.histrvar[:rewards.shape[0], :]


        return rewards


    def reshape_TD(self, rewards_all, punish_masks, class_divs, epoch, first_flag, budget):
        if type(class_divs) == torch.Tensor:
            class_divs = np.array(class_divs.cpu())
        rewards_sub, latest_reward_unshaped = self._roughProcess_TD(rewards_all, punish_masks, class_divs, epoch, budget)
        rewards = np.zeros_like(rewards_sub)
        rewards += rewards_sub

        hist_s = epoch - rewards_sub.shape[0] + 1
        hist_e = epoch + 1

        if epoch == budget - 1:
            rewards = rewards[:-1]
            hist_e -= 1

        # 【mean】
        if "3" in self.shaping:
            if first_flag:
                self.historymean[hist_s:hist_e, :] += rewards.mean(1,keepdims=True)
            else:
                self.historymean[hist_s:hist_e, :] = \
                    self.historymean[hist_s:hist_e, :] * (1 - self.rate) + self.rate * rewards.mean(1,keepdims=True)
            rewards = rewards - self.historymean[hist_s:hist_e, :]
        # 【std】
        if "4" in self.shaping:
            if first_flag:
                self.histrvar[hist_s:hist_e, :] += rewards.std(1,keepdims=True) + np.ones_like(rewards.std(1,keepdims=True), dtype=float) * 1e-5
            else:
                self.histrvar[hist_s:hist_e, :] = \
                    self.histrvar[hist_s:hist_e, :] * (1 - self.rate) + self.rate * rewards.std(1,keepdims=True)
            rewards = rewards/self.histrvar[hist_s:hist_e, :]

        return rewards, latest_reward_unshaped


    def reshape_TD_final(self, latest_reward_unshaped, latest_reward, latest_punish_mask, latest_cd, first_flag, budget):
        if latest_cd is not None:
            latest_cd = np.array(latest_cd.cpu())
        _, finalmac, _, finalf1std, _, _, _ = [np.array(x) for x in latest_reward]
        finalreward = finalmac

        finalreward = self.alpha * finalreward - latest_reward_unshaped + (1-self.alpha) * latest_cd - self.punishment * latest_punish_mask
        

        if "3" in self.shaping:
            if first_flag:
                self.historymean[budget-1, :] += finalreward.mean()
            else:
                self.historymean[budget-1, :] = \
                    self.historymean[budget-1, :] * (1 - self.rate) + self.rate * finalreward.mean()
            finalreward -= self.historymean[budget-1, :]
        # 【std】
        if "4" in self.shaping:
            if first_flag:
                self.histrvar[budget-1, :] += finalreward.std()
            else:
                self.histrvar[budget-1, :] = \
                    self.histrvar[budget-1, :] * (1 - self.rate) + self.rate * finalreward.std()
            finalreward /= self.histrvar[budget-1, :]

        return finalreward


    def _roughProcess(self, rewards_all, finalrewards_all, punish_masks, class_divs):
        _, mac, _,  f1_std, _, _, _ = [np.array(x) for x in list(zip(*rewards_all))]
        _, finalmac, _, finalf1std, _, _, _ = [np.array(x) for x in finalrewards_all]
        rewards, finalrewards = mac, finalmac

        rewards_sub = rewards[1:,:] - rewards[:-1,:]
        rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * class_divs[1:] - self.punishment * punish_masks
        finalrewards = self.alpha * finalrewards + (1-self.alpha) * class_divs[-1]

        return rewards_sub, finalrewards
        

    def _roughProcess_TD(self, rewards_all, punish_masks, class_divs, epoch, budget):
        _, mac, _, f1_std, _, _, _ = [np.array(x) for x in list(zip(*rewards_all))]  # 均为 (policyfreq+1, batchsize)
        rewards = mac

        rewards_sub = rewards[1:,:] - rewards[:-1,:]
        rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * class_divs[1:] - self.punishment * punish_masks[1:]
        
        latest_reward_unshaped = self.alpha * rewards[-2,:]

        return rewards_sub, latest_reward_unshaped


    def componentRatio(self, rewards_sub, finalrewards_all, logprobs):
        r_mean = np.mean(np.abs(rewards_sub))
        f_mean = np.mean(finalrewards_all)
        lp_mean = np.mean(np.abs(logprobs))
        f_ratio = f_mean/r_mean*self.alpha
        lp_ratio = lp_mean/r_mean*self.entcoef