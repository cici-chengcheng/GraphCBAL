import numpy as np
import torch
from src.utils.common import *


class RewardShaper(object):
    def __init__(self, args, p):
        self.args = args
        self.p = p
        self.metric = args.metric
        self.shaping = args.shaping
        self.entcoef = args.entcoef
        self.gamma = 0.9  # GPA用的是0.5
        self.rate = 0.05  # shaped奖励的均值、标准差的更新率
        self.ratio_rate = 0.1  # 相对大小的更新率
        self.frweight = args.frweight  # to make tune the ratio of finalreward and midreward

        if "0" in self.shaping and "1" in self.shaping:
            raise ValueError("arguments invalid")
        self.hashistorymean, self.hashistoryvar = False, False
        if "3" in self.shaping:
            self.historymean = np.zeros((500, args.batchsize))
        if "4" in self.shaping:
            self.histrvar = np.zeros((500, args.batchsize))
        
        self.rectify_reward = args.rectify_reward
        if self.rectify_reward == 3:  # α*macro_f1 + (1-α)*(-f1_std+num_regret) 
            self.alpha = args.alpha
            self.std_rgt_wgt = args.std_rgt_wgt  # f1_std和num_regret的相对大小
        elif self.rectify_reward == 1:  # cos(Πt/2T)*(macro_f1-f1_std) + (1-cos(Πt/2T))*num_regrets
            self.decay_start = args.decay_start
            self.min_cos = args.min_cos
            self.f1_rgt_wgt = args.f1_rgt_wgt
            self.cos = None
        elif self.rectify_reward == 2:  # directly punish
            self.punishment = args.punishment
        elif self.rectify_reward == 4:  # class diversity
            self.alpha = args.alpha
        elif self.rectify_reward in [5,6]:  # class diversity + punish surpass max_pernum
            self.alpha = args.alpha
            self.punishment = args.punishment
        
        self.hashistratiomean, self.hashistfinalratiomean = False, False
        self.hist_ratio_mean, self.hist_final_ratio_mean = np.ones((500, args.batchsize)), np.ones(args.batchsize)


    def reshape(self, rewards_all, finalrewards_all, num_regrets, punish_masks, class_divs, class_diff_divs):
        # print(num_regrets.shape)  # (budget, batchsize)
        # print(len(rewards_all))   # (budget+1, batchsize)
        if type(class_diff_divs) == torch.Tensor:
            class_diff_divs = np.array(class_diff_divs.cpu())
            class_divs = np.array(class_divs.cpu())
        rewards_sub, finalrewards = self._roughProcess(rewards_all, finalrewards_all, num_regrets, punish_masks, class_divs, class_diff_divs)

        rewards = np.zeros_like(rewards_sub)  # 奖励初始化为0  (budget, batchsize)
        # 【不做奖励重塑】
        if "0" in self.shaping:
            rewards += rewards_sub
        # 【考虑未来奖励】
        if "1" in self.shaping:
            for i in range(rewards_sub.shape[0]-1, 0, -1):
                rewards_sub[i-1] += self.gamma*rewards_sub[i]
            rewards += rewards_sub
        # 【收敛后的奖励】
        if "2" in self.shaping:
            rewards += finalrewards*self.frweight  # 默认一个episode中的每一步的奖励均相同
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
        # print("标准化后\n", rewards)
        
        return rewards
    

    def reshape_TD(self, rewards_all, num_regrets, punish_masks, class_divs, class_diff_divs, epoch, first_flag, budget):
        if type(class_diff_divs) == torch.Tensor:
            class_diff_divs = np.array(class_diff_divs.cpu())
            class_divs = np.array(class_divs.cpu())
        rewards_sub, latest_reward_unshaped = self._roughProcess_TD(rewards_all, num_regrets, punish_masks, class_divs, class_diff_divs, epoch, budget)
        rewards = np.zeros_like(rewards_sub)  # 奖励初始化为0  (policyfreq, batchsize)
        rewards += rewards_sub

        hist_s = epoch - rewards_sub.shape[0] + 1
        hist_e = epoch + 1
        
        if epoch == budget - 1:  # 若已到最后一步 则舍弃最后一步的奖励 因为要用classification net收敛后的奖励替代
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

        # print("中间奖励\n", rewards)
        return rewards, latest_reward_unshaped
    

    def reshape_TD_final(self, latest_reward_unshaped, latest_reward, latest_num_regret, latest_punish_mask, latest_cd, latest_cdd, first_flag, budget):
        if latest_cdd is not None:
            latest_cdd = np.array(latest_cdd.cpu())
            latest_cd = np.array(latest_cd.cpu())
        _, finalmac, _, finalf1std, _, _, _ = [np.array(x) for x in latest_reward]
        finalreward = finalmac

        if self.rectify_reward == 1:
            if not self.hashistfinalratiomean:
                self.hist_final_ratio_mean = np.abs(latest_num_regret.mean() / (finalreward-finalf1std).mean())
                self.hashistfinalratiomean= True
                # print("初次最终ratio均值", self.hist_final_ratio_mean)
                # print()
            else:
                self.hist_final_ratio_mean = \
                    (1-self.ratio_rate)*self.hist_final_ratio_mean + \
                        self.ratio_rate*np.abs(latest_num_regret.mean() / (finalreward-finalf1std).mean())
                # print("非初次最终ratio均值", self.hist_final_ratio_mean)
                # print()
            finalreward = (finalreward-finalf1std)*self.hist_final_ratio_mean*self.f1_rgt_wgt + (2-self.cos[-1])*latest_num_regret*(1-self.f1_rgt_wgt)

        elif self.rectify_reward == 2:
            finalreward -= self.punishment * latest_punish_mask

        elif self.rectify_reward == 3:
            if not self.hashistfinalratiomean:
                self.hist_final_ratio_mean = np.abs(latest_num_regret.mean() / finalf1std.mean())
                self.hashistfinalratiomean= True
                # print("初次最终ratio均值", self.hist_final_ratio_mean)
            else:
                self.hist_final_ratio_mean = \
                    (1-self.ratio_rate)*self.hist_final_ratio_mean + \
                        self.ratio_rate*np.abs(latest_num_regret.mean() / finalf1std.mean())
                # print("非初次最终ratio均值", self.hist_final_ratio_mean)
            
            # finalrewards是classification net收敛后的奖励 最终按frweight比例均匀分配到每个时间步上
            finalreward = self.alpha * finalreward + \
                (1-self.alpha) * (-finalf1std*self.hist_final_ratio_mean*self.std_rgt_wgt + latest_num_regret*(1-self.std_rgt_wgt))
               
        # 奖励增益
        if self.rectify_reward in [1,2,3]:
            finalreward -= latest_reward_unshaped

        if self.rectify_reward == 4:
            # 奖励增益
            # finalreward = self.alpha * finalreward - (1-self.alpha) * finalf1std - latest_reward_unshaped
            finalreward = self.alpha * finalreward - latest_reward_unshaped
            # print("alpha*r_sub", self.alpha * finalreward.mean(), "1-alpha*cd", (1-self.alpha) * latest_cd.mean())
            # print()
            # if not self.hashistfinalratiomean:
            #     self.hist_final_ratio_mean = np.abs(finalf1std.mean() / latest_cd.mean())
            #     self.hashistfinalratiomean= True
            #     # print("初次最终ratio均值", self.hist_final_ratio_mean)
            # else:
            #     self.hist_final_ratio_mean = \
            #         (1-self.ratio_rate)*self.hist_final_ratio_mean + self.ratio_rate*np.abs(finalf1std.mean() / latest_cd.mean())
                # print("后续最终ratio均值", self.hist_final_ratio_mean)
            # print("finalreward", finalreward.mean())
            # print("finalcd", latest_cd.mean())
            # print(latest_cd.mean())
            # print("finalf1std", finalf1std.mean())
            # finalreward += (1-self.alpha) * latest_cd * self.hist_final_ratio_mean
            finalreward += (1-self.alpha) * latest_cd
        
        elif self.rectify_reward == 5:
            finalreward = self.alpha * finalreward - latest_reward_unshaped + (1-self.alpha) * latest_cd - self.punishment * latest_punish_mask
        
        elif self.rectify_reward == 6:
            finalreward = self.alpha * finalreward - latest_reward_unshaped + (1-self.alpha) * latest_cd
            finalreward = np.where(latest_punish_mask>0.1, -self.punishment*latest_punish_mask, finalreward)
        # 【mean】
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
        # print("最终奖励", finalreward)

        return finalreward


    def _roughProcess(self, rewards_all, finalrewards_all, num_regrets, punish_masks, class_divs, class_diff_divs):
        _, mac, _,  f1_std, _, _, _ = [np.array(x) for x in list(zip(*rewards_all))]  # 均为 (budget+1, batchsize)
        _, finalmac, _, finalf1std, _, _, _ = [np.array(x) for x in finalrewards_all]  # 均为 (batchsize,)
        rewards, finalrewards = mac, finalmac

        if self.rectify_reward == 1:
            if not self.hashistratiomean and not self.hashistfinalratiomean:
                # print(num_regrets[23:28])
                # print(rewards[23:28])
                # print(f1_std[23:28])
                self.hist_ratio_mean[:rewards.shape[0], :] = np.abs(num_regrets.mean(1, keepdims=True) / (rewards-f1_std).mean(1, keepdims=True))
                self.hist_final_ratio_mean = np.abs(num_regrets[-1].mean() / (finalrewards-finalf1std).mean())
                self.hashistratiomean, self.hashistfinalratiomean= True, True
                # print("初次过程ratio均值\n", self.hist_ratio_mean[10:15, :])
                # print("初次最终ratio均值", self.hist_final_ratio_mean)
                # print()
            else:
                # print(num_regrets[23:28])
                # print(rewards[23:28])
                # print(f1_std[23:28])
                self.hist_ratio_mean[:rewards.shape[0], :] = \
                    (1-self.ratio_rate)*self.hist_ratio_mean[:rewards.shape[0], :] + \
                        self.ratio_rate*np.abs(num_regrets.mean(1, keepdims=True) / (rewards-f1_std).mean(1, keepdims=True))
                self.hist_final_ratio_mean = \
                    (1-self.ratio_rate)*self.hist_final_ratio_mean + \
                        self.ratio_rate*np.abs(num_regrets[-1].mean() / (finalrewards-finalf1std).mean())
                # print("非初次过程ratio均值\n", self.hist_ratio_mean[10:15 :])
                # print("非初次最终ratio均值", self.hist_final_ratio_mean)
                # print()
            budget = rewards.shape[0] - 1
            decay_start = int(budget * self.decay_start)
            if self.cos is None:
                rest_len = rewards.shape[0] - decay_start
                cos = np.cos((np.arange(decay_start, budget+1) - decay_start) * np.pi / (2 * (rest_len-1))).reshape(rest_len, 1)
                cos = np.clip(cos, self.min_cos, 1.0)
                self.cos = cos
            # print(2-self.cos)
            rewards[:decay_start,:] = (rewards[:decay_start,:]-f1_std[:decay_start,:])*self.hist_ratio_mean[:decay_start,:]*self.f1_rgt_wgt + \
                num_regrets[:decay_start,:]*(1-self.f1_rgt_wgt)
            rewards[decay_start:,:] = (rewards[decay_start:,:]-f1_std[decay_start:,:])*self.hist_ratio_mean[decay_start:rewards.shape[0],:]*self.f1_rgt_wgt + \
                (2-self.cos)*num_regrets[decay_start:,:]*(1-self.f1_rgt_wgt)
            # print(rewards)
            finalrewards = (finalrewards-finalf1std)*self.hist_final_ratio_mean*self.f1_rgt_wgt + \
                (2-self.min_cos)*num_regrets[-1]*(1-self.f1_rgt_wgt)
            # print(finalrewards)
        
        elif self.rectify_reward == 2:
            # rewards包含第[0,budget]步 共budget+1步的即时奖励
            # print("原奖励\n", rewards)
            # print("惩罚\n", self.punishment * punish_masks)
            rewards -= self.punishment * punish_masks
            # print("惩罚后\n", rewards)
            # finalrewards无需修改

        elif self.rectify_reward == 3:
            if not self.hashistratiomean and not self.hashistfinalratiomean:
                self.hist_ratio_mean[:rewards.shape[0], :] = np.abs(num_regrets.mean(1, keepdims=True) / f1_std.mean(1, keepdims=True))
                self.hist_final_ratio_mean = np.abs(num_regrets[-1].mean() / finalf1std.mean())
                self.hashistratiomean, self.hashistfinalratiomean= True, True
                # print("初次过程ratio均值\n", self.hist_ratio_mean[100:105, :])
                # print("初次最终ratio均值", self.hist_final_ratio_mean)
            else:
                self.hist_ratio_mean[:rewards.shape[0], :] = \
                    (1-self.ratio_rate)*self.hist_ratio_mean[:rewards.shape[0], :] + \
                        self.ratio_rate*np.abs(num_regrets.mean(1, keepdims=True) / f1_std.mean(1, keepdims=True))
                self.hist_final_ratio_mean = \
                    (1-self.ratio_rate)*self.hist_final_ratio_mean + \
                        self.ratio_rate*np.abs(num_regrets[-1].mean() / finalf1std.mean())
                # print("非初次过程ratio均值\n", self.hist_ratio_mean[100:105 :])
                # print("非初次最终ratio均值", self.hist_final_ratio_mean)
            # rewards包含第[0,budget]步 共budget+1步的即时奖励  f1_std范围[0,0.5] -> *2后[0,1]
            rewards = self.alpha * rewards + \
                (1-self.alpha) * (-f1_std*self.hist_ratio_mean[:rewards.shape[0], :]*self.std_rgt_wgt + num_regrets*(1-self.std_rgt_wgt))     
            # finalrewards是classification net收敛后的奖励 最终按frweight比例均匀分配到每个时间步上
            finalrewards = self.alpha * finalrewards + \
                (1-self.alpha) * (-finalf1std*self.hist_final_ratio_mean*self.std_rgt_wgt + num_regrets[-1]*(1-self.std_rgt_wgt))
        
        rewards_sub = rewards[1:,:] - rewards[:-1,:]  # 奖励增益 rsub[0]=r[1]-r[0] ,..., rsub[budget-1]=r[budget]-r[budget-1]

        if self.rectify_reward == 4:  # cd和cdd无需计算增益
            # f1std_sub = f1_std[1:,:] - f1_std[:-1,:]
            # if not self.hashistratiomean and not self.hashistfinalratiomean:
            #     self.hist_ratio_mean[:rewards_sub.shape[0], :] = np.abs(f1std_sub.mean(1, keepdims=True) / class_divs[1:].mean(1, keepdims=True))
            #     self.hist_final_ratio_mean = np.abs(finalf1std.mean() / class_divs[-1].mean())
            #     self.hashistratiomean, self.hashistfinalratiomean= True, True
            #     # print("初次过程ratio均值\n", self.hist_ratio_mean[100:105, :])
            #     # print("初次最终ratio均值", self.hist_final_ratio_mean)
            # else:
            #     self.hist_ratio_mean[:rewards_sub.shape[0], :] = \
            #         (1-self.ratio_rate)*self.hist_ratio_mean[:rewards_sub.shape[0], :] + \
            #             self.ratio_rate*np.abs(f1std_sub.mean(1, keepdims=True) / class_divs[1:].mean(1, keepdims=True))
            #     self.hist_final_ratio_mean = \
            #         (1-self.ratio_rate)*self.hist_final_ratio_mean + \
            #             self.ratio_rate*np.abs(finalf1std.mean() / class_divs[-1].mean())
                
            # print("f1_std", f1std_sub[-1])
            # print("class_div", class_divs[-1])
            # rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * (class_divs[1:] * self.hist_ratio_mean - f1std_sub)
            # finalrewards = self.alpha * finalrewards + (1-self.alpha) * (class_divs[-1] * self.hist_final_ratio_mean - finalf1std)
            rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * class_divs[1:]
            finalrewards = self.alpha * finalrewards + (1-self.alpha) * class_divs[-1]

        elif self.rectify_reward == 5:
            rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * class_divs[1:] - self.punishment * punish_masks
            finalrewards = self.alpha * finalrewards + (1-self.alpha) * class_divs[-1]

        return rewards_sub, finalrewards
        

    def _roughProcess_TD(self, rewards_all, num_regrets, punish_masks, class_divs, class_diff_divs, epoch, budget):
        _, mac, _, f1_std, _, _, _ = [np.array(x) for x in list(zip(*rewards_all))]  # 均为 (policyfreq+1, batchsize)
        rewards = mac
        
        hist_s = epoch - rewards.shape[0] + 2
        hist_e = epoch + 2

        if self.rectify_reward == 1:
            if not self.hashistratiomean:
                # print(num_regrets[23:28])
                # print(rewards[23:28])
                # print(f1_std[23:28])
                self.hist_ratio_mean[hist_s:hist_e,:] = np.abs(num_regrets.mean(1, keepdims=True) / (rewards-f1_std).mean(1, keepdims=True))
                self.hashistratiomean = True, True
                # print("初次过程ratio均值\n", self.hist_ratio_mean[10:15, :])
                # print()
            else:
                # print(num_regrets[23:28])
                # print(rewards[23:28])
                # print(f1_std[23:28])
                self.hist_ratio_mean[hist_s:hist_e,:] = \
                    (1-self.ratio_rate)*self.hist_ratio_mean[hist_s:hist_e,:] + \
                        self.ratio_rate*np.abs(num_regrets.mean(1, keepdims=True) / (rewards-f1_std).mean(1, keepdims=True))
                # print("非初次过程ratio均值\n", self.hist_ratio_mean[10:15 :])
                # print()
            decay_start = int(budget * self.decay_start)
            if hist_e < decay_start:  # 当前片段的步数在权重衰减的步数之前
                rewards = (rewards - f1_std)*self.hist_ratio_mean[hist_s:hist_e,:]*self.f1_rgt_wgt + num_regrets[hist_s:hist_e,:]*(1-self.f1_rgt_wgt)
            else:
                if self.cos is None:
                    rest_len = budget+1 - decay_start
                    cos = np.cos((np.arange(decay_start, budget+1) - decay_start) * np.pi / (2 * (rest_len-1))).reshape(rest_len, 1)
                    total_cos = np.concatenate((np.ones((rest_len-1, 1)), cos), axis=0)
                    total_cos = np.clip(total_cos, self.min_cos, 1.0)
                    # print(total_cos.shape)
                    self.cos = total_cos
                cur_cos = self.cos[hist_s:hist_e]
                # print(2-cur_cos)
                rewards = (rewards-f1_std)*self.hist_ratio_mean[hist_s:hist_e,:]*self.f1_rgt_wgt + (2-cur_cos)*num_regrets*(1-self.f1_rgt_wgt)

        elif self.rectify_reward == 2:
            # rewards包含第[0,budget]步 共budget+1步的即时奖励
            # print("原奖励\n", rewards)
            # print("惩罚\n", self.punishment * punish_masks)
            rewards -= self.punishment * punish_masks
            # print("惩罚后\n", rewards)
            # finalrewards无需修改
        
        elif self.rectify_reward == 3:
            if not self.hashistratiomean:
                self.hist_ratio_mean[hist_s:hist_e,:] = np.abs(num_regrets.mean(1, keepdims=True) / f1_std.mean(1, keepdims=True))
                self.hashistratiomean = True
                # print("初次过程ratio均值\n", self.hist_ratio_mean[100:105, :])
            else:
                self.hist_ratio_mean[hist_s:hist_e,:] = \
                    (1-self.ratio_rate)*self.hist_ratio_mean[hist_s:hist_e,:] + \
                        self.ratio_rate*np.abs(num_regrets.mean(1, keepdims=True) / f1_std.mean(1, keepdims=True))
                # print("非初次过程ratio均值\n", self.hist_ratio_mean[100:105 :])
            # rewards包含第[0, policyfreq]步 共policyfreq+1步的即时奖励
            rewards = self.alpha * rewards + \
                (1-self.alpha) * (-f1_std*self.hist_ratio_mean[hist_s:hist_e,:]*self.std_rgt_wgt + num_regrets*(1-self.std_rgt_wgt))
            # print("奖励", rewards)
    
        rewards_sub = rewards[1:,:] - rewards[:-1,:]  # 奖励增益 rsub[0] = r[1] - r[0] ,..., rsub[freq*k-1] = r[freq*k] - r[freq*k-1]
        # print(rewards_sub)
        if self.rectify_reward == 4:  # cd和cdd无需计算增益
            # f1std_sub = f1_std[1:,:] - f1_std[:-1,:]  # f1_std增益
            # if not self.hashistratiomean:
            #     self.hist_ratio_mean[hist_s:hist_e-1,:] = np.abs(f1std_sub.mean(1, keepdims=True) / class_divs[1:].mean(1, keepdims=True))
            #     self.hashistratiomean = True
            #     # print("初次过程ratio均值", self.hist_ratio_mean[hist_e-1, :])
            # else:
            #     self.hist_ratio_mean[hist_s:hist_e-1,:] = \
            #         (1-self.ratio_rate)*self.hist_ratio_mean[hist_s:hist_e-1,:] + \
            #             self.ratio_rate*np.abs(f1std_sub.mean(1, keepdims=True) / class_divs[1:].mean(1, keepdims=True))
                # print("后续过程ratio均值", self.hist_ratio_mean[hist_e-1, :])
            # print("f1std_sub", f1std_sub[-1].mean())
            # print("class_div", class_divs[-1].mean())
            # print("reward_sub", rewards_sub[-1].mean())
            # print()
            # rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * (class_divs[1:] * self.hist_ratio_mean[hist_s:hist_e-1,:] - f1std_sub)
            rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * class_divs[1:]
            # print("alpha*r_sub", self.alpha * rewards_sub[-1].mean(), "1-alpha*cd", (1-self.alpha) * class_divs[-1].mean())
        
        elif self.rectify_reward == 5:
            rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * class_divs[1:] - self.punishment * punish_masks[1:]
        
        elif self.rectify_reward == 6:  # 若是大类别，直接punishiment，不加macro和cd
            rewards_sub = self.alpha * rewards_sub + (1-self.alpha) * class_divs[1:]
            rewards_sub = np.where(punish_masks[1:]>0.1, -self.punishment*punish_masks[1:], rewards_sub)  # 若punish mask为1，则直接punishiment，否则用原rewards_sub

        # 保留索引为freq*k-1步的奖励 已加入惩罚项，但未计算增益 待获得finalreward后计算->rsub_[freq*k-1] = r_final - r[freq*k-1]
        # 索引为freq*k-1步的奖励 对应reward的索引 是倒数第二位
        # latest_reward_unshaped = self.alpha * rewards[-2,:] - (1-self.alpha) * f1_std[-2,:]
        latest_reward_unshaped = self.alpha * rewards[-2,:]
        # print("增益", rewards_sub)
        
        return rewards_sub, latest_reward_unshaped


    def componentRatio(self, rewards_sub, finalrewards_all, logprobs):
        r_mean = np.mean(np.abs(rewards_sub))
        f_mean = np.mean(finalrewards_all)
        lp_mean = np.mean(np.abs(logprobs))
        f_ratio = f_mean/r_mean*self.alpha
        lp_ratio = lp_mean/r_mean*self.entcoef
        logger.debug("rmean {:.4f},fratio {:.2f}x{:.4f}={:.3f}, "
                     "lpratio {:.1f}x{:.5f}={:.3f}".format(r_mean,
                                                           f_mean/r_mean,self.alpha,f_ratio,
                                                           lp_mean/r_mean,self.entcoef,lp_ratio))