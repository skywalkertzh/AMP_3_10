import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict

# from humanoidverse.agents.amp_modules.amp_modules import AMPLoader, AMPDiscriminator, Normalizer

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
import glob
from humanoidverse.agents.amp_modules.replay_buffer import ReplayBuffer
from humanoidverse.utils.motion_lib.motion_lib_amp_loader import MotionLibAMPLoader as AMPLoader
from humanoidverse.agents.amp_modules.amp_discriminator import AMPDiscriminator
from humanoidverse.agents.amp_modules.utils import Normalizer
console = Console()

class AMP_PPO(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device) 

        self.amp_rew_buffer = deque(maxlen=100)
        self.task_rew_buffer = deque(maxlen=100)
        self.cur_amp_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_task_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        #TODO_loader1:
        #! AMP dataloader/normalizer/discriminator initialization
        self.amp_data = self.env.amp_loader
        #TODO_loader2: done
        #! Q: env init在前还是在后？-》在前
        self.amp_data_obs_dim = self.env.get_amp_observations_dim()
        # self.amp_normalizer = Normalizer(self.amp_data.observation_dim)
        self.amp_normalizer = Normalizer(self.amp_data_obs_dim)
        self.amp_discriminator = AMPDiscriminator(
            self.amp_data_obs_dim * 2,
            self.amp_reward_coef,
            self.amp_discr_hidden_dims, device,
            self.amp_task_reward_lerp).to(self.device)
        self.amp_storage = ReplayBuffer(                                    #* 在hv中新加了这个数据类型用于记录amp所需的storage
            self.amp_discriminator.input_dim // 2, self.amp_replay_buffer_size, device)
        self.amp_transition = RolloutStorage.Transition()

        self.min_std = (
            torch.tensor(self.amp_min_normalized_std, device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))        
        ### used in update ###
        self.sample_amp_policy = None
        self.sample_amp_expert = None


    def _init_config(self):
        super()._init_config()
        # AMP related Config
        #TODO: change them in config file
        self.amp_reward_coef = self.config.amp.amp_reward_coef #*
        # self.amp_num_preload_transitions = self.config.amp.amp_num_preload_transitions #*
        self.amp_task_reward_lerp = self.config.amp.amp_task_reward_lerp #*
        self.amp_discr_hidden_dims = self.config.amp.amp_discr_hidden_dims #*
        self.amp_replay_buffer_size = self.config.amp.amp_replay_buffer_size
        self.amp_learning_rate = self.config.amp.amp_learning_rate
        
        min_normalized_std_list = list(self.config.amp.min_normalized_std)
        self.amp_min_normalized_std = min_normalized_std_list*self.env.num_dof

    def _setup_models_and_optimizer(self):
        # 
        # get called in amp_ppo.setup():
        # def setup(self):
        #     # import ipdb; ipdb.set_trace()
        #     logger.info("Setting up PPO")
        #     self._setup_models_and_optimizer()
        #     logger.info(f"Setting up Storage")
        #     self._setup_storage()    
        super()._setup_models_and_optimizer()
        #! setup amp_optimizer
        amp_parameters = [
            {'params': self.amp_discriminator.trunk.parameters(),
             'weight_decay': 10e-4, 
             'name': 'amp_trunk'},
            {'params': self.amp_discriminator.amp_linear.parameters(),
             'weight_decay': 10e-2, 
             'name': 'amp_head'}]
        self.amp_optimizer = optim.Adam(amp_parameters, lr=self.amp_learning_rate)

    ## Zenghao: only a script. remove when all done
    def learn(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
            
        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        for it in range(self.current_learning_iteration, tot_iter):
            self.start_time = time.time()


            obs_dict =self._rollout_step(obs_dict) #*done
            #_rollout_step: need heavily modified
            #obs_dict with amp element
            

            loss_dict = self._training_step() #*done

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                'it': it,
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'amp_rew_buffer': self.amp_rew_buffer,
                'task_rew_buffer': self.task_rew_buffer,
                'num_learning_iterations': num_learning_iterations,
            }
            self._post_epoch_logging(log_dict)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

            self.ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        

    #* amp version
    def _train_mode(self):
        super()._train_mode()
        self.amp_discriminator.train()

    #* amp version, change noted with #!
    def _rollout_step(self, obs_dict):
        amp_obs = obs_dict["amp_obs"] #!dont know if function or not
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                # actions = self.actor.act(obs_dict["actor_obs"]).detach()

                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                #! change 1
                obs_dict, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actor_state)
                #!
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                #! change 2
                next_amp_obs = self.env.get_amp_observations()
                next_amp_obs = next_amp_obs.to(self.device)
                #!

                #! change 3. AMP related reward change
                next_amp_obs_with_term = torch.clone(next_amp_obs)
                next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
                """
                    terminated envs' amp_obs is fixed to terminal_amp_states
                """
                self.task_rewards = rewards

                # rewards = self.amp_discriminator.predict_amp_reward(
                #     amp_obs, next_amp_obs_with_term, rewards, normalizer=self.amp_normalizer)[0]
                

                # self.amp_rewards = self.amp_discriminator.predict_amp_reward(
                #     amp_obs, next_amp_obs_with_term, rewards, normalizer=self.amp_normalizer)[2]
                
                rew_ret_dict = self.amp_discriminator.predict_amp_reward(
                    amp_obs, next_amp_obs_with_term, rewards, normalizer=self.amp_normalizer)
                
                self.debug_record = {}
                self.debug_record['rew_ret_dict'] = rew_ret_dict
                self.debug_record['amp_obs'] = amp_obs
                self.debug_record['next_amp_obs_with_term'] = next_amp_obs_with_term
                self.debug_record['rewards'] = rewards

                rewards = rew_ret_dict[0]                
                self.amp_rewards = rew_ret_dict[2]

                #! temp
                """
                    reward modification
                """
                amp_obs = torch.clone(next_amp_obs)
                self.amp_transition.observations = amp_obs
                #!
                self.episode_env_tensors.add(infos["to_log"])            
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                #! change 4
                self._process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                #!
                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_task_reward_sum += self.task_rewards
                    self.cur_amp_reward_sum += self.amp_rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.amp_rew_buffer.extend(self.cur_amp_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.task_rew_buffer.extend(self.cur_task_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_amp_reward_sum[new_ids] = 0
                    self.cur_task_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # prepare data for training

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'), 
                dones=self.storage.query_key('dones'), 
                rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict
    
    def _setup_storage(self):
        super()._setup_storage()
        self.storage.register_key('amp_obs', shape=(self.amp_data_obs_dim,), dtype=torch.float)

    def _process_env_step(self, rewards, dones, infos, next_amp_obs_with_term):        
        self.amp_storage.insert( self.amp_transition.observations, next_amp_obs_with_term)
        self.amp_transition.clear()
        super()._process_env_step(rewards, dones, infos)

    ######################
    ### AMP_PPO update ###
    ######################

    # Need to be removed
    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step() 
        #TODO1: add amp_related loss

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        #TODO2: add policy generator and expert generator
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
        
        # for sample, self.sample_amp_policy, self.sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            
        #     for policy_state_dict in sample: 
        #         # Move everything to the device
        #         # import ipdb; ipdb.set_trace()
        #         for policy_state_key in policy_state_dict.keys():
        #             policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
        #         loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        for sample, self.sample_amp_policy, self.sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            policy_state_dict = sample 
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)
        #TODO 6: 需要把amp相关loss加入到loss_dict中组成；随后能完成loss_dict
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict.keys():
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict
    
    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        #TODO1: add amp_related loss
        loss_dict['AMP'] = 0
        loss_dict['Grad_Pen'] = 0
        loss_dict['Policy_Pred'] = 0
        loss_dict['Expert_Pred'] = 0
        return loss_dict
    
    def _update_ppo(self, policy_state_dict, loss_dict):
        actions_batch = policy_state_dict['actions']
        target_values_batch = policy_state_dict['values']
        advantages_batch = policy_state_dict['advantages']
        returns_batch = policy_state_dict['returns']
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob']
        old_mu_batch = policy_state_dict['action_mean']
        old_sigma_batch = policy_state_dict['action_sigma']

        self._actor_act_step(policy_state_dict)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        value_batch = self._critic_eval_step(policy_state_dict)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                    self.critic_learning_rate = max(1e-5, self.critic_learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)
                    self.critic_learning_rate = min(1e-2, self.critic_learning_rate * 1.5)

                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = self.actor_learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = self.critic_learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        #TODO 3: 计算判别器损失和总损失
        policy_state, policy_next_state = self.sample_amp_policy
        expert_state, expert_next_state = self.sample_amp_expert
        if self.amp_normalizer is not None:
            with torch.no_grad():
                policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
        policy_d = self.amp_discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.amp_discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
        expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
        policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen_loss = self.amp_discriminator.compute_grad_pen(*self.sample_amp_expert, lambda_=10)
        amp_total_loss = amp_loss + grad_pen_loss
        
        #TODO 4: 计算总损失

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        
        critic_loss = self.value_loss_coef * value_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.amp_optimizer.zero_grad()
        # print("skip backward")
        actor_loss.backward()
        critic_loss.backward()  
        amp_total_loss.backward()        #! 不知道这里会否出问题
        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.amp_optimizer.step()
        #TODO 5: 更新AMP正则器
        if self.amp_normalizer is not None:
            self.amp_normalizer.update(policy_state.cpu().numpy())
            self.amp_normalizer.update(expert_state.cpu().numpy())
        
        loss_dict['Value'] += value_loss.item()
        loss_dict['Surrogate'] += surrogate_loss.item()
        loss_dict['Entropy'] += entropy_loss.item()
        loss_dict['AMP'] += amp_loss.item()
        loss_dict['Grad_Pen'] += grad_pen_loss.item()
        loss_dict['Policy_Pred'] += policy_loss.item()
        loss_dict['Expert_Pred'] += expert_loss.item()

        return loss_dict


    ###############################
    ### AMP_PPO model save/load ###
    ###############################

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'amp_optimizer_state_dict': self.amp_optimizer.state_dict(),
            'amp_discriminator_state_dict': self.amp_discriminator.state_dict(),            
            'amp_normalizer_state_dict': self.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, ckpt_path): #!可能有问题
        #! Q： normalizer 不需要learning rate吗？
        # import ipdb; ipdb.set_trace()
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"])            
            self.amp_normalizer=loaded_dict["amp_normalizer_state_dict"]
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])

                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.amp_learning_rate = loaded_dict['amp_optimizer_state_dict']['param_groups'][0]['lr']

                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate, self.amp_learning_rate) #!
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
                logger.info(f"AMP Learning rate: {self.amp_learning_rate}")

            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]
    
    def set_learning_rate(self, actor_learning_rate, critic_learning_rate, amp_learning_rate):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.amp_learning_rate = amp_learning_rate

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras, amp_1, amp_2 = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras, "amp_1": amp_1, "amp_2": amp_2}
        )
        return actor_state


    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        ep_string = f''
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        train_log_dict = {}
        mean_std = self.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))
        train_log_dict['fps'] = fps
        train_log_dict['mean_std'] = mean_std.item()

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        self._logging_to_writer(log_dict, train_log_dict, env_log_dict)

        str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict[
                            'collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                        #   f"""{'Value function loss:':>{pad}} {log_dict['mean_value_loss']:.4f}\n"""
                        #   f"""{'Surrogate loss:':>{pad}} {log_dict['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean AMP reward:':>{pad}} {statistics.mean(log_dict['amp_rew_buffer']):.2f}\n"""
                          f"""{'Mean task reward:':>{pad}} {statistics.mean(log_dict['task_rew_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (collection: {log_dict[
                            'collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                        #   f"""{'Value function loss:':>{pad}} {log_dict['mean_value_loss']:.4f}\n"""
                        #   f"""{'Surrogate loss:':>{pad}} {log_dict['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n""")

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (
                               log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        # Use rich Live to update a specific section of the console
        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            # Your training loop or other operations
            pass

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        # Logging Loss Dict
        for loss_key, loss_value in log_dict['loss_dict'].items():
            self.writer.add_scalar(f'Loss/{loss_key}', loss_value, log_dict['it'])
        self.writer.add_scalar('Loss/actor_learning_rate', self.actor_learning_rate, log_dict['it'])
        self.writer.add_scalar('Loss/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
        self.writer.add_scalar('Policy/mean_noise_std', train_log_dict['mean_std'], log_dict['it'])
        self.writer.add_scalar('Perf/total_fps', train_log_dict['fps'], log_dict['it'])
        self.writer.add_scalar('Perf/collection time', log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_amp_reward', statistics.mean(log_dict['amp_rew_buffer'])/statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_task_reward', statistics.mean(log_dict['task_rew_buffer'])/statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_amp_reward/time', statistics.mean(log_dict['amp_rew_buffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_task_reward/time', statistics.mean(log_dict['task_rew_buffer']), self.tot_time)
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])