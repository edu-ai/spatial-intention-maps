
import os
import argparse
import random
import sys
from pathlib import Path
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from machin.frame.algorithms import DDPG
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
import torch.nn.functional as F
import resnet
import numpy as np 
import utils
import dill
from envs import VectorEnv
from torchvision import transforms
from tqdm import tqdm 
import copy
# configurations

action_dim = 1
action_low = -1 
action_high = 1

class TransitionTracker:
    def __init__(self, initial_state):
        self.num_buffers = len(initial_state)
        self.prev_state = initial_state
        self.prev_action = [[None for _ in g] for g in self.prev_state]

    def update_action(self, action):
        for i, g in enumerate(action):
            for j, a in enumerate(g):
                if a is not None:
                    self.prev_action[i][j] = a

    def update_step_completed(self, reward, state, done):
        transitions_per_buffer = [[] for _ in range(self.num_buffers)]
        for i, g in enumerate(state):
            for j, s in enumerate(g):
                if s is not None or done:
                    if self.prev_state[i][j] is not None:
                        transition = (self.prev_state[i][j], self.prev_action[i][j], reward[i][j], s)
                        transitions_per_buffer[i].append(transition)
                    self.prev_state[i][j] = s
        return transitions_per_buffer



class OUNoise(object):
    def __init__(self, action_dim,low,high, mu=0.0, theta=0.10, max_sigma=0.3, min_sigma=0.05, decay_period=30000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.low          = low
        self.high         = high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# model definition
class Actor(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1, action_range=1):
        super().__init__()
        '''
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.action_range = action_range
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        '''
        self.action_range = action_range
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, num_output_channels, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(96*96,512)
        self.fc2 = nn.Linear(512,128) 
        self.fc3 = nn.Linear(128,1)



    def forward(self, state):
        '''
        state = self.resnet18.features(state)
        state = self.avgpool(state)
        state = state.view(state.size(0), -1)
        state = t.tanh(self.fc(state)) * self.action_range
        '''
        state = self.resnet18.features(state)
        state = self.conv1(state)
        state = self.bn1(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv3(state)
        state = t.flatten(state,start_dim=1)
        #print(state.size())    
        #state = self.avgpool(state) 
        #state = state.view(state.size(0), -1)
        #print(state.size())
        state  = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state) 
        state = F.relu(state)
        state = t.tanh(self.fc3(state)) * self.action_range

        return state


class Critic(nn.Module):
    def __init__(self, num_input_channels, action_dim):
        super().__init__()
        '''
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512+1, 256)
        self.fc2 = nn.Linear(256, 1)
        '''
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,1, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(96*96+1,512)
        self.fc2 = nn.Linear(512,128) 
        self.fc3 = nn.Linear(128,1)

    def forward(self, state, action):
        '''
        state = self.resnet18.features(state)
        state = self.avgpool(state)
        state = state.view(state.size(0), -1)
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = self.fc2(q)
        return q
        '''
        state = self.resnet18.features(state)
        state = self.conv1(state)
        state = self.bn1(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv3(state)
        state = t.flatten(state,start_dim=1)
        #state = self.avgpool(state) 
        #state = state.view(state.size(0), -1)
        state_action = t.cat([state, action], 1)
        state_action = self.fc1(state_action) 
        state_action = F.relu(state_action) 
        state_action = self.fc2(state_action) 
        state_action = F.relu(state_action) 
        return self.fc3(state_action)



def apply_transform(s):
    transform = transforms.ToTensor()
    return transform(s).unsqueeze(0)

def create_action_from_model_action(action): 
    action_n = copy.deepcopy(action) 
    for i, g in enumerate(action):
        for j, s in enumerate(g):                
            if s is not None:
                max_actual_action_val = VectorEnv.get_action_space("pushing_robot")-1
                min_actual_action_val  = 0 
                actual_action = min_actual_action_val + ((s-action_low) /(action_high-action_low)) *(max_actual_action_val-min_actual_action_val)
                #print(int(actual_action))
                action_n[i][j] = int(actual_action)
    return action_n
def step(state,cfg,ddpg,device,noise):

   
    action_n_model = [[None for _ in g] for g in state]

    with t.no_grad():
        #tmp_observations = [[None for _ in g] for g in state]
        for i, g in enumerate(state):
            ddpg.actor.eval()
            for j, s in enumerate(g):                
                if s is not None:
                    old_state = s 
                    old_state = apply_transform(old_state).to(device)                    
                    action = ddpg.actor(old_state).squeeze(0)
                    action = t.flatten(action).cpu()[0].item()
                    
                    #print(action)
                    action = noise.get_action(action)                     
                    action_n_model[i][j] = action 
                    #action_to_insert = t.tensor(action, dtype=t.long).to(device).view(1,-1)
                    #tmp_observations[i][j] = {"state": {"state": old_state},"action": {"action": action_to_insert},}
            ddpg.actor.train()    
        action_n = create_action_from_model_action(action_n_model)
        return action_n,action_n_model

def main(cfg):
    
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    print("starting on device ",device)
    actor = Actor(cfg.num_input_channels, 1,1).to(device)
    actor_t = Actor(cfg.num_input_channels, 1, 1).to(device)
    critic = Critic(cfg.num_input_channels, 1).to(device)
    critic_t = Critic(cfg.num_input_channels, 1).to(device) 

    
    if cfg.checkpoint_path is not None:
        policy_checkpoint = t.load(cfg.policy_path, map_location=device)        
        actor.load_state_dict(policy_checkpoint['actor_state_dicts'])
        critic.load_state_dict(policy_checkpoint['critic_state_dicts'])
    
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())

    kwargs = {}
    if cfg.show_gui:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use('agg')
    if cfg.use_predicted_intention:  # Enable ground truth intention map during training only
        kwargs['use_intention_map'] = True
        kwargs['intention_map_encoding'] = 'ramp'
    
    env = utils.get_env_from_cfg(cfg, **kwargs)

    #TODO need to include the  momentum=0.9, weight_decay=cfg.weight_decay) for SGD
    gradient_norm_cut_off = np.inf 
    if cfg.grad_norm_clipping is not None:
        gradient_norm_cut_off = cfg.grad_norm_clipping
   
    ddpg = DDPG(
        actor, actor_t, critic, critic_t, optimizer=t.optim.SGD, criterion=F.smooth_l1_loss, learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay
    )

    noise = OUNoise(action_dim,action_low,action_high)
    
    start_timestep = 0
    episode = 0
    if cfg.checkpoint_path is not None:
        checkpoint =  dill.load(open(str(cfg.checkpoint_path),'rb'))
        start_timestep = checkpoint['timestep']
        episode = checkpoint['episode']
        ddpg.critic_optim.load_state_dict(checkpoint['critic_optimizer'])
        ddpg.actor_optim.load_state_dict(checkpoint['actor_optimizer'])
        ddpg.replay_buffer.buffer = checkpoint['replay_buffers']
    
    learning_starts  = 0
    #learning_starts = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    
    state = env.reset()
    transition_tracker = TransitionTracker(state)

    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):
        # Select an action for each robot
        #exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))
        
        #action = policy.step(state, exploration_eps=exploration_eps)

        #terminal = False
        #state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []
        action_n,action_n_model = step(state,cfg,ddpg,device,noise)

        transition_tracker.update_action(action_n_model)

        # Step the simulation
        state, reward, done, info = env.step(action_n)

        # Store in buffers
        transitions_per_buffer = transition_tracker.update_step_completed(reward, state, done)
        for i, transitions in enumerate(transitions_per_buffer):
            for transition in transitions:
                ddpg.replay_buffer.push(*transition)
        

        #print(len(dqn.replay_buffer.buffer))
        if done:
            print(info["total_cubes"])
            state = env.reset()
            episode += 1
        
        # update, update more if episode is longer, else less
        if( len(ddpg.replay_buffer.buffer) > cfg.batch_size and (timestep + 1) % cfg.train_freq == 0): 
            ddpg.update(update_value= True,update_policy= True, update_target=False,concatenate_samples=False)
            ddpg.soft_update_models()
        #if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
        #    dqn.update(update_value= True,update_target=False,concatenate_samples=False)


        #if (timestep + 1) % cfg.target_update_freq == 0:
        #   dqn.update(update_value= False,update_target=True,concatenate_samples=False)
        #    ddpg.actor_target.load_state_dict(ddpg.actor.state_dict())
        #    ddpg.critic_target.load_state_dict(ddpg.critic.state_dict())
        
        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save policy
            policy_filename = 'policy_{:08d}_ddpg.pth.tar'.format(timestep + 1)
            policy_path = checkpoint_dir / policy_filename
            policy_checkpoint = {
                'timestep': timestep + 1,
                'actor_state_dicts': ddpg.actor.state_dict(),
                'critic_state_dicts': ddpg.critic.state_dict(),
            }
            t.save(policy_checkpoint, str(policy_path))

            # Save checkpoint
            checkpoint_filename = 'checkpoint_{:08d}_ddpg.pth.tar'.format(timestep + 1)
            checkpoint_path = checkpoint_dir / checkpoint_filename
            
            checkpoint = {
                'timestep': timestep + 1,
                'episode': episode,
                'critic_optimizer': ddpg.critic_optim.state_dict() ,
                'actor_optimizer': ddpg.actor_optim.state_dict() ,
                'replay_buffers': ddpg.replay_buffer.buffer,
            }
            dill.dump(checkpoint,open(str(checkpoint_path),mode='wb'))
            #t.save(checkpoint, str(checkpoint_path))

            # Save updated config file
            cfg.policy_path = str(policy_path)
            cfg.checkpoint_path = str(checkpoint_path)
            utils.save_config(log_dir / 'config.yml', cfg)

            # Remove old checkpoint
            checkpoint_paths = list(checkpoint_dir.glob('checkpoint_*.pth.tar'))
            checkpoint_paths.remove(checkpoint_path)
            for old_checkpoint_path in checkpoint_paths:
                old_checkpoint_path.unlink()
            '''
            # Remove old policies
            checkpoint_paths = list(checkpoint_dir.glob('policy_*.pth.tar'))
            checkpoint_paths.remove(checkpoint_path)
            for old_checkpoint_path in checkpoint_paths:
                old_checkpoint_path.unlink()
            '''

if __name__ == '__main__':
    t.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    config_path = parser.parse_args().config_path
    if config_path is None:
        if sys.platform == 'darwin':
            config_path = 'config/local/lifting_4-small_empty-local.yml'
        else:
            config_path = utils.select_run()
    if config_path is not None:
        config_path = utils.setup_run(config_path)
        main(utils.load_config(config_path))


#def main():
    
    '''
    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = ddpg.act_with_noise(
                    {"state": old_state}, noise_param=noise_param, mode=noise_mode
                )
                state, reward, terminal, _ = env.step(action.numpy())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward[0]

                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward[0],
                        "terminal": terminal or step == max_steps,
                    }
                )

        ddpg.store_episode(tmp_observations)
        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                ddpg.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
    '''
