
import os
import argparse
import random
import sys
from pathlib import Path
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

from machin.frame.algorithms import DQNPer
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import resnet
import numpy as np 
import utils
import dill
from envs import VectorEnv
from torchvision import transforms
from tqdm import tqdm 
from collections import namedtuple
#from machin.frame.algorithms.dqn import Transition
# configurations
'''
env = gym.make("CartPole-v0")
observe_dim = 4
action_num = 2
max_episodes = 1000
max_steps = 200
solved_reward = 190
solved_repeat = 5
'''

#Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

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
                        transition = (self.prev_state[i][j], self.prev_action[i][j], reward[i][j], s,done)
                        transitions_per_buffer[i].append(transition)
                    self.prev_state[i][j] = s
        return transitions_per_buffer


# model definition
class QNet(nn.Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, num_output_channels, kernel_size=1, stride=1)
    def forward(self, state):
        state = self.resnet18.features(state)
        state = self.conv1(state)
        state = self.bn1(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        state = self.conv2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(state)



def apply_transform(s):
    transform = transforms.ToTensor()
    return transform(s).unsqueeze(0)

def step(state,cfg,dqn,device,exploration_eps=None):
    if exploration_eps is None:
            exploration_eps = cfg.final_exploration

    action_n = [[None for _ in g] for g in state]
    
    with t.no_grad():
        #tmp_observations = [[None for _ in g] for g in state]
        for i, g in enumerate(state):
            dqn.qnet.eval()
            for j, s in enumerate(g):                
                if s is not None:
                    old_state = s 

                    old_state = apply_transform(old_state).to(device)
                    #TODO need to include this function ```act``` in the dqn portion to get just the value not the max index 
                    action = dqn.qnet(old_state).squeeze(0)
                    action = action.view(1, -1).max(1)[1].item()
                    if random.random() < exploration_eps:
                        action = random.randrange(VectorEnv.get_action_space("pushing_robot"))
                    #else:
                    #    a = o.view(1, -1).max(1)[1].item()
                    action_n[i][j] = action
                    #action_to_insert = t.tensor(action, dtype=t.long).to(device).view(1,-1)
                    #tmp_observations[i][j] = {"state": {"state": old_state},"action": {"action": action_to_insert},}
            dqn.qnet.train()    
        return action_n
        '''
        new_state, reward, terminal, _ = env.step(action_n)
        buffer_tmp = []
        for i, g in enumerate(new_state):
            for j, s in enumerate(g): 
                if((s is not None) and tmp_observations[i][j] is not None ): 
                    
                    tmp_obs = tmp_observations[i][j] 
                    new_state_val  = s
                    if(s is not None): 
                        new_state_val = apply_transform(new_state_val)
                    tmp_obs["next_state"] =  {"next_state": new_state_val}
                    tmp_obs["reward"] =  reward[i][j]
                    tmp_obs["terminal"] =  terminal
                    buffer_tmp.append(tmp_obs)
        if(len(buffer_tmp)>0):
            #print(buffer_tmp[0]["action"])
            dqn.store_episode(buffer_tmp)
            print("replay buffer length",len(dqn.replay_buffer.storage))
        return new_state,terminal
        '''
        #state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
        #total_reward += reward
    
     

def main(cfg):
    
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    print("starting on device ",device)
    q_net = QNet(cfg.num_input_channels, 1).to(device)  
    q_net_t = QNet(cfg.num_input_channels, 1).to(device)  
    if cfg.checkpoint_path is not None:
        policy_checkpoint = t.load(cfg.policy_path, map_location=device)        
        q_net.load_state_dict(policy_checkpoint['state_dicts'])
    
   
    q_net_t.load_state_dict(q_net.state_dict())
    q_net_t.eval()
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
    dqn = DQNPer(qnet=q_net, qnet_target=q_net_t, optimizer=t.optim.SGD, criterion=F.smooth_l1_loss, learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay)
    start_timestep = 0
    episode = 0
    if cfg.checkpoint_path is not None:
        checkpoint = dill.load(open(str(cfg.checkpoint_path),'rb'))
        start_timestep = checkpoint['timestep']
        episode = checkpoint['episode']
        dqn.qnet_optim.load_state_dict(checkpoint['optimizers'])
        dqn.replay_buffer.memory = checkpoint["replay_buffer_memory"] 
        dqn.replay_buffer.memory_data = checkpoint["replay_buffer_memory_data"]

        dqn.replay_buffer.sampled_batches = checkpoint["replay_sampled_batches"]
        dqn.replay_buffer.experience_count = checkpoint["replay_experience_count"]
        dqn.replay_buffer.current_batch = checkpoint["replay_current_batch"]
        dqn.replay_buffer.priorities_sum_alpha = checkpoint["replay_priorities_sum_alpha"]
        dqn.replay_buffer.priorities_max = checkpoint["replay_priorities_max"] 
        dqn.replay_buffer.weights_max = checkpoint["replay_weight_max"]
    
    learning_starts = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    
    state = env.reset()
    transition_tracker = TransitionTracker(state)

    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):
        # Select an action for each robot
        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))
        
        #action = policy.step(state, exploration_eps=exploration_eps)

        #terminal = False
        #state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []
        action_n = step(state,cfg,dqn,device,exploration_eps)

        transition_tracker.update_action(action_n)

        # Step the simulation
        state, reward, done, info = env.step(action_n)

        # Store in buffers
        transitions_per_buffer = transition_tracker.update_step_completed(reward, state, done)
        for i, transitions in enumerate(transitions_per_buffer):
            for transition in transitions:
                dqn.replay_buffer.add(transition[0],transition[1],transition[2],transition[3],transition[4])
        #print(len(dqn.replay_buffer.buffer))
        if done:
            print("num boxes", info["total_cubes"])
            state = env.reset()
            episode += 1
        
        # update, update more if episode is longer, else less
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            dqn.update(update_value= True,update_target=False,concatenate_samples=False)
        if (timestep+1)%20 == 0 : 
            dqn.replay_buffer.update_memory_sampling() 
        if  (timestep+1)%3000 == 0: 
            dqn.replay_buffer.update_parameters()
        if (timestep + 1) % cfg.target_update_freq == 0:
            #dqn.update(update_value= False,update_target=True,concatenate_samples=False)
            dqn.qnet_target.load_state_dict(dqn.qnet.state_dict())
        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save policy
            policy_filename = 'policy_{:08d}_dqn_per.pth.tar'.format(timestep + 1)
            policy_path = checkpoint_dir / policy_filename
            policy_checkpoint = {
                'timestep': timestep + 1,
                'state_dicts': dqn.qnet.state_dict(),
            }
            t.save(policy_checkpoint, str(policy_path))

            # Save checkpoint
            checkpoint_filename = 'checkpoint_{:08d}_dqn_per.pth.tar'.format(timestep + 1)
            checkpoint_path = checkpoint_dir / checkpoint_filename
            
            checkpoint = {
                'timestep': timestep + 1,
                'episode': episode,
                'optimizers': dqn.qnet_optim.state_dict() ,
                'replay_buffer_memory': dqn.replay_buffer.memory, 
                'replay_buffer_memory_data': dqn.replay_buffer.memory_data,
                'replay_sampled_batches' : dqn.replay_buffer.sampled_batches, 
                'replay_experience_count': dqn.replay_buffer.experience_count, 
                'replay_current_batch': dqn.replay_buffer.current_batch, 
                'replay_priorities_sum_alpha': dqn.replay_buffer.priorities_sum_alpha, 
                'replay_priorities_max': dqn.replay_buffer.priorities_max, 
                'replay_weight_max': dqn.replay_buffer.weights_max
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