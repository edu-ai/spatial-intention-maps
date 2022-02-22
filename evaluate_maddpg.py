import argparse
import random
import copy
# Prevent numpy from using up all cpu
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position
import random
from machin.frame.algorithms import MADDPG
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import resnet
import numpy as np 
import utils
from envs import VectorEnv
from torchvision import transforms
from copy import deepcopy

action_dim = 1
action_low = -1 
action_high = 1
# number of agents in env, fixed, do not change
num_agents = 4



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
        state = F.interpolate(state, scale_factor=2.0, mode='bilinear', align_corners=True)
        state = self.conv2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2.0, mode='bilinear', align_corners=True)
        state = self.conv3(state)
        
        state = t.flatten(state,start_dim=1)
        #state = t.cat([state], 1)
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
        self.fc1 = nn.Linear(96*96+1*num_agents,512)
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
        state = F.interpolate(state, scale_factor=2.0, mode='bilinear', align_corners=True)
        state = self.conv2(state)
        state = self.bn2(state)
        state = F.relu(state)
        state = F.interpolate(state, scale_factor=2.0, mode='bilinear', align_corners=True)
        state = self.conv3(state)
        state = t.flatten(state,start_dim=1)
        #print(state.size())
        #state = self.avgpool(state) 
        #state = state.view(state.size(0), -1)
        #state_action = t.cat([state, receptacle_num], 1)
        state_action = t.cat([state,action],1)
        state_action = self.fc1(state_action) 
        state_action = F.relu(state_action) 
        state_action = self.fc2(state_action) 
        state_action = F.relu(state_action) 
        return self.fc3(state_action)


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

def apply_transform(s):
    transform = transforms.ToTensor()
    return transform(s).unsqueeze(0)

def get_states(state,device):
    states = []
    for j, s in enumerate(state[0]):                
        if s is not None:
            states.append(apply_transform(s).to(device)) 
    return states
def step(states,cfg,maddpg):
    #print(type(states[0]))
   
    action_n_model = [ None for g in states[0]]

    with t.no_grad():
        #tmp_observations = [[None for _ in g] for g in state]
        
        
        l = [{"state": st} for st in states]        
        results = maddpg.act(l)
        #print(results)
        action_n = [] 
        for action in results: 
            #print("gotten rsult", results)
            action = results[0].item()
            action_n.append(action )
        action_n_model[0] = action_n

        action_n = create_action_from_model_action(action_n_model)
        return action_n,action_n_model


def run_eval(cfg, num_episodes=20):
    random_seed = 0
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    actor = Actor(cfg.num_input_channels, 1,1).to(device)
    actors = [deepcopy(actor) for _ in range(num_agents)]
    actors_target = [deepcopy(actor) for _ in range(num_agents)]
    critic = Critic(cfg.num_input_channels*num_agents, 1*num_agents).to(device)
    critics = [deepcopy(critic) for _ in range(num_agents)] 
    critic_targets = [deepcopy(critic) for _ in range(num_agents)]

    gradient_norm_cut_off = np.inf 
    if cfg.grad_norm_clipping is not None:
        gradient_norm_cut_off = cfg.grad_norm_clipping

    if cfg.policy_path is not None:
        print("loading",cfg.policy_path)

        policy_checkpoint = t.load(cfg.policy_path, map_location=device)   
        for i, actor_mod  in enumerate(actors): 
            actor_mod.load_state_dict(policy_checkpoint["actor_state_dicts"][i][0])
        
        for i,actors_t in enumerate(actors_target): 
            actors_t.load_state_dict(policy_checkpoint["actor_target_state_dicts"][i][0])            
            
        for i,critic in enumerate(critic_targets): 
            critic.load_state_dict(policy_checkpoint["critic_targets_state_dicts"][i])
       
        for i,critic in enumerate(critics): 
            critic.load_state_dict(policy_checkpoint["critic_state_dicts"][i])
    
    maddpg = MADDPG(
        actors,
        actors_target,
        critics,
        critic_targets,
        optimizer=t.optim.SGD, criterion=F.smooth_l1_loss, 
        learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,
        update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, 
        replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay,
        critic_visible_actors=[list(range(num_agents))] * num_agents,use_jit=True
    )
    
    env = utils.get_env_from_cfg(cfg,equal_distribution=False,random_seed=random_seed)


    
    # Run policy
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    states = env.reset()
    states = get_states(states,device)
    receptacle_num = env.num_cubes_per_receptacle
    current_ratios = []

    print("starting",cfg.checkpoint_path) 
    while True:
        old_states = states 
        action_n,_ = step(old_states,cfg,maddpg)
        states, _, done, info = env.step(action_n)
        states = get_states(states,device)
        data[episode_count].append({
            'simulation_steps': info['simulation_steps'],
            'cubes': info['total_cubes'],
            'robot_collisions': info['total_robot_collisions'],
        })
        if done:
            #if(info["total_num_cubes_per_receptacle"][1] != 0): 
            #    current_ratio = info#[info["total_num_cubes_per_receptacle"][0]/info["total_num_cubes_per_receptacle"][1]]
            #else: 
            #    current_ratio = 0
            current_ratio = info["total_num_cubes_per_receptacle"]
            current_ratios.append(current_ratio)
            episode_count += 1
            print('Completed {}/{} ||| episodes number of cubes:{} ||| total number of cubes per receptacle: {}'.format(episode_count, num_episodes,info["total_cubes"], info["total_num_cubes_per_receptacle"]))
            if episode_count >= num_episodes:
                break
            states = env.reset()
            states = get_states(states,device)
    env.close()

    return data,current_ratios

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        return
    cfg = utils.load_config(config_path)
    eval_dir = utils.get_eval_dir()
    eval_path = eval_dir / '{}.npy'.format(cfg.run_name)
    data,ratios = run_eval(cfg)
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(eval_path, np.array(data, dtype=object))
    eval_path = eval_dir / '{}_ratio.npy'.format(cfg.run_name)
    np.save(eval_path, np.array(ratios, dtype=object))
    print(eval_path,ratios)

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
main(parser.parse_args())
