import argparse
import random
import copy
# Prevent numpy from using up all cpu
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position
import random
from machin.frame.algorithms import DDPG
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import resnet
import numpy as np 
import utils
from envs import VectorEnv
from torchvision import transforms

action_dim = 1
action_low = -1 
action_high = 1 

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
        self.fc1 = nn.Linear(96*96+2,512)
        self.fc2 = nn.Linear(512,128) 
        self.fc3 = nn.Linear(128,1)



    def forward(self, state,receptacle_num):
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
        state = t.cat([state, receptacle_num], 1)
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
        self.fc1 = nn.Linear(96*96+2+1,512)
        self.fc2 = nn.Linear(512,128) 
        self.fc3 = nn.Linear(128,1)

    def forward(self, state,receptacle_num, action):
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
        state_action = t.cat([state, receptacle_num], 1)
        state_action = t.cat([state_action,action],1)
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


def step(state,cfg,ddpg,device,receptacle_num):

   
    action_n_model = [[None for _ in g] for g in state]

    with t.no_grad():
        #tmp_observations = [[None for _ in g] for g in state]
        for i, g in enumerate(state):
            ddpg.actor.eval()
            for j, s in enumerate(g):                
                if s is not None:
                    old_state = s 
                    old_state = apply_transform(old_state).to(device)    
                    old_receptacle_num = t.unsqueeze(t.tensor(np.array(receptacle_num),dtype=t.long).to(device),0)                
                    action = ddpg.actor(old_state,old_receptacle_num).squeeze(0)
                    action = t.flatten(action).cpu()[0].item()                                   
                    action_n_model[i][j] = action 
                    #action_to_insert = t.tensor(action, dtype=t.long).to(device).view(1,-1)
                    #tmp_observations[i][j] = {"state": {"state": old_state},"action": {"action": action_to_insert},}
            ddpg.actor.train()    
        action_n = create_action_from_model_action(action_n_model)
        return action_n,action_n_model


def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        return
    print(config_path)
    cfg = utils.load_config(config_path)

    # Create env
    if args.real:
        real_robot_indices = list(map(int, args.real_robot_indices.split(',')))
        real_cube_indices = list(map(int, args.real_cube_indices.split(',')))
        env = utils.get_env_from_cfg(cfg, real=True, real_robot_indices=real_robot_indices, real_cube_indices=real_cube_indices)
    else:
        env = utils.get_env_from_cfg(cfg, show_gui=True)

    # Create policy
    random_seed = 0
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    actor = Actor(cfg.num_input_channels, 1,1).to(device)
    actor_t = Actor(cfg.num_input_channels, 1, 1).to(device)
    critic = Critic(cfg.num_input_channels, 1).to(device)
    critic_t = Critic(cfg.num_input_channels, 1).to(device) 

    if cfg.checkpoint_path is not None:
        print("loading",cfg.policy_path)
        policy_checkpoint = t.load(cfg.policy_path, map_location=device)   
        actor.load_state_dict(policy_checkpoint['actor_state_dicts'])
        critic.load_state_dict(policy_checkpoint['critic_state_dicts'])    
    
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())

    gradient_norm_cut_off = np.inf 
    if cfg.grad_norm_clipping is not None:
        gradient_norm_cut_off = cfg.grad_norm_clipping
    #dqn = DQN(qnet=q_net, qnet_target=q_net_t, optimizer=t.optim.SGD, criterion=F.smooth_l1_loss, learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay)
    ddpg = DDPG(
        actor, actor_t, critic, critic_t, optimizer=t.optim.SGD, criterion=F.smooth_l1_loss, learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay
    )
    

    # Run policy
    state = env.reset()
    receptacle_num = env.num_cubes_per_receptacle
    try:
        while True:
            action_n,_ = step(state,cfg,ddpg,device,receptacle_num)
            
            state, _, done, info = env.step(action_n)
            k1 = []
            
            print(info["total_num_cubes_per_receptacle"])
            receptacle_num = env.num_cubes_per_receptacle 
            if done:
                
                state = env.reset()
                receptacle_num = env.num_cubes_per_receptacle 
    finally:
        env.close()

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
parser.add_argument('--real', action='store_true')
parser.add_argument('--real-robot-indices', default='0,1,2,3')
parser.add_argument('--real-cube-indices', default='0,1,3,5,6,7,8,9,10,11')
main(parser.parse_args())