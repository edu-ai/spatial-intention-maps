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


# Important note:
# In order to successfully run the environment, please git clone the project
# then run:
#    pip install -e ./test_lib/multiagent-particle-envs/
# in project root directory


def create_env(env_name):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(env_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        shared_viewer=False,
    )
    return env


# configurations
env = create_env("simple_spread")
env.discrete_action_input = True
observe_dim = env.observation_space[0].shape[0]
action_num = env.action_space[0].n
max_episodes = 1000
max_steps = 200
# number of agents in env, fixed, do not change
agent_num = 4



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
def step(states,cfg,madddpg,device,exploration_eps=None):
    if(exploration_eps is None): 
        exploration_eps = cfg.final_exploration 
   
    action_n_model = [ None for g in state[0]]

    with t.no_grad():
        #tmp_observations = [[None for _ in g] for g in state]
        
        
        
        results = maddpg.act(
            [{"state": st} for st in states]
        )
        action_n = [] 
        for action in results: 
            if ( random.random() < exploration_eps):
                action = random.randrange(VectorEnv.get_action_space("pushing_robot"))
                action = -1 +((action-0)/(VectorEnv.get_action_space("pushing_robot")-1))*(2)
            else: 
                action = results 
            action_n.append(action )
        action_n_model[i] = action_n

        action_n = create_action_from_model_action(action_n_model)
        return action_n,action_n_model

def main(cfg): 
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    actor = Actor(cfg.num_input_channels, 1,1).to(device)
    critic = Critic(cfg.num_input_channels*num_agents, 1).to(device)

    maddpg = MADDPG(
        [deepcopy(actor) for _ in range(agent_num)],
        [deepcopy(actor) for _ in range(agent_num)],
        [deepcopy(critic) for _ in range(agent_num)],
        [deepcopy(critic) for _ in range(agent_num)],
        t.optim.Adam,
        optimizer=t.optim.SGD, criterion=F.smooth_l1_loss, 
        learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,
        update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, 
        replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay,
        critic_visible_actors=[list(range(agent_num))] * agent_num,
    )

    start_timestep = 0
    episode = 0
    learning_starts = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    
    states = env.reset()
    states = get_states(states,device)
    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):
        # Select an action for each robot
        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))

        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        
        tmp_observations_list = [[] for _ in range(agent_num)]
        
        while not terminal and step <= max_steps:
            step += 1
        
            old_states = states
            action_n,action_n_model = step(old_states,cfg,maddpg,device,exploration_eps)
            # agent model inference
            
            #actions = [int(r[0]) for r in results]
            #action_probs = [r[1] for r in results]

            states, rewards, terminals, _ = env.step(actions_n)
            states = get_states(states,device)

            for tmp_observations, ost, act, st, rew, term in zip(
                tmp_observations_list,
                old_states,
                action_n_model,#action_probs,
                states,
                rewards,
                terminals,
            ):
                tmp_observations.append(
                    {
                        "state": {"state": ost},
                        "action": {"action": act},
                        "next_state": {"state": st},
                        "reward": float(rew),
                        "terminal": term ,
                    }
                )

        maddpg.store_episodes(tmp_observations_list)
        # total reward is divided by steps here, since:
        # "Agents are rewarded based on minimum agent distance
        #  to each landmark, penalized for collisions"
        if done:
            print(info["total_cubes"])
            states = env.reset()
            states = get_states(states,device)
            episode += 1
        
        # update, update more if episode is longer, else less
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            maddpg.update()

        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save policy
            policy_filename = 'policy_{:08d}_maddpg.pth.tar'.format(timestep + 1)
            policy_path = checkpoint_dir / policy_filename
            policy_checkpoint = {
                'timestep': timestep + 1,
                'actor_state_dicts': [actor.state_dict() for actor in maddpg.actors],
                'critic_state_dicts': [critic.state_dict() for critic in maddpg.critics]
            }
            t.save(policy_checkpoint, str(policy_path))

