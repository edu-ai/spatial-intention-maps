import argparse
import random
# Prevent numpy from using up all cpu
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position
import random
from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import resnet
import numpy as np 
import utils
from envs import VectorEnv
from torchvision import transforms

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


def step(state,cfg,dqn,exploration_eps=None):
    if exploration_eps is None:
            exploration_eps = cfg.final_exploration

    action_n = [[None for _ in g] for g in state]
    
    with t.no_grad():
        tmp_observations = [[None for _ in g] for g in state]
        for i, g in enumerate(state):
            for j, s in enumerate(g):                
                if s is not None:
                    old_state = s 

                    old_state = apply_transform(old_state)
                    #TODO need to include this function ```act``` in the dqn portion to get just the value not the max index 
                    action = dqn.act({"state": old_state})
                    action = action.view(1, -1).max(1)[1].item()
                    if random.random() < exploration_eps:
                        action = random.randrange(VectorEnv.get_action_space("pushing_robot"))
                    #else:
                    #    a = o.view(1, -1).max(1)[1].item()
                    action_n[i][j] = action
    return action_n

def run_eval(cfg, num_episodes=20):
    random_seed = 0
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    q_net = QNet(cfg.num_input_channels, 1).to(device)  
    q_net_t = QNet(cfg.num_input_channels, 1).to(device)  
    if cfg.checkpoint_path is not None:
        print("loading",cfg.policy_path)
        policy_checkpoint = t.load(cfg.policy_path, map_location=device)   
        q_net.load_state_dict(policy_checkpoint['state_dicts'])
    
        
    q_net_t.load_state_dict(q_net.state_dict())    
    # Create env
    env = utils.get_env_from_cfg(cfg, random_seed=random_seed, use_egl_renderer=False)
    gradient_norm_cut_off = np.inf 
    if cfg.grad_norm_clipping is not None:
        gradient_norm_cut_off = cfg.grad_norm_clipping
    dqn = DQN(qnet=q_net, qnet_target=q_net_t, optimizer=t.optim.SGD, criterion=F.smooth_l1_loss, learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay)

    
    # Run policy
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    state = env.reset()
    current_ratios = []

    print("starting",cfg.checkpoint_path) 
    while True:
        action = step(state,cfg,dqn)
        state, _, done, info = env.step(action)
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
            state = env.reset()
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
