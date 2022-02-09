
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

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
from tqdm import tqdm 

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

    def forward(self, x):
        x = self.resnet18.features(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(x)



def apply_transform(s):
    transform = transforms.ToTensor()
    return transform(s).unsqueeze(0)

def step(state,exploration_eps=None,cfg,env,dqn):
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
                    
                    tmp_observations[i][j] = {"state": {"state": old_state},"action": {"action": action},}
                
        new_state, reward, terminal, _ = env.step(action_n)
        buffer_tmp = []
        for i, g in enumerate(tmp_observations):
            for j, s in enumerate(tmp_observations): 
                if(s is not None or done): 
                    tmp_obs = s 
                    new_state_val  = new_state[i][j]
                    new_state_val = apply_transform(new_state_val)
                    tmp_obs["next_state"] =  {"next_state": new_state_val}
                    tmp_obs["reward"] =  {"reward": reward}
                    tmp_obs["terminal"] =  {"terminal": terminal}
                    buffer_tmp.append(tmp_obs)
        if(len(buffer_tmp)>0):
            dqn.store_episode(buffer_tmp)
        return new_state,terminal
        #state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
        #total_reward += reward
    
     

def main(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("starting on device ")
    q_net = QNet(self.cfg.num_input_channels, 1).to(device)  
    q_net_t = QNet(self.cfg.num_input_channels, 1).to(device)  
    if cfg.checkpoint_path is not None:
        policy_checkpoint = torch.load(cfg.policy_path, map_location=device)        
        q_net.load_state_dict(policy_checkpoint['state_dicts'])
        
    q_net_t.load_state_dict(q_net_t.state_dict())

    env = utils.get_env_from_cfg(cfg, **kwargs)

    #TODO need to include the  momentum=0.9, weight_decay=cfg.weight_decay) for SGD
    gradient_norm_cut_off = np.inf 
    if cfg.grad_norm_clipping is not None:
        gradient_norm_cut_off = cfg.grad_norm_clipping
    dqn = DQN(qnet=q_net, qnet_target=q_net_t, optimizer=t.optim.SGD, criterion=F.smooth_l1_loss(), learning_rate = cfg.learning_rate,batch_size=cfg.batch_size,update_rate=None,update_steps=1, discount=cfg.discount_factors[0],gradient_max = gradient_norm_cut_off, replay_size = cfg.replay_buffer_size,momentum=0.9,weight_decay=cfg.weight_decay)
    start_timestep = 0
    episode = 0
    if cfg.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_path)
        start_timestep = checkpoint['timestep']
        episode = checkpoint['episode']
        dqn.qnet_optim.load_state_dict(checkpoint['optimizers'][i])
        dqn.replay_buffer[0].data = checkpoint['replay_buffers'][i]
    

    learning_starts = np.round(cfg.learning_starts_frac * cfg.total_timesteps).astype(np.uint32)
    total_timesteps_with_warm_up = learning_starts + cfg.total_timesteps
    
    state = env.reset()

    for timestep in tqdm(range(start_timestep, total_timesteps_with_warm_up), initial=start_timestep, total=total_timesteps_with_warm_up, file=sys.stdout):
        # Select an action for each robot
        exploration_eps = 1 - (1 - cfg.final_exploration) * min(1, max(0, timestep - learning_starts) / (cfg.exploration_frac * cfg.total_timesteps))
        
        #action = policy.step(state, exploration_eps=exploration_eps)

        #terminal = False
        #state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        tmp_observations = []
        state,done = step(state,exploration_eps,cfg,env,dqn)

        if done:
            state = env.reset()
            episode += 1
        
        # update, update more if episode is longer, else less
        if timestep >= learning_starts and (timestep + 1) % cfg.train_freq == 0:
            dqn.update(update_value= True,update_target=False)

        if (timestep + 1) % cfg.target_update_freq == 0:
            dqn.update(update_value= False,update_target=True)

        if (timestep + 1) % cfg.checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warm_up:
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save policy
            policy_filename = 'policy_{:08d}.pth.tar'.format(timestep + 1)
            policy_path = checkpoint_dir / policy_filename
            policy_checkpoint = {
                'timestep': timestep + 1,
                'state_dicts': dqn.qnet.state_dict(),
            }
            torch.save(policy_checkpoint, str(policy_path))

            # Save checkpoint
            checkpoint_filename = 'checkpoint_{:08d}.pth.tar'.format(timestep + 1)
            checkpoint_path = checkpoint_dir / checkpoint_filename
            
            checkpoint = {
                'timestep': timestep + 1,
                'episode': episode,
                'optimizers': dqn.qnet_optim.state_dict() ,
                # TODO need to change the replaybuffer from tuple to list so that can update later 
                'replay_buffers': dqn.replay_buffer[0].data,
            }
            
            torch.save(checkpoint, str(checkpoint_path))

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
    torch.backends.cudnn.benchmark = True
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