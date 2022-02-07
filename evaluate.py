import argparse

# Prevent numpy from using up all cpu
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position

import numpy as np
import utils

def run_eval(cfg, num_episodes=20):
    random_seed = 0

    # Create env
    env = utils.get_env_from_cfg(cfg, random_seed=random_seed, use_egl_renderer=False)

    # Create policy
    policy = utils.get_policy_from_cfg(cfg, env.get_robot_group_types(), random_seed=random_seed)

    # Run policy
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    state = env.reset()
    current_ratios = [] 
    while True:
        action = policy.step(state)
        state, _, done, info = env.step(action)
        data[episode_count].append({
            'simulation_steps': info['simulation_steps'],
            'cubes': info['total_cubes'],
            'robot_collisions': info['total_robot_collisions'],
        })
        if done:
            if(info[info["total_num_cubes_per_receptacle"])[1] != 0): 
                current_ratio = [info["total_num_cubes_per_receptacle"])[0]/info["total_num_cubes_per_receptacle"])[1]]
            else: 
                current_ratio = 0
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
    print(eval_path,ratios,sum(ratios)/len(ratios))

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
main(parser.parse_args())
