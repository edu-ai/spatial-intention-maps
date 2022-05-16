## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements (tested on Ubuntu 18.04.3 LTS):

```bash
# Create and activate new conda env
conda create -y -n my-conda-env python=3.7.10
conda activate my-conda-env

# Install mkl numpy
conda install -y numpy==1.19.2

# Install pytorch
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Install pip requirements
pip install -r requirements.txt

# Install shortest paths module (used in simulation environment)
cd shortest_paths
python setup.py build_ext --inplace
```

## Quickstart

[`Please use the following drive link to download the pretrained model for DDPG agents`](https://drive.google.com/file/d/14eQEg1owN4b243NN_QK1eqLZHNM0G0ui/view?usp=sharing) 
```bash

# 4 pushing robots
python enjoy_ddpg.py --config-path logs/20220307T194519254398-pushing_4-small_divider-ours-original/config.yml
```


## Training in the 3D Simulator

The [`config/experiments`](config/experiments) directory contains the template config files used for all experiments in the paper. To start a training run, you can provide one of the template config files to the `train.py` script. For example, the following will train a policy on the `SmallDivider` environment:

```bash
python train.py config/experiments/ours/pushing_4-small_divider-original.yml
```

The training script will create a log directory and checkpoint directory for the new training run inside `logs/` and `checkpoints/`, respectively. Inside the log directory, it will also create a new config file called `config.yml`, which stores training run config variables and can be used to resume training or to load a trained policy for evaluation.


### Evaluation

Trained policies can be evaluated using the `evaluate.py` script, which takes in the config path for the training run. For example, to evaluate the DDPG agent pretrained policy, you can run:

```
python evaluate_ddpg.py --config-path logs/20220307T194519254398-pushing_4-small_divider-ours-original/config.yml
```

This will load the trained policy from the specified training run, and run evaluation on it. The results are saved to an `.npy` file in the `eval` directory. You can then run `jupyter notebook` and navigate to [`eval_summary.ipynb`](eval_summary.ipynb) to load the `.npy` files and generate tables and plots of the results.


### Algorithms supported

For different algorithms to test please use the following branches and training script : 

- DDPG: Branch: ddpg_final |  train script: ddpg.py | test script: evaluate_ddpg.py 
- DDQN: Branch: ddqn_final |  train script: train.py | test script: evaluate.py 
- MADDPG: Branch: maddpg_final |  train script: maddpg.py | test script: evaluate_maddpg.py

For DDQN and MADDPG, please install the custom machin library in this organisation 


## References

Adapted from: 
```
@inproceedings{wu2021spatial,
  title = {Spatial Intention Maps for Multi-Agent Mobile Manipulation},
  author = {Wu, Jimmy and Sun, Xingyuan and Zeng, Andy and Song, Shuran and Rusinkiewicz, Szymon and Funkhouser, Thomas},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2021}
}
```
