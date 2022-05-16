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

We provide pretrained policies for each test environment. The `download-pretrained.sh` script will download the pretrained policies and save their configs and network weights into the `logs` and `checkpoints` directories, respectively. Use the following command to run it:

```bash
./download-pretrained.sh
```

You can then use `enjoy.py` to run a pretrained policy in the simulation environment. Here are a few examples you can try:

```bash

# 4 pushing robots
python enjoy.py --config-path logs/20201214T092814688334-pushing_4-small_divider-ours/config.yml
python enjoy.py --config-path logs/20201217T171253620771-pushing_4-large_empty-ours/config.yml
```


## Training in the Simulation Environment

The [`config/experiments`](config/experiments) directory contains the template config files used for all experiments in the paper. To start a training run, you can provide one of the template config files to the `train.py` script. For example, the following will train a policy on the `SmallDivider` environment:

```bash
python train.py config/experiments/ours/pushing_4-small_divider-original.yml
```

The training script will create a log directory and checkpoint directory for the new training run inside `logs/` and `checkpoints/`, respectively. Inside the log directory, it will also create a new config file called `config.yml`, which stores training run config variables and can be used to resume training or to load a trained policy for evaluation.


### Evaluation

Trained policies can be evaluated using the `evaluate.py` script, which takes in the config path for the training run. For example, to evaluate the `SmallDivider` pretrained policy, you can run:

```
python evaluate.py --config-path logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml
```

This will load the trained policy from the specified training run, and run evaluation on it. The results are saved to an `.npy` file in the `eval` directory. You can then run `jupyter notebook` and navigate to [`eval_summary.ipynb`](eval_summary.ipynb) to load the `.npy` files and generate tables and plots of the results.


### Algorithms supported

For different algorithms to test please use the following branches and training script : 

- DDPG: Branch: ddpg_final |  train script: ddpg.py | test script: evaluate_ddpg.py 
- DDQN: Branch: ddqn_final |  train script: train.py | test script: evaluate.py 
- MADDPG: Branch: maddpg_final |  train script: maddpg.py | test script: evaluate_maddpg.py

For DDQN and MADDPG, please install the custom machin library in this organisation 

## Citation

If you find this work useful for your research, please consider citing:

```
@inproceedings{wu2021spatial,
  title = {Spatial Intention Maps for Multi-Agent Mobile Manipulation},
  author = {Wu, Jimmy and Sun, Xingyuan and Zeng, Andy and Song, Shuran and Rusinkiewicz, Szymon and Funkhouser, Thomas},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2021}
}
```
