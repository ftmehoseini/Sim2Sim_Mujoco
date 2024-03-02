import sys
sys.path.append("/home/lenovo/gym_legged/rsl_rl/legged_gym")
import os
import isaacgym
from legged_gym.envs import *
import numpy as np
import torch
import numpy as np
from modules import ActorCritic
from algorithms import PPO
import time

device = "cpu"
path  = "/home/lenovo/legged_gym/logs/rough_iust/Feb27_16-22-51_/model_0.pt"
print(f"Loading model from: {path}")
loaded_dict = torch.load(path)
# print("model I loaded: ...")
# print(loaded_dict['model_state_dict'].keys())
actor_critic = ActorCritic(48, 48, 12,[512,256,128],[512,256,128])
ppo = PPO(actor_critic = actor_critic)
# print("model I imported: ....")
# print(ppo.actor_critic)
ppo.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
ppo.actor_critic.eval()
ppo.actor_critic.to(device)

def Policy(obs):
    # obs = np.zeros(2 * 48, dtype=np.float32)  
    # obs = obs.reshape((2, 48))
    # obs_tensor = torch.from_numpy(obs).to(device)
    obs_critic = obs
    t_start = time.time()
    actions = ppo.act(obs, obs_critic)
    print("Model Inference Time: {:.5f}".format(time.time()-t_start))
    # actions = policy(obs.detach())
    return actions.detach().numpy()[0]
    