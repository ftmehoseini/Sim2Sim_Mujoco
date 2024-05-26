# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


# from humanoid import LEGGED_GYM_ROOT_DIR
import os
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'sim2sim_leggedgym', 'envs')
print(LEGGED_GYM_ENVS_DIR,'***',LEGGED_GYM_ROOT_DIR)
# import cv2
import numpy as np
from isaacgym import gymapi
import isaacgym
from envs import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import time


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = True 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    env_cfg.domain_rand.randomize_friction = False

    
    train_cfg.seed = 1 #123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()
    pri_obs = env.get_privileged_observations()
    # print('pri observation:',pri_obs)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # for i in tqdm(range(stop_state_log)):

    for i in range(10*int(env.max_episode_length)):
        p_time=time.time()
        actions = policy(obs.detach())
        if FIX_COMMAND:
            env.commands[:, 0] = 1.   # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.
        # print(actions,'****actions*****')
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        # print('obs',np.shape(obs)) #720
        # print('critic_obs',np.shape(critic_obs)) #210
        rate = time.time() - p_time
        # print('rate:' ,rate)

        # print('#######################################################')


        # logger.log_states(
        #     {
        #         'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #         'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #         'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #         'dof_torque': env.torques[robot_index, joint_index].item(),
        #         'command_x': env.commands[robot_index, 0].item(),
        #         'command_y': env.commands[robot_index, 1].item(),
        #         'command_yaw': env.commands[robot_index, 2].item(),
        #         'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #         'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #         'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #         'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #         'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        #     }
        #     )
    #     # ====================== Log states ======================
    #     if infos["episode"]:
    #         num_episodes = torch.sum(env.reset_buf).item()
    #         if num_episodes>0:
    #             logger.log_rewards(infos["episode"], num_episodes)

    # logger.print_rewards()
    # logger.plot_states()


if __name__ == '__main__':
    FIX_COMMAND = True
    args = get_args()
    play(args)
