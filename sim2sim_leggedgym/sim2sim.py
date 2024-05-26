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

import os
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(LEGGED_GYM_ROOT_DIR)
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'sim2sim_leggedgym', 'envs')
from envs import IUSTRoughCfg
import torch
from algo.ppo.actor_critic import ActorCritic
from algo.ppo.ppo import PPO
from mujoco_py import load_model_from_xml, MjSim, MjViewer
#########################################
MODEL_XML = """
<?xml version="1.0" ?>
<mujoco model="legs">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true" meshdir="/home/lenovo/projects/cheetah_RL/IUST-Cheetah-Software-RL/RL/resource/meshes/" />
    <option integrator="RK4" timestep="0.01"/>
    <size njmax="500" nconmax="100" />
        <default>
            <joint armature="0.0" damping="0.0" limited="false"/>
            <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
        </default>
    <asset>
        <mesh name="trunk" file="trunk.stl" />
        <mesh name="hip" file="hip.stl" />
        <mesh name="thigh_mirror" file="thigh_mirror.stl" />
        <mesh name="calf" file="calf.stl" />
        <mesh name="thigh" file="thigh.stl" />

        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>

    </asset>
    <worldbody>

        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <!-- <geom name="ground" type="plane" conaffinity="1" pos="98 0 0" size="100 .8 .5" material="grid"/> -->
        <geom conaffinity="1" group="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
        

        <body name="base" pos="0.0 0.0 0.5">
            <joint type="free" name="floating_base"/>
            <geom type="mesh" mesh="trunk" group="1"/>

            <inertial pos="-0.0033599 -0.00084664 0.1336" mass="9.9576" diaginertia="0.089966 0.060002 0.056135" />
            <!-- <joint name="floating_base" type='free' limited='false'/> -->

            <site name='imu' size='0.01' pos='0.0 0 0.0'/>
            <body name="FR_hip" pos="0.13 -0.05 0">
                <inertial pos="-0.00266413 -0.0163358 2.49436e-05" quat="0.475134 0.521822 -0.477036 0.523818" mass="0.864993" diaginertia="0.00151807 0.00143717 0.000946744" />
                <joint name="FR_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom quat="0 1 0 0" type="mesh" group="1" mesh="hip" />

                <body name="FR_thigh" pos="0.06 -0.015 0">
                    <inertial pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="FR_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh_mirror" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->

                    <body name="FR_calf" pos="0 -0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="FR_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0.018 -0.22" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
            <body name="FL_hip" pos="0.13 0.05 0">
                <inertial pos="0.0227338 0.0102794 2.49436e-05" quat="0.415693 0.415059 0.572494 0.571993" mass="0.864993" diaginertia="0.00366077 0.00338628 0.000591358" />
                <joint name="FL_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="hip" />
                <!-- <geom size="0.046 0.02" quat="0.707107 0.707107 0 0" type="cylinder" rgba="0.12 0.15 0.2 1" />
                <geom size="0.041 0.016" pos="0.13 0.05 0" quat="0.707107 0.707107 0 0" type="cylinder" /> -->
                <body name="FL_thigh" pos="0.06 0.015 0">
                    <inertial pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="FL_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->
                    <body name="FL_calf" pos="0 0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="FL_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0 -0.24" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
            <body name="RR_hip" pos="-0.13 -0.05 0">
                <inertial pos="-0.0227338 -0.0102794 2.49436e-05" quat="0.415059 0.415693 0.571993 0.572494" mass="0.864993" diaginertia="0.00366077 0.00338628 0.000591358" />
                <joint name="RR_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom quat="0 0 0 -1" type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="hip" />
                <!-- <geom size="0.046 0.02" quat="0.707107 0.707107 0 0" type="cylinder" rgba="0.12 0.15 0.2 1" />
                <geom size="0.041 0.016" pos="-0.13 -0.05 0" quat="0.707107 0.707107 0 0" type="cylinder" /> -->
                <body name="RR_thigh" pos="-0.06 -0.015 0">
                    <inertial pos="-0.003237 0.022327 -0.027326" quat="0.999125 -0.00256393 -0.0409531 -0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="RR_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh_mirror" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->
                    <body name="RR_calf" pos="0 -0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="RR_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0 -0.24" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
            <body name="RL_hip" pos="-0.13 0.05 0">
                <inertial pos="-0.0227338 0.0102794 2.49436e-05" quat="0.572494 0.571993 0.415693 0.415059" mass="0.864993" diaginertia="0.00366077 0.00338628 0.000591358" />
                <joint name="RL_hip_joint" pos="0 0 0" axis="1 0 0" limited="true" range="-0.802851 0.802851" />
                <geom quat="0 0 1 0" type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="hip" />
                <!-- <geom size="0.046 0.02" quat="0.707107 0.707107 0 0" type="cylinder" rgba="0.12 0.15 0.2 1" />
                <geom size="0.041 0.016" pos="-0.13 0.05 0" quat="0.707107 0.707107 0 0" type="cylinder" /> -->
                <body name="RL_thigh" pos="-0.06 0.015 0">
                    <inertial pos="-0.003237 -0.022327 -0.027326" quat="0.999125 0.00256393 -0.0409531 0.00806091" mass="1.013" diaginertia="0.00555739 0.00513936 0.00133944" />
                    <joint name="RL_thigh_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0472 4.18879" />
                    <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="thigh" />
                    <!-- <geom size="0.0215 0.017 0.11" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" /> -->
                    <body name="RL_calf" pos="0 0.07125 -0.21183">
                        <inertial pos="0.00472659 0 -0.142595" quat="0.706906 0.0168526 0.0168526 0.706906" mass="0.226" diaginertia="0.00380047 0.00379113 3.53223e-05" />
                        <joint name="RL_calf_joint" pos="0 0 0" axis="0 1 0" limited="true" range="-2.69653 -0.916298" />
                        <geom type="mesh" group="1" rgba="0.12 0.15 0.2 1" mesh="calf" />
                        <!-- <geom size="0.008 0.008 0.1" pos="0 0 -0.1" quat="0.707107 0 0.707107 0" type="box" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.01" pos="0 0 -0.24" contype="0" conaffinity="0" group="1" rgba="0.12 0.15 0.2 1" />
                        <geom size="0.02" pos="0 0 -0.24" rgba="0.12 0.15 0.2 1" /> -->
                    </body>
                </body>
            </body>
    </body>
    </worldbody>

    <actuator>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RR_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RR_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RR_calf_joint" gear="150"/>

      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RL_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RL_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="RL_calf_joint" gear="150"/>

      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FL_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FL_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FL_calf_joint" gear="150"/>

      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FR_hip_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FR_thigh_joint" gear="150"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="FR_calf_joint" gear="150"/>  
    </actuator> 
</mujoco>
"""
#########################################

class cmd:
    vx = 0.0 # 0.4
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    # model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model = load_model_from_xml(MODEL_XML)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0:3] = eu_ang # base lin vel
            obs[0, 3:6] = omega # base ang vel
            obs[0, 6:9] = [0, 0, -9.8]
            obs[0, 9:10] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 10:11] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 11:12] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 12:24] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 24:36] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 36:48] = action

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale


        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    # parser.add_argument('--load_model', type=str, required=True,
    #                     help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(IUSTRoughCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = "/home/lenovo/sim_leggedgym/resources/robots/XBot/mjcf/XBot-L-terrain.xml"
            else:
                mujoco_model_path = "/home/lenovo/sim_leggedgym/resources/mjcf/quad.xml"
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)

    # /home/lenovo/sim_leggedgym
    # policy_net_file = "{LEGGED_GYM_ROOT_DIR}/logs/rough_iust/bestpolicy-10march.pt"
    path = "/home/lenovo/sim_leggedgym/logs/rough_iust/bestpolicy-10march.pt"
    # policy_network_path = policy_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
    loaded_dict = torch.load(path)
    actor_critic = ActorCritic(48, 48, 12,[512,256,128],[512,256,128])
    ppo = PPO(actor_critic = actor_critic)
    ppo.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    ppo.actor_critic.eval()
    ppo.actor_critic.to('cpu')
    
    policy = ppo
    # policy = torch.jit.load(policy_network_path)
    print('policy loaded!')
    # policy = torch.jit.load(args.load_model)

    run_mujoco(policy, Sim2simCfg())
