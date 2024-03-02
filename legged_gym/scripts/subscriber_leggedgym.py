import os
import isaacgym
import lcm
import torch
from obslcm import observ_t 
from obslcm import action_t
from policy_leggedgym import *
import numpy as np
import time
# 48 obs and 12 action / leggedgym

device = 'cpu'
# ***PUBLISH ACTION***
msg_action = action_t() 
msg_obslist = []
def my_handler(channel, data):
    global obs, msg_obslist, device
    msg_obs = observ_t.decode(data)
    t_p = time.time()
    print("Received message on channel \"%s\"" % channel)
    # print(" q = %s" % str(msg_obs.q))
    # print(" qdot = %s" % str(msg_obs.qdot))
    # print(" orientation = %s" % str(msg_obs.orientation))
    # print(" pre_action = %s" % str(msg_obs.pre_action))
    # print(" cmd_vel = %s" % str(msg_obs.cmd_vel))
    msg_obslist.append(msg_obs.q)
    msg_obslist.append(msg_obs.qdot)
    msg_obslist.append(msg_obs.orientation)
    msg_obslist.append(msg_obs.pre_action)
    msg_obslist.append(msg_obs.cmd_vel)
    msg_obsarray = np.concatenate(msg_obslist).ravel() # shape (48,)
    obs = torch.tensor([msg_obsarray], requires_grad=False, dtype = torch.float, device = device)
    # obs.to(device)
    print('--------------------------------------------------------------------------------')
    print('observations:',msg_obsarray)
    print('--------------------------------------------------------------------------------')
    # action = policy(obs.detach()) 
    action = Policy(obs.detach()) # shape (12,)
    # print(np.shape(action))
    msg_action.tau = action
    msg_obslist = []
    # msg_action.tau = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print('Transfer Time: {:.5f}'.format(time.time()- t_p))
    lc.publish("ACTION", msg_action.encode())
    print('actions:', action)
    print('################################################################################')

# ***SUBSCRIBE OBSERVATION***
msg_obs = observ_t()                                  
lc = lcm.LCM() # "udpm://224.0.55.55:5001?ttl=225"
subscription = lc.subscribe("OBSERVATION", my_handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass