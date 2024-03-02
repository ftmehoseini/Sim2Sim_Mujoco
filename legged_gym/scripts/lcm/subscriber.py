import isaacgym
import lcm
import torch
from obslcm import observ_t 
from obslcm import action_t
from policy_loader import WeightPolicy
import numpy as np
# 48 obs and 12 action / rlmpc

model = WeightPolicy()
print('-------------hi----------')
msg_action = action_t()
def my_handler(channel, data):
    global model
    msg_obs = observ_t.decode(data)
    print("desired obs dtype:", type(model.obs))
    print("shape observation", np.shape(model.obs) )
    # print("msg type: ", type(msg_obs))
    # print('shape msg:', np.shape(msg_obs))
    print('***done!***')
    
    model.obs = msg_obs
    action = model.step()
    msg_action.tau = action

    lc.publish("ACTION", msg_action.encode())

msg_obs = observ_t()
lc = lcm.LCM() # "udpm://224.0.55.55:5001?ttl=225"
subscription = lc.subscribe("OBSERVATION", my_handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass