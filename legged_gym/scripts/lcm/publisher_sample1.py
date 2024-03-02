# this script is a test for input/output of policy
import lcm
from obslcm import observ_t
from obslcm import action_t
import numpy as np

msg_obs = observ_t()

msg_obs.q = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
msg_obs.qdot = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
msg_obs.orientation = (1,2,3,4,5,6)
msg_obs.pre_action = (6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0,6.0)
msg_obs.cmd_vel = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

lc = lcm.LCM() # "udpm://224.0.55.55:5001?ttl=225"
lc.publish("OBSERVATION", msg_obs.encode())

def my_handler(channel, data):
    msg_action = observ_t.decode(data)
    print("desired action dtype:", type(msg_action))
    print("shape action", np.shape(msg_action) )
    print('***done!***')

msg_action = action_t()
lc = lcm.LCM() # "udpm://224.0.55.55:5001?ttl=225"
subscription = lc.subscribe("ACTION", my_handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass