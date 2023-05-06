import remote
import socket
import select
import struct
import time
import os
import numpy as np
import utils
from remote import sim


sim.simxFinish(-1) # just in case, close all opened connections

# 端口号默认要用19997, 在remoteApiConnections有定义
clientID=sim.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
if clientID !=-1:
    print ('Connected to remote API server')
else:
    print ('Failed connecting to remote API server')


ret, targetObj = sim.simxGetObjectHandle(clientID, 'Sphere', 
                  sim.simx_opmode_blocking)
ret, arr = sim.simxGetObjectPosition(clientID, targetObj, -1, sim.simx_opmode_blocking)

if ret==sim.simx_return_ok:
    print(arr)
    
# sim.simxSetObjectPosition(clientID, targetObj,-1,(arr[0],arr[1] + 0.5,arr[2]), 
                #   sim.simx_opmode_blocking)
stop = sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
print('Stop the simulation')
time.sleep(4)#需要反应时间
start = sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
print('resume the simulation')

time.sleep(4)


sim.simxGetPingTime(clientID)
sim.simxFinish(clientID)