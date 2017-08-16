import sys, os, math
# from rllab.spaces import Box, Discrete
import numpy as np
import time
## v-rep
from environment.vrep_plugin import vrep
import pickle as pickle

print ('import env vrep')

observation_space = 9 # 8 joints 1 target height
action_space = 8
target_height = 0.5

terrain_map = np.zeros((100, 100), np.float32)

class Simu_env():
    def __init__(self, port_num):
        # self.action_space = ['l', 'f', 'r', 'h', 'e']

        self.port_num = port_num
        self.dist_pre = 100

        self.path_used = 1
        self.step_inep = 0
        self.object_num = 0
        self.game_level = 3
        self.succed_time = 0
        self.pass_ep = 1
        self.ep_reap_time = 0

        self.connect_vrep()
        # self.reset()

    #@property
    #def observation_space(self):
    #    return Box(low=-np.inf, high=np.inf, shape=(1, 182))

    #@property
    #def action_space(self):
    #    return Discrete(len(action_list))

    # def get_terrain_map(self):
    #     collection_hd = vrep.simxGetCollectionHandle(self.clientID, vrep.simx_opmode_blocking)
    #     obstacles_hds = simGetCollectionObjects(obstacle_low_hd)

    def convert_state(self, laser_points, path):
        path = np.asarray(path)
        laser_points = np.asarray(laser_points)
        state = np.append(laser_points, path)
        state = state.flatten()

        # state = np.asarray(path)
        # state = state.flatten()
        return state

    def reset(self):
        # time.sleep(2)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.5)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.5)

        self.step_inep = 0

        res, retInts, retFloats, retStrings, retBuffer = self.call_sim_function('centauro', 'reset', [self.game_level])        
        state, reward, is_finish, info = self.step([0, 0, 0, 0, 0, 0, 0, 0])
        return state

    def step(self, action):
        self.step_inep += 1

        _, _, _, _, found_pose = self.call_sim_function('centauro', 'step', action)

        robot_state = self.get_robot_state()
        # print (robot_state)
        state_ = robot_state[4:]

        #compute reward and is_finish
        reward, is_finish = self.compute_reward(robot_state, target_height, found_pose)

        state_.append(target_height)
        state_ = np.asarray(state_)

        return state_, reward, is_finish, ''

    def compute_reward(self, robot_state, target_height, found_pose):
        is_finish = False
        reward = 0
        current_h = robot_state[2]
        dist = abs(current_h - target_height)

        # if abs(current_h - target_height) < 0.01:
        if dist < self.dist_pre:
            reward = 0.1
        else:
            reward = -0.1
            # is_finish = True

        self.dist_pre = dist

        if dist < 0.01:
            reward = 10
            is_finish = True            

        if found_pose == bytearray(b"f"):       # when collision or no pose can be found
            is_finish = True 
            reward = -10
        return reward, is_finish



    ####################################  interface funcytion  ###################################

    def get_robot_state(self):
        _, _, state, _, _ = self.call_sim_function('centauro', 'get_robot_state')
        return state

    def connect_vrep(self):

        clientID = vrep.simxStart('127.0.0.1', self.port_num, True, True, 5000, 5)
        if clientID != -1:
            print ('Connected to remote API server with port: ', self.port_num)
        else:
            print ('Failed connecting to remote API server with port: ', self.port_num)


        self.clientID = clientID

    def disconnect_vrep(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)
        vrep.simxFinish(self.clientID)
        print ('Program ended')


    ########################################################################################################################################
    ###################################   interface function to communicate to the simulator ###############################################
    def call_sim_function(self, object_name, function_name, input_floats=[]):
        inputInts = []
        inputFloats = input_floats
        inputStrings = []
        inputBuffer = bytearray()
        res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID, object_name,vrep.sim_scripttype_childscript,
                    function_name, inputInts, inputFloats, inputStrings,inputBuffer, vrep.simx_opmode_blocking)

        # print 'function call: ', self.clientID
        return res, retInts, retFloats, retStrings, retBuffer


# env = Simu_env(20000)
# action = [0,0,0,0,0,0,0,0]
# for i in range(1000):
#     for j in range(8):
#         a = (np.random.rand()-0.5) * 2
#         action[j] = a

#     s_, r, done, _ = env.step(action)
#     print (s_, r, done)
# print (env.action_space())
# print (env.observation_space())