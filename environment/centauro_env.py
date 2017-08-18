import sys, os, math
# from rllab.spaces import Box, Discrete
import numpy as np
import time
## v-rep
from environment.vrep_plugin import vrep
import pickle as pickle
import cv2
import matplotlib.pyplot as plt

print ('import env vrep')

map_shift = 2.5
observation_range = 1.5
grid_size = 0.05 

observation_pixel = int(observation_range/grid_size)

observation_space = 8 #[observation_pixel, observation_pixel, 8]  
action_space = 5

class Simu_env():
    def __init__(self, port_num):
        self.port_num = port_num
        self.dist_pre = 100

        self.path_used = 1
        self.step_inep = 0
        self.object_num = 0
        self.game_level = 3
        self.succed_time = 0
        self.pass_ep = 1
        self.ep_reap_time = 0
        self.terrain_map = np.zeros((100, 100), np.float32)

        self.connect_vrep()
        self.get_terrain_map()
        # self.reset()

    #@property
    #def observation_space(self):
    #    return Box(low=-np.inf, high=np.inf, shape=(1, 182))

    #@property
    #def action_space(self):
    #    return Discrete(len(action_list))

    def convert_state(self, robot_state, obs_grid):
        observation = obs_grid.flatten()
        state = robot_state[2:]  # theta, h, l. tx, ty, ttheta, th, tl
        state = np.asarray(state)
        # print(len(state))

        # state = np.append(obs_grid, state)
        # state = state.flatten()

        return state

    def reset(self):
        # time.sleep(2)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.5)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)

        self.step_inep = 0

        res, retInts, retFloats, retStrings, retBuffer = self.call_sim_function('centauro', 'reset', [self.game_level])        
        state, reward, is_finish, info = self.step([0, 0, 0, 0, 0])
        return state

    def step(self, action):
        self.step_inep += 1

        _, _, _, _, found_pose = self.call_sim_function('centauro', 'step', action)
        _, _, robot_state, _, _ = self.call_sim_function('centauro', 'get_robot_state') # x, y, theta, h, l,   ////   tx, ty t_theta, th, tl
        # print(len(robot_state))
        obs_grid = self.get_observation_gridmap(robot_state[0], robot_state[1])

        #compute reward and is_finish
        reward, is_finish = self.compute_reward(robot_state, found_pose)

        state_ = self.convert_state(robot_state, obs_grid)

        return state_, reward, is_finish, ''

    def compute_reward(self, robot_state, found_pose):
        # x, y, theta, h, l,   ////   tx, ty t_theta, th, tl
        is_finish = False
        reward = 0
        current_h = robot_state[3]
        target_h = robot_state[8]
        dist = abs(current_h - target_h)

        if dist < 0.01:
            reward = 1
            is_finish = True            

        if found_pose == bytearray(b"f"):       # when collision or no pose can be found
            is_finish = True 
            reward = -1

        # if abs(robot_state[0]) > 2 or abs(robot_state[1]) > 2:
        #     is_finish = True
            
        return reward, is_finish



    ####################################  interface funcytion  ###################################

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

    def get_observation_gridmap(self, robot_x, robot_y):
        x = robot_x + map_shift
        y = robot_y + map_shift
        c_row = self.terrain_map.shape[0] - int(y/grid_size)
        c_col = int(x/grid_size)

        sub_start_r = 0
        sub_end_r = observation_pixel*2
        sub_start_c = 0
        sub_end_c = observation_pixel*2

        start_r = c_row - observation_pixel
        end_r = c_row + observation_pixel

        start_c = c_col - observation_pixel
        end_c = c_col + observation_pixel

        if start_r < 0:
            sub_start_r = -start_r
            start_r = 0
        if end_r >= self.terrain_map.shape[0]:
            sub_end_r = self.terrain_map.shape[0] - start_r - 1
            end_r = self.terrain_map.shape[0] -1

        if start_c < 0:
            sub_start_c = -start_c
            start_c = 0
        if end_c >= self.terrain_map.shape[1]:
            sub_end_c = self.terrain_map.shape[1] - start_c - 1
            end_c = self.terrain_map.shape[1] -1

        # print(x, y, c_row, c_col)
        # print(start_r, end_r, start_c, end_c)
        # print(sub_start_r, sub_end_r, sub_start_c, sub_end_c)
        self.obs_grid = np.zeros((observation_pixel*2, observation_pixel*2), np.float32)
        self.obs_grid[sub_start_r:sub_end_r, sub_start_c:sub_end_c] = self.terrain_map[start_r:end_r, start_c:end_c]

        return self.obs_grid 

    def get_terrain_map(self):
        # self.terrain_map
        _, _, obstacle_info, _, _ = self.call_sim_function('centauro', 'get_obstacle_info')
        for i in range(0, len(obstacle_info), 5):
            x = obstacle_info[i+0] + map_shift
            y = obstacle_info[i+1] + map_shift

            if x >= 5 or x <= 0:
                continue
            if y >= 5 or y <= 0:
                continue
            r = obstacle_info[i+2]
            h = obstacle_info[i+4]

            row = self.terrain_map.shape[0] - int(y/grid_size)
            col = int(x/grid_size)
            radius = int(r/grid_size )
            height = int(255/0.5 * h )
        
            self.terrain_map = cv2.circle(self.terrain_map, (col,row), radius, height, -1)
        cv2.line(self.terrain_map, (0, 0), (0, self.terrain_map.shape[1]), 255, 4)
        cv2.line(self.terrain_map, (0, 0), (self.terrain_map.shape[0], 0), 255, 4)
        cv2.line(self.terrain_map, (0, self.terrain_map.shape[1]), (self.terrain_map.shape[0], self.terrain_map.shape[1]), 255, 4)
        cv2.line(self.terrain_map, (self.terrain_map.shape[0], 0), (self.terrain_map.shape[0], self.terrain_map.shape[1]), 255, 4)

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
# env.reset()
# env.get_terrain_map()
# img = env.get_observation_gridmap(0, 0)
# plt.imshow(env.obs_grid, cmap='gray')
# plt.imshow(env.terrain_map, cmap='gray')
# plt.show()

# action = [0,0,0,0,0.1]
# for i in range(100):
#     for j in range(5):
#         a = (np.random.rand()-0.5) * 2
#         action[j] = a

#     s_, r, done, _ = env.step(action)
#     print (r, done)

# print (env.action_space())
# print (env.observation_space())