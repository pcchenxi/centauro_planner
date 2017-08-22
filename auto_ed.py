import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
from environment import centauro_env
import tensorflow as tf


def get_subgrid(robot_x, robot_y, img):
    x = robot_x + centauro_env.map_shift
    y = robot_y + centauro_env.map_shift
    c_row = img.shape[0] - int(y/centauro_env.grid_size)
    c_col = int(x/centauro_env.grid_size)

    sub_start_r = 0
    sub_end_r = centauro_env.observation_pixel*2
    sub_start_c = 0
    sub_end_c = centauro_env.observation_pixel*2

    start_r = c_row - centauro_env.observation_pixel
    end_r = c_row + centauro_env.observation_pixel

    start_c = c_col - centauro_env.observation_pixel
    end_c = c_col + centauro_env.observation_pixel

    if start_r < 0:
        sub_start_r = -start_r
        start_r = 0
    if end_r >= img.shape[0]:
        sub_end_r = img.shape[0] - start_r - 1
        end_r = img.shape[0] -1

    if start_c < 0:
        sub_start_c = -start_c
        start_c = 0
    if end_c >= img.shape[1]:
        sub_end_c = img.shape[1] - start_c - 1
        end_c = img.shape[1] -1

    # print(x, y, c_row, c_col)
    # print(start_r, end_r, start_c, end_c)
    # print(sub_start_r, sub_end_r, sub_start_c, sub_end_c)
    obs_grid = np.zeros((centauro_env.observation_pixel*2, centauro_env.observation_pixel*2), np.float32)

    obs_grid[sub_start_r:sub_end_r, sub_start_c:sub_end_c] = img[start_r:end_r, start_c:end_c]

    return obs_grid 

def create_ds():
    img = cv2.imread('./data/auto/map.png', 0)
    counter = 0
    for x in np.arange(-2.5, 2.5, 0.1):
        for y in np.arange(-2.5, 2.5, 0.1):
            grid = get_subgrid(x, y, img)
            filename = str(counter)+'.png'
            cv2.imwrite('./data/auto/' + filename, grid)
            counter += 1
            # plt.clf()
            # plt.imshow(grid, cmap='gray')
            # # plt.show()
            # plt.pause(0.01)

def get_ds(percent):
    list = os.listdir('./data/auto/') 
    ds_size = len(list) -1
    training_size = int(ds_size*percent)
    index = np.random.choice(ds_size, training_size)
    grid_set = []
    for i in index:
        filename = './data/auto/' + str(i) + '.png'
        grid = cv2.imread(filename, 0)
        grid_set.append(grid)
        plt.clf()
        plt.imshow(grid, cmap='gray')
        # plt.show()
        plt.pause(0.01)
    
    print(len(grid_set))
    return grid_set

def get_batch(data_set, batch_size);
    index = np.random.choice(len(data_set), batch_size)
    batch = []
    for i in index:
        batch.append(data_set[i])
    return batch

grid_set = get_ds(0.7)

learning_rate = 0.01
training_epoch = 20
batch_size = 100

n_hidden = 256  
n_input = [30, 30, 1]   

# network
X = tf.placeholder(tf.float32, [None, n_input])