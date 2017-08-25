import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from a3c import a3c_net

MAX_GLOBAL_EP = 100000
UPDATE_GLOBAL_ITER = 50
GAMMA = 0.9
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

auto_epoch_num = 1000
auto_batch_size = 100

def get_ds():
    list = os.listdir('./data/auto/') 
    ds_size = len(list) -1

    grid_set = []
    for i in range(ds_size):
        filename = './data/auto/' + str(i) + '.npy'
        grid = np.load(filename)
        robot_state = np.zeros(8, np.float32)
        state = np.append(grid.flatten(), robot_state)
        grid_set.append(state)
        # print(state.shape)

    print(len(grid_set))
    return grid_set

def get_batch(data_set, batch_size):
    data_range = int(len(data_set))
    index = np.random.choice(data_range, batch_size)
    batch = []
    for i in index:
        batch.append(data_set[i])
    return batch

class Worker(object):
    def __init__(self, sess, name, env, summary_writer, globalAC):
        self.env = env
        self.name = name
        self.AC = a3c_net.ACNet(sess, name, globalAC)
        self.sess = sess
        self.summary_writer = summary_writer

    def write_summary(self, saver, sum_reward, mean_return, c_loss, a_loss, auto_loss):
        global GLOBAL_EP
        print(GLOBAL_EP)
        saver.save(self.sess, './data/model.cptk') 
        # if self.name == 'W_0':
        summary = tf.Summary()

        summary.value.add(tag='Perf/Sum reward', simple_value=float(sum_reward))
        summary.value.add(tag='Perf/Avg return', simple_value=float(mean_return))
        summary.value.add(tag='Loss/V loss', simple_value=float(c_loss))
        summary.value.add(tag='Loss/A loss', simple_value=float(a_loss))
        summary.value.add(tag='Loss/auto loss', simple_value=float(auto_loss))
                
        # summary.value.add(tag='Losses/loss', simple_value=float(loss))
        # summary.histogram.add(tag='Losses/start_prob', simple_value=float(start_prob))
        self.summary_writer.add_summary(summary, GLOBAL_EP)
        self.summary_writer.flush()  

    def work(self, saver):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        buffer_s, buffer_a, buffer_r = [], [], []
        while GLOBAL_EP < MAX_GLOBAL_EP:
            self.AC.pull_ac_global()
            self.AC.pull_auto_global()            
            s = self.env.reset()
            ep_r = 0
            total_step = 1
            GLOBAL_EP += 1
            while True:
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)

                # decoded_grid = self.sess.run(self.AC.decoded_grid, {self.AC.s: s[np.newaxis, :]})
                # plt.clf()
                # plt.imshow(decoded_grid[0,:,:,0], cmap='gray')
                # plt.pause(0.001)

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    a_loss, c_loss, total_loss, auto_loss = self.AC.update_ac_global(feed_dict)
                    # self.AC.pull_ac_global()
                    # self.AC.pull_auto_global()

                    sum_reward = np.sum(buffer_r)
                    mean_return = np.mean(buffer_r)
                    self.write_summary(saver, sum_reward, mean_return, c_loss, a_loss, auto_loss)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    print('updated', total_loss)
                    break

                s = s_
                total_step += 1
                # if done:
                #     break

