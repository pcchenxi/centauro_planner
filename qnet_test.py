import tensorflow as tf 
import numpy as np
import pickle
import os
import math


C_LR = 0.0001                # learning rate for critic
S_DIM = 2
min_batch_size = 200
batch_size = 200

C_UPDATE_STEP = 25

ep_dir='./batch/'

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.s_clip = tf.clip_by_value(self.tfs, -1, 1)
        # critic
        self.v = self._build_vnet('critic')

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage)) 

        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # optimizer
        # self.c_grads = tf.gradients(self.closs, c_params)

        # self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs, var_list=c_params)
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        #############################################################################################
        tf.summary.scalar('c loss', self.closs)
        self.merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('data/log', self.sess.graph)

        # print ('Loading Model...')
        # ckpt = tf.train.get_checkpoint_state('./model/rl/')
        # if ckpt and ckpt.model_checkpoint_path:
        #     self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        #     print ('loaded')
        # else:
        #     print ('no model file')  

    def random_select_sample(self, s, r, num):
        selected_index = np.random.choice(len(s), num)
        selected_s, selected_r = [], []
        for index in selected_index:
            selected_s.append(s[index])
            selected_r.append(r[index])
        return selected_s, selected_r

    def get_batch(self, s_mean, r_mean, s_std, r_std):
        list = os.listdir(ep_dir) # dir is your directory path
        number_files = len(list) -1

        batch_r, batch_s = [], []
        batch_size = 0
        while batch_size < min_batch_size:
            try:
                ep_i = np.random.randint(number_files)
                filename_r = ep_dir + 'return_' + str(ep_i)
                filename_s = ep_dir + 'state_' + str(ep_i)
                f_r = open(filename_r, 'rb')
                f_s = open(filename_s, 'rb')
            except:
                continue      
            r = pickle.load(f_r)
            s = pickle.load(f_s)
            if batch_size == 0:
                batch_r = r
                batch_s = s
            else:
                batch_r = np.concatenate((batch_r, r), axis = 0)
                batch_s = np.concatenate((batch_s, s), axis = 0)
            batch_size = batch_s.shape[0]

        # ################## normalize data #################
        # for i in range(batch_r.shape[0]):
        #     x = batch_s[i][1]
        #     y = batch_s[i][2]
        #     dist = math.sqrt(x*x+y*y)
        #     batch_r[i] = batch_r[i]/dist
        #     # batch_r[i] = (batch_r[i]-r_mean)/r_std*2 -1 
        #     # print(batch_r[i])

        #     batch_s[i][0] = (batch_s[i][0]-s_mean[0])/s_std[0] *2 -1
        #     for j in range(1, 5):
        #         batch_s[i][j] = (batch_s[i][j]-s_mean[1])/s_std[1] *2 -1

        #     print(batch_s[i], batch_r[i])
            
        for i in range(batch_r.shape[0]):
            batch_r[i] = (batch_r[i]+1)/2
            # batch_r[i] = (batch_r[i]-r_mean)/r_std *2 -1
            # print(batch_r[i])

            # # batch_s[i][0] = batch_s[i][0]*2 -1
            # for j in range(batch_s[i].shape[0]):
            #     # if batch_s[i][j] < -1:
            #     #     batch_s[i][j] = 1
            #     # if batch_s[i][j] > 1:
            #     #     batch_s[i][j] = 1
            #     batch_s[i][j] = (batch_s[i][j]-s_mean[j])/s_std[j] *2 -1

        #     # print(batch_s[i])

        return batch_s, batch_r


    def update(self, GLOBAL_EP, s_mean, r_mean, s_std, r_std):
        s, r = self.get_batch(s_mean, r_mean, s_std, r_std)
        # adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        a_loss, v_loss= [], []
        # update actor and critic in a update loop

        selected_s, selected_r = s, r
        for _ in range(C_UPDATE_STEP):
            selected_s, selected_r = self.random_select_sample(s, r, batch_size)
            feed_direct = {self.tfs: selected_s, self.tfs: selected_s, self.tfdc_r: selected_r}
            # value_before, loss_before = self.sess.run([self.v, self.closs], feed_direct)
            _, summary = self.sess.run([self.ctrain_op, self.merged], feed_direct)
            self.summary_writer.add_summary(summary, GLOBAL_EP)
            # GLOBAL_EP += 1

            # value, closs = self.sess.run([self.v, self.closs], feed_direct)

        # print ('return', r.flatten())
        # print('value before update', value_before.flatten())
        # print('value after update', value.flatten())
        # print('loss ', loss_before, closs)

        # self.saver.save(self.sess, './model/test/model.cptk') 

        # print(GLOBAL_EP, 'update with batch', len(s))

    def _build_vnet(self, name):
        # init = tf.random_normal_initializer(0., .001)
        init = tf.contrib.layers.xavier_initializer()

        activation = tf.nn.tanh
        with tf.variable_scope(name):
            lc = tf.layers.dense(self.s_clip, 16, activation, kernel_initializer=init, name='c_fc1')
            # lc = tf.layers.dense(lc, 16, activation, kernel_initializer=init, name='c_fc2')
            # lc = tf.layers.dense(lc, 16, activation, kernel_initializer=init, name='c_fc3')
            # lc = tf.layers.dense(lc, 32, activation, kernel_initializer=init, name='c_fc4')


            # lc = tf.layers.dense(lc, a_unit_num3, activation, kernel_initializer=init, name='c_fc3')
            
            v = 2 * tf.layers.dense(lc, 1, activation, name='value')
        return v

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


# with tf.device("/gpu:0"):
ppo = PPO()

# print ('Loading Model...')
# ckpt = tf.train.get_checkpoint_state('./q_net_log/')
# if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     print ('loaded')
# else:
#     print ('no model file')

def get_norm():
    list = os.listdir(ep_dir) # dir is your directory path
    number_files = len(list) -1

    batch_r, batch_s = [], []
    for i in range(number_files):
        try:
            ep_i = np.random.randint(number_files)
            filename_r = ep_dir + 'return_' + str(i)
            filename_s = ep_dir + 'state_' + str(i)
            f_r = open(filename_r, 'rb')
            f_s = open(filename_s, 'rb')
        except:
            continue      
        r = pickle.load(f_r)
        s = pickle.load(f_s)
        if i == 0:
            batch_r = r
            batch_s = s
        else:
            batch_r = np.concatenate((batch_r, r), axis = 0)
            batch_s = np.concatenate((batch_s, s), axis = 0)


    # for i in range(len(batch_r)):
        # if batch_r[i] < -1 or batch_r[i] > 1:
        #     print(batch_r[i])
    #     x = batch_s[i][1]
    #     y = batch_s[i][2]
    #     dist = math.sqrt(x*x+y*y)
    #     batch_r[i] = batch_r[i]/dist

    ################## normalize data #################
    s_mean = np.mean(batch_s, axis=0)
    s_std = np.std(batch_s, axis=0)
    s_max = np.amax(batch_s, axis=0)
    s_min = np.amin(batch_s, axis=0)

    r_mean = np.mean(batch_r)
    r_std = np.std(batch_r)   
    r_max = np.amax(batch_r)
    r_min = np.amin(batch_r)

    print(s_mean, s_std) 
    print(r_mean, r_std)   
    print('diff s', s_max-s_min)
    print('diff r',r_max-r_min)
    print('min s', s_min)
    print('min r', r_min, r_max)
    return s_min, r_min, s_max-s_min, r_max-r_min
    # return s_mean, r_mean, s_std, r_std

s_mean, r_mean, s_std, r_std = get_norm()
for i in range(300):
    print(i)
    ppo.update(i, s_mean, r_mean, s_std, r_std)

