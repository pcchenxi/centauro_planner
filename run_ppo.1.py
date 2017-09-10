import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
import math
import pickle, os

from environment import centauro_env

EP_MAX = 100000
EP_LEN = 60
N_WORKER = 4                # parallel workers
GAMMA = 0.98                 # reward discount factor
A_LR = 0.00005               # learning rate for actor
C_LR = 0.0001                # learning rate for critic
All_LR = 0.0005
MIN_BATCH_SIZE = 500         #EP_LEN*N_WORKER         # minimum batch size for updating PPO
A_UPDATE_STEP = 5             # loop update operation n-steps
C_UPDATE_STEP = 5
UPDATE_STEP = 2
EPSILON = 0.2               # for clipping surrogate objective

###############################
observation_bound = 3

c_loss_weight = 0.5
a_loss_weight = 1

seperate_update = True

a_unit_num1 = 200
a_unit_num2 = 100
a_unit_num3 = 0

c_unit_num1 = 100
c_unit_num2 = 100
c_unit_num3 = 0

# tf.random_normal_initializer(0., .001)
# tf.contrib.layers.xavier_initializer()

init = tf.contrib.layers.xavier_initializer()
init_a = tf.contrib.layers.xavier_initializer()
activation = tf.nn.relu
activation_a = tf.nn.relu

S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space         # state and action dimension

ep_dir = './batch/'
# N_image_size = centauro_env.observation_image_size
# N_robot_state_size = centauro_env.observation_control

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        self.feature = self.tfs #self._build_featurenet('state_feature')
        # critic
        self.v = self._build_vnet('critic')

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage)) 

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        #total loss
        self.total_loss = self.closs*c_loss_weight + self.aloss*a_loss_weight

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        f_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='state_feature')

        # optimizer
        # self.a_grads = tf.gradients(self.aloss, a_params)
        # self.c_grads = tf.gradients(self.closs, c_params)

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss, var_list=a_params)
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs, var_list=c_params)
        # self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        # self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        self.all_train_op = tf.train.AdamOptimizer(All_LR).minimize(self.total_loss)

        #############################################################################################
        # tf.summary.histogram("a grad", self.a_grads)
        # tf.summary.histogram("c grad", self.c_grads)
        tf.summary.scalar('a loss', self.aloss)
        tf.summary.scalar('c loss', self.closs)
        # tf.summary.histogram("real r", self.tfdc_r)
        # tf.summary.histogram("predict r", self.v)

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

    def random_select_sample(self, s, a, r, adv):
        selected_index = np.random.choice(len(s), int(len(s)/3))
        selected_s, selected_a, selected_adv, selected_r = [], [], [], []
        for index in selected_index:
            selected_s.append(s[index])
            selected_a.append(a[index])
            selected_adv.append(adv[index])
            selected_r.append(r[index])
        return selected_s, selected_a, selected_adv, selected_r

    def update(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                # print(len(s))
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                a_loss, v_loss= [], []
                # update actor and critic in a update loop
                if seperate_update:
                    selected_s, selected_a, selected_adv, selected_r = s, a, adv, r
                    for _ in range(A_UPDATE_STEP):
                        # selected_s, selected_a, selected_adv, selected_r = self.random_select_sample(s, a, r, adv)
                        feed_direct = {self.tfs: selected_s, self.tfa: selected_a, self.tfadv: selected_adv, self.tfs: selected_s, self.tfdc_r: selected_r}
                        
                        action_before = self.sess.run(self.sample_op, {self.tfs: selected_s})[-1]
                        self.sess.run(self.atrain_op, feed_direct)
                        action_after = self.sess.run(self.sample_op, {self.tfs: selected_s})[-1]

                    for _ in range(C_UPDATE_STEP):
                        # selected_s, selected_a, selected_adv, selected_r = self.random_select_sample(s, a, r, adv)
                        feed_direct = {self.tfs: selected_s, self.tfa: selected_a, self.tfadv: selected_adv, self.tfs: selected_s, self.tfdc_r: selected_r}
                        value_before, loss_before = self.sess.run([self.v, self.closs], feed_direct)
                        self.sess.run(self.ctrain_op, feed_direct)
                        
                        value, closs, summary = self.sess.run([self.v, self.closs, self.merged], feed_direct)
                else:
                    for _ in range(UPDATE_STEP):
                        selected_s, selected_a, selected_adv, selected_r = self.random_select_sample(s, a, r, adv)
                        feed_direct = {self.tfs: selected_s, self.tfa: selected_a, self.tfadv: selected_adv, self.tfs: selected_s, self.tfdc_r: selected_r}
                        _, summary = self.sess.run([self.all_train_op, self.merged], feed_direct)

                # ########################## save ep value #################################333
                # list = os.listdir(ep_dir) # dir is your directory path
                # number_files = len(list)
                # with open(ep_dir + 'state_'+str(number_files), 'wb') as handle:
                #     pickle.dump(s, handle)
                # with open(ep_dir + 'return_'+str(number_files), 'wb') as handle:
                #     pickle.dump(r, handle)

                # print ('return', r.flatten())
                # print('value before update', value_before.flatten())
                # print('value after update', value.flatten())
                # print('loss ', loss_before, closs)
                # print('action ', action_before, action_after)           
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

                self.saver.save(self.sess, './model/rl/model.cptk') 

                self.summary_writer.add_summary(summary, GLOBAL_EP)
                print(GLOBAL_EP, 'update with batch', len(s), len(r))
                self.saver.save(self.sess, './model/rl/model.cptk') 

                summary = tf.Summary()
                # summary.value.add(tag='Perf/EP return', simple_value=float(r[0]))
                summary.value.add(tag='Perf/Avg return', simple_value=float(np.mean(r)))
                self.summary_writer.add_summary(summary, GLOBAL_EP)
                self.summary_writer.flush() 

    def _build_featurenet(self, name):
        with tf.variable_scope(name):
            # self.robot_state = tf.slice(self.tfs, [0, 0], [-1, 2])
            # self.obs = tf.slice(self.tfs, [0, 2], [-1, -1])

            # obs_f = tf.layers.dense(self.obs, 16, activation, kernel_initializer=init, name='obs_fc')
            # state_f = tf.layers.dense(self.obs, 16, activation, kernel_initializer=init, name='state_fc')

            # feature = tf.concat([obs_f, state_f], 1, name = 'concat')

            # feature = tf.layers.dense(feature, 32, activation, kernel_initializer=init, name='f_fc')
            # feature = tf.layers.dense(feature, 32, activation, kernel_initializer=init, name='f_fc2')


            #############################
            feature = tf.layers.dense(self.tfs, 64, activation, kernel_initializer=init, name='f_fc')
            # feature = tf.layers.dense(feature, 32, activation, kernel_initializer=init, name='f_fc2')
            return feature


    def _build_vnet(self, name):
        with tf.variable_scope(name):
            if c_unit_num1 == 0:
                lc = self.feature
            else:
                lc = tf.layers.dense(self.feature, c_unit_num1, activation, kernel_initializer=init, name='c_fc1')
            
            if a_unit_num2 != 0:
                lc = tf.layers.dense(lc, c_unit_num2, activation, kernel_initializer=init, name='c_fc2')
            if a_unit_num3 != 0:
                lc = tf.layers.dense(lc, c_unit_num3, activation, kernel_initializer=init, name='c_fc3')
            
            v = tf.layers.dense(lc, 1, name='value')
        return v

    def _build_anet(self, name, trainable):
        # init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(name):
            if a_unit_num1 == 0:
                l1 = self.feature
            else:
                l1 = tf.layers.dense(self.feature, a_unit_num1, activation_a, kernel_initializer=init_a, trainable=trainable, name='a_fc1')

            if c_unit_num2 != 0:
                l1 = tf.layers.dense(l1, a_unit_num2, activation_a, kernel_initializer=init_a, trainable=trainable, name='a_fc2')
            if c_unit_num3 != 0:
                l1 = tf.layers.dense(l1, a_unit_num3, activation_a, kernel_initializer=init_a, trainable=trainable, name='a_fc3')

            mu = tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable, name='mu')
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable, name='sigma')

            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -1, 1)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, env, wid):
        self.wid = wid
        self.env = env
        self.ppo = GLOBAL_PPO


    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset(EP_LEN)
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            # for t in range(EP_LEN):
            start = -1
            start_s = []
            end_s = []
            # while(True):
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                    # break

                a = self.ppo.choose_action(s)
                # print(s)
                end_s = s
                if start == -1:
                    start_s = s
                    start = 1
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                # if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                # if done:
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    
                    v_s_ = self.ppo.get_v(s_)    
                    if done:
                        v_s_ = 0        

                    # print('s before:')
                    # print(buffer_s[0])
                    
                    discounted_r = self.compute_return(buffer_r, buffer_s, v_s_)                           # compute discounted reward
                    buffer_s = self.process_state(buffer_s)

                    # print('s after:')
                    # print(buffer_s[0])

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
    
                    # s_obs1 = np.asarray([ 0.78, 0.60825562, -1.19020414,  0.05742053, -0.35558999])
                    # s_obs2 = np.asarray([ 0.12, -1.29276776,  0.47612619, -0.34833077,  0.14743352])
                    # s_g = np.asarray([ 0.02, 0.10751012,  0.21840394, -0.42156056,  1.06698167])
                    # s_g2 = np.asarray([ 0.02, 0.20556368, -0.06315947, -0.08847831,  0.89263314])

                    # print('--------------test obs', self.ppo.get_v(s_obs1), self.ppo.get_v(s_obs2)) 
                    # print('--------------test goal', self.ppo.get_v(s_g), self.ppo.get_v(s_g2)) 
                    # print('reward', buffer_r)

                    mean_reward = np.mean(buffer_r)
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Mean reward', simple_value=float(mean_reward))
                    self.ppo.summary_writer.add_summary(summary, GLOBAL_EP)
                    self.ppo.summary_writer.flush() 

                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update
                        GLOBAL_EP += 1

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                    
                    if done:
                        break

            # GLOBAL_EP += 1

    def compute_return(self, buffer_r, buffer_s, v_s_):
        discounted_r = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)                    
        discounted_r.reverse()

        # print('r before:')
        # print(discounted_r)

        normolized_r = a = np.array(discounted_r)
        s_dist = []
        for i in range(len(buffer_s)):
            dx = buffer_s[i][1]
            dy = buffer_s[i][2]
            dist = math.sqrt(dx*dx + dy*dy)
            s_dist.append(dist)

        for i in range(len(buffer_s)):
            normolized_r[i] = discounted_r[i]/s_dist[i]
            # if normolized_r[i] > 1:
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     print(i, normolized_r[i], discounted_r[i], dist)
            #     print(buffer_s[i])
            #     print(buffer_r)
            #     print(np.sum(buffer_r))
            #     print(discounted_r)
            #     print(s_dist)

        return discounted_r

    def process_state(self, buffer_s):
        for i in range(len(buffer_s)):
            buffer_s[i][0] = buffer_s[i][0]*2 -1
            buffer_s[i][1] = buffer_s[i][1]/observation_bound
            buffer_s[i][2] = buffer_s[i][2]/observation_bound
            buffer_s[i][3] = buffer_s[i][3]/observation_bound
            buffer_s[i][4] = buffer_s[i][4]/observation_bound

        return buffer_s



if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out

    workers = []
    for i in range(N_WORKER):
        env = centauro_env.Simu_env(20000 + i)
        workers.append(Worker(env, i))
    # workers = [Worker(wid=i) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 1, 1
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    
    for worker in workers:          # worker threads
        # job = lambda: worker.work(saver, summary_writer)
        # t = threading.Thread(target=job)
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    # # plot reward change and test
    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    # env = gym.make('Pendulum-v0')
    # while True:
    #     s = env.reset()
    #     for t in range(300):
    #         env.render()
    #         s = env.step(GLOBAL_PPO.choose_action(s))[0]