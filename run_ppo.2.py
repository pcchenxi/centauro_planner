import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
import math
import pickle, os

from environment import centauro_env

EP_MAX = 100000
EP_LEN = 100
N_WORKER = 1                # parallel workers
GAMMA = 0.98                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002                # learning rate for critic
All_LR = 0.0005
MIN_BATCH_SIZE = 300         #EP_LEN*N_WORKER         # minimum batch size for updating PPO
MINIBATCH_SIZE = int(MIN_BATCH_SIZE/2)
A_UPDATE_STEP = 3             # loop update operation n-steps
C_UPDATE_STEP = 3
UPDATE_STEP = 2
EPSILON = 0.5               # for clipping surrogate objective

###############################
observation_bound = 3

c_loss_weight = 0.5
a_loss_weight = 1

seperate_update = True

a_unit_num1 = 64
a_unit_num2 = 0
a_unit_num3 = 0

c_unit_num1 = 32
c_unit_num2 = 0
c_unit_num3 = 0

# tf.random_normal_initializer(0., .001)
# tf.contrib.layers.xavier_initializer()
# tf.zeros_initializer()
# tf.constant_initializer(1)
init = tf.zeros_initializer()
init_a = tf.zeros_initializer()
activation = tf.nn.tanh
activation_a = tf.nn.tanh
value_activation = tf.nn.sigmoid

S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space         # state and action dimension

ep_dir = './batch/'
# N_image_size = centauro_env.observation_image_size
# N_robot_state_size = centauro_env.observation_control

DISCONTED_GAMMA = [1]
d_g = 1
for r in range(EP_LEN+1):  
    d_g = GAMMA * d_g
    DISCONTED_GAMMA.append(d_g)

REWARD_GOAL = 0
REWARD_CRASH = -0.5

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
        global MINIBATCH_SIZE
        selected_index = np.random.choice(len(s), MINIBATCH_SIZE)
        selected_s, selected_a, selected_adv, selected_r = [], [], [], []
        for index in selected_index:
            selected_s.append(s[index])
            selected_a.append(a[index])
            selected_adv.append(adv[index])
            selected_r.append(r[index])
        return selected_s, selected_a, selected_adv, selected_r

    def update(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP, MINIBATCH_SIZE
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
                        selected_s, selected_a, selected_adv, selected_r = self.random_select_sample(s, a, r, adv)
                        feed_direct = {self.tfs: selected_s, self.tfa: selected_a, self.tfadv: selected_adv, self.tfdc_r: selected_r}
                        self.sess.run(self.atrain_op, feed_direct)

                    for _ in range(C_UPDATE_STEP):
                        selected_s, selected_a, selected_adv, selected_r = self.random_select_sample(s, a, r, adv)

                        feed_direct = {self.tfs: selected_s, self.tfa: selected_a, self.tfadv: selected_adv, self.tfdc_r: selected_r}
                        _, summary = self.sess.run([self.ctrain_op, self.merged], feed_direct)
                        # value, closs, summary = self.sess.run([self.v, self.closs, self.merged], feed_direct)
                else:
                    for _ in range(UPDATE_STEP):
                        selected_s, selected_a, selected_adv, selected_r = self.random_select_sample(s, a, r, adv)
                        feed_direct = {self.tfs: selected_s, self.tfa: selected_a, self.tfadv: selected_adv, self.tfdc_r: selected_r}
                        _, summary = self.sess.run([self.all_train_op, self.merged], feed_direct)

                ########################## save ep value #################################333
                # list = os.listdir(ep_dir) # dir is your directory path
                # number_files = int(len(list)/2)
                # with open(ep_dir + 'state_'+str(number_files), 'wb') as handle:
                #     pickle.dump(s, handle)
                # with open(ep_dir + 'return_'+str(number_files), 'wb') as handle:
                #     pickle.dump(r, handle)
      

                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available
                GLOBAL_EP += 1

                self.saver.save(self.sess, './model/rl/model.cptk') 

                self.summary_writer.add_summary(summary, GLOBAL_EP)
                print(GLOBAL_EP, 'update with batch', len(s), len(r))
                self.saver.save(self.sess, './model/rl/model.cptk') 

                summary = tf.Summary()
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
            
            v = tf.layers.dense(lc, 1, value_activation, name='value')
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
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, HISTORY_READY, HISTORY
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
                    GLOBAL_UPDATE_COUNTER -= len(buffer_r)
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
                # if done:
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done!= 0:
                    discounted_r = self.compute_return(buffer_r, buffer_s, done, s_)                           # compute discounted reward
                    buffer_s = self.process_state(buffer_s)

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]

                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                    
                    if done!= 0:
                        break

            # GLOBAL_EP += 1

    def compute_step_reward(self, buffer_r, s0, v_s_):
        global DISCONTED_GAMMA, REWARD_GOAL, REWARD_CRASH
        dist_s0 = math.sqrt(s0[0]*s0[0] + s0[1]*s0[1])
        reward_line = dist_s0
        ################################
        # reward_0 = 0
        norm_dist = dist_s0
        ################################
        reward_0 = np.exp(-dist_s0)
        norm_dist = 1

        disconted_rewards = [reward_0]

        for i in range(len(buffer_r)):
            if buffer_r[i] < reward_line:
                # print('     reward line:', reward_line)
                # reward = reward_line - buffer_r[i]
                reward = np.exp(-buffer_r[i]) - np.exp(-reward_line)
                dis_reward = reward * DISCONTED_GAMMA[i]
                disconted_rewards.append(dis_reward)
                reward_line = buffer_r[i]
                # print('     got reward', reward, dis_reward)
            # else:
            #     print('     no reward')
        
        disconted_rewards.append(v_s_ * DISCONTED_GAMMA[len(buffer_r)])
        value = np.sum(disconted_rewards)
        normalized_value = (value - REWARD_CRASH) / (norm_dist - REWARD_CRASH + REWARD_GOAL)
        if normalized_value > 1 or normalized_value < 0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(normalized_value, value, norm_dist, disconted_rewards, buffer_r, v_s_)
        # print(' last reward', v_s_, DISCONTED_GAMMA[len(buffer_r)-1], disconted_rewards[-1])
        # print(' sum reward', value)
        # print(' normalized', normalized_value, dist_s0)        
        return normalized_value

    def compute_return(self, buffer_r, buffer_s, done, s_):
        dist_v_s_ = math.sqrt(s_[0]*s_[0] + s_[1]*s_[1])

        if done == 1:
            v_s_ = REWARD_GOAL
        elif done == -1:
            v_s_ = REWARD_CRASH 
        else:
            # dist_v_s_ = 1 - np.exp(-dist_v_s_)
            # v_s_ = self.ppo.get_v(s_) * (dist_v_s_ - REWARD_CRASH + REWARD_GOAL) + REWARD_CRASH
            v_s_ = self.ppo.get_v(s_) * (1 - REWARD_CRASH + REWARD_GOAL) + REWARD_CRASH
            # print(v_s_, 'ppo s', self.ppo.get_v(s_) , 'dist', dist_v_s_)

        # print('reward:', buffer_r)
        normolized_r = []
        for i in range(len(buffer_r)):
            # print(' ',i, '--')
            normalized_value = self.compute_step_reward(buffer_r[i:], buffer_s[i], v_s_)
            normolized_r.append(normalized_value)

        # print(normolized_r, v_s_)
        # if done != 0:
            # print(normolized_r)
        print(dist_v_s_, 'ppo s', self.ppo.get_v(s_), normolized_r[-1], done)
        for i in range(0, len(buffer_s), 10):
            s = buffer_s[i]
            print ('    ppo', self.ppo.get_v(s), normolized_r[i], self.ppo.choose_action(s))
        return normolized_r

    def process_state(self, buffer_s):
        for i in range(len(buffer_s)):
            # buffer_s[i][0] = buffer_s[i][0]*2 -1
             for j in range(buffer_s[i].shape[0]):
                # buffer_s[i][0] = buffer_s[i][0]*2 -1
                buffer_s[i][j] = buffer_s[i][j]/observation_bound

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