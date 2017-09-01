import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
import math

from environment import centauro_env

EP_MAX = 900000000
EP_LEN = 150
N_WORKER = 4                # parallel workers
GAMMA = 0.99                 # reward discount factor
A_LR = 0.00005               # learning rate for actor
C_LR = 0.000005                # learning rate for critic
MIN_BATCH_SIZE = 500         # minimum batch size for updating PPO
UPDATE_STEP = 5             # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective

S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space         # state and action dimension

# N_image_size = centauro_env.observation_image_size
# N_robot_state_size = centauro_env.observation_control

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        self.v = self._build_vnet('critic')

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

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

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        aparams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='/pi')
        cparams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='/critic')
        self.a_grads = tf.gradients(self.aloss, aparams)
        self.c_grads = tf.gradients(self.closs, cparams)

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
                for _ in range(UPDATE_STEP):
                    _, aloss = self.sess.run([self.atrain_op, self.aloss], {self.tfs: s, self.tfa: a, self.tfadv: adv})
                for _ in range(UPDATE_STEP):
                    _, closs = self.sess.run([self.ctrain_op, self.closs], {self.tfs: s, self.tfdc_r: r})
                
                # agrad = self.sess.run(self.a_grads, {self.tfs: s, self.tfa: a, self.tfadv: adv})
                # vgrad = self.sess.run(self.c_grads, {self.tfs: s, self.tfdc_r: r})
                # print(agrad, vgrad)
                # [_, aloss = self.sess.run([self.atrain_op, self.aloss], {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                # [_, closs = self.sess.run([self.ctrain_op, self.closs], {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

                self.saver.save(self.sess, './model/rl/model.cptk') 
                summary = tf.Summary()

                summary.value.add(tag='Perf/EP return', simple_value=float(r[0]))
                summary.value.add(tag='Perf/Avg return', simple_value=float(np.mean(r)))
                summary.value.add(tag='Loss/A loss', simple_value=float(aloss))
                summary.value.add(tag='Loss/C loss', simple_value=float(closs))
                self.summary_writer.add_summary(summary, GLOBAL_EP)
                self.summary_writer.flush() 

    def _build_vnet(self, name):
        # init = tf.contrib.layers.xavier_initializer()
        init = tf.random_normal_initializer(0., .01)
        with tf.variable_scope(name):
            lc = tf.layers.dense(self.tfs, 64, tf.nn.tanh, kernel_initializer=init)
            # lc = tf.layers.dense(lc, 32, tf.nn.tanh, kernel_initializer=init)
            # lc = tf.layers.dense(lc, 10, tf.nn.tanh, kernel_initializer=init)
            
            v = tf.layers.dense(lc, 1)
        return v

    def _build_anet(self, name, trainable):
        # init = tf.contrib.layers.xavier_initializer()
        init = tf.random_normal_initializer(0., .01)
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 32, tf.nn.tanh, kernel_initializer=init, trainable=trainable)
            # l1 = tf.layers.dense(l1, 32, tf.nn.tanh, kernel_initializer=init, trainable=trainable)
            
            # l1 = tf.layers.dense(l1, 300, tf.nn.relu, trainable=trainable)

            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
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
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                    # break

                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                # if t == 0:
                #     print('first recieve', r)
                # if self.wid == 0:
                #     print(s)

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    v_s_ = self.ppo.get_v(s_)
                    if done:
                        v_s_ = 0                    
                    discounted_r = []                           # compute discounted reward

                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    
                    # for i in range(len(discounted_r)):
                    #     discounted_r[i] = discounted_r[i]/(i+1)

                    # print(buffer_r)
                    # print(discounted_r)
                    discounted_r.reverse()
                    # print(discounted_r)
                    
                    ############## scale return #####################
                    for i in range(len(buffer_s)):
                        dx = buffer_s[i][3]
                        dy = buffer_s[i][4]
                        dist = math.sqrt(dx*dx + dy*dy)
                        discounted_r[i] = discounted_r[i]/dist
                    # print(discounted_r)

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    sum_reward = np.sum(buffer_r[-1])
    
                    print('-----')
                    # if done:
                    print(GLOBAL_EP, 'sum reward:', sum_reward, discounted_r[0], self.ppo.get_v(s_), done)
                        # if abs(sum_reward) > 0.0001 and abs(sum_reward) != 1:
                        #     print(buffer_r)
                    # print(buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                        GLOBAL_EP += 1

                        # print(self.wid, 'update', discounted_r[0])

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
                    
                    if done:
                        break

            # GLOBAL_EP += 1


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