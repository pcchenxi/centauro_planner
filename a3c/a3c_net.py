import tensorflow as tf
import tensorflow.contrib.layers as lays
import numpy as np
from environment import centauro_env

N_S = centauro_env.observation_space
N_A = centauro_env.action_space
N_image_size = centauro_env.observation_image_size
N_robot_state_size = centauro_env.observation_control

OPT_A = tf.train.RMSPropOptimizer(0.001, name='RMSPropA')
OPT_C = tf.train.RMSPropOptimizer(0.001, name='RMSPropC')
OPT_Auto = tf.train.RMSPropOptimizer(0.0001, name='RMSPropAuto')

GLOBAL_NET_SCOPE = 'Global_AC'
ENTROPY_BETA = 0.001

class ACNet(object):
    def __init__(self, SESS, scope, globalAC=None):
        self.SESS = SESS
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_prob, self.v, self.ori_grid, self.decoded_grid = self._build_net(scope)
                self.a_params, self.c_params, self.auto_params, self.encoder_params = self._get_params(scope)
                self.all_aprams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                # self.pull_auto_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.auto_params, globalAC.auto_params)]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.ori_grid, self.decoded_grid = self._build_net(scope)
                self.a_params, self.c_params, self.auto_params, self.encoder_params = self._get_params(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = tf.stop_gradient(-tf.reduce_sum(self.a_prob * tf.log(self.a_prob),
                                                              axis=1, keep_dims=True))  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('ac_loss'):
                    self.ac_loss = self.c_loss*0.5 + self.a_loss

                with tf.name_scope('auto_ed_loss'):
                    self.auto_ed_loss = tf.reduce_mean(tf.square(self.decoded_grid - self.ori_grid))

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    self.ac_grads = tf.gradients(self.ac_loss, self.a_params + self.c_params)
                    self.auto_grids = tf.gradients(self.auto_ed_loss, self.auto_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                    # auto encoder and decoder
                    self.pull_encoder_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.encoder_params, globalAC.encoder_params)]
                    self.pull_auto_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.auto_params, globalAC.auto_params)]

                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

                    self.update_ac_op = OPT_A.apply_gradients(zip(self.ac_grads, globalAC.a_params+globalAC.c_params))
                    # self.update_ac_op = OPT_A.apply_gradients(zip(self.ac_grads, globalAC.a_params+globalAC.c_params+globalAC.encoder_params))

                    self.update_auto_op = OPT_Auto.apply_gradients(zip(self.auto_grids, globalAC.auto_params))
                    
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        num_img = N_image_size*N_image_size
        self.grid = tf.slice(self.s, [0, 0], [-1, num_img])
        self.robot_state = tf.slice(self.s, [0, num_img], [-1, N_robot_state_size])
        reshaped_grid = tf.reshape(self.grid,shape=[-1, N_image_size, N_image_size, 1]) 
        reshaped_robot_state = tf.reshape(self.robot_state,shape=[-1, N_robot_state_size])  

        with tf.variable_scope('auto'):
            with tf.variable_scope('encoder'):
                encoded = lays.conv2d(reshaped_grid, 32, [5, 5], stride=1, padding='SAME')
                encoded = lays.conv2d(encoded, 64, [4, 4], stride=1, padding='SAME')
                encoded = lays.conv2d(encoded, 64, [3, 3], stride=1, padding='SAME')
                encoded_flat = tf.contrib.layers.flatten(encoded)  # 32*32*64
                encoded_fc = tf.layers.dense(inputs=encoded_flat, units=60, activation=tf.nn.relu6, name = 'encoded_fc')
            with tf.variable_scope('decoder'):
                decoded_fc = tf.layers.dense(inputs=encoded_fc, units=60*60*64, activation=tf.nn.relu6, name = 'decoded_fc')
                encoded_reshape = tf.reshape(decoded_fc, shape=[-1, 60, 60, 64])  
                decoded = lays.conv2d_transpose(encoded_reshape, 64, [3, 3], stride=1, padding='SAME')
                decoded = lays.conv2d_transpose(decoded, 32, [4, 4], stride=1, padding='SAME')
                decoded = lays.conv2d_transpose(decoded, 1, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.tanh)

        # concat
        feature = tf.concat([encoded_fc, reshaped_robot_state], 1, name = 'concat')

        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(feature, 100, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(feature, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        return a_prob, v, reshaped_grid, decoded

    def _get_params(self, scope):
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/auto/encoder')
        auto_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/auto')

        return a_params, c_params, auto_params, encoder_params

    # update function
    def update_ac_global(self, feed_dict):  # run by a local
        # _, _, a_loss, v_loss, total_loss, auto_loss = self.SESS.run([self.update_a_op, self.update_c_op, self.a_loss, self.c_loss, self.ac_loss, self.auto_ed_loss], feed_dict)  # local grads applies to global net
        _, a_loss, v_loss, total_loss, auto_loss = self.SESS.run([self.update_ac_op, self.a_loss, self.c_loss, self.ac_loss, self.auto_ed_loss], feed_dict)  # local grads applies to global net
        return a_loss, v_loss, total_loss, auto_loss 
        
    def update_auto_global(self, feed_dict):  # run by a local
        print('in update')
        loss = self.SESS.run(self.auto_ed_loss, feed_dict)  # local grads applies to global net
        # _, loss, decoded = self.SESS.run([self.update_auto_op, self.auto_ed_loss, self.decoded_grid], feed_dict)  # local grads applies to global net
        print('finish update')
        return loss, decoded
    # pull function
    def pull_ac_global(self):  # run by a local
        self.SESS.run([self.pull_a_params_op, self.pull_c_params_op, self.pull_encoder_params_op])

    def pull_auto_global(self):  # run by a local
        self.SESS.run(self.pull_auto_params_op)


    def choose_action(self, s):  # run by a local
        prob_weights = self.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


