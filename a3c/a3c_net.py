import tensorflow as tf
import tensorflow.contrib.layers as lays
from layers import spatial_softmax
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
                self.a_prob, self.v, self.keypoints = self._build_net(scope)
                self.a_params, self.c_params, self.encoder_params = self._get_params(scope)
                self.all_aprams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.keypoints = self._build_net(scope)
                self.a_params, self.c_params, self.encoder_params = self._get_params(scope)

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

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    self.ac_grads = tf.gradients(self.ac_loss, self.a_params + self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                    # auto encoder and decoder
                    self.pull_encoder_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.encoder_params, globalAC.encoder_params)]

                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

                    self.update_ac_op = OPT_A.apply_gradients(zip(self.ac_grads, globalAC.a_params+globalAC.c_params))
                    # self.update_ac_op = OPT_A.apply_gradients(zip(self.ac_grads, globalAC.a_params+globalAC.c_params+globalAC.encoder_params))
                    
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        # num_img = N_image_size*N_image_size
        # self.grid = tf.slice(self.s, [0, 0], [-1, num_img])
        # self.robot_state = tf.slice(self.s, [0, num_img], [-1, N_robot_state_size])
        # reshaped_grid = tf.reshape(self.grid,shape=[-1, N_image_size, N_image_size, 1]) 
        # reshaped_robot_state = tf.reshape(self.robot_state,shape=[-1, N_robot_state_size])  

        # with tf.variable_scope('auto'):
        #     with tf.variable_scope('encoder'):
        #         conv1 = lays.conv2d(reshaped_grid, 64, [7, 7], stride=2, padding='SAME', activation_fn=tf.nn.relu)
        #         conv2 = lays.conv2d(conv1,32, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.relu)
        #         conv3 = lays.conv2d(conv2, 32, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.relu)
        #         # conv3_normalized = conv3/max_v
        #         arg_max, softmax = spatial_softmax(conv3)  # 16 number
        #         shape = conv3.get_shape().as_list()

        #         divide = arg_max // shape[2]
        #         divide_int = tf.cast(divide, tf.int64)
        #         mod = arg_max - divide_int*shape[2]
        #         row_index = divide / shape[2] *2 -1 #tf.divide(arg_max, shape[2])
        #         col_index = mod / shape[2] *2 -1 #tf.mod(arg_max, shape[2])

        #         expected_xy = tf.concat([row_index, col_index], 2)

        #         arg_max_reshape = tf.reshape(arg_max, shape=[-1, 16])  
        #         expected_xy_reshape = tf.reshape(expected_xy, shape=[-1, 64])  
        #         expected_xy_float32 = tf.cast(expected_xy_reshape, tf.float32)

        # feature = tf.concat([expected_xy_float32, reshaped_robot_state], 1, name = 'concat')

        with tf.variable_scope('actor'):
            # feature = tf.layers.dense(feature, 256, tf.nn.relu6, kernel_initializer=w_init, name='feature_fc')

            l_a = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 50, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            # l_c = tf.layers.dense(feature, 1, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value

        return a_prob, v, self.s

    def _get_params(self, scope):
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/auto/encoder')

        return a_params, c_params, encoder_params

    # update function
    def update_ac_global(self, feed_dict):  # run by a local
        _, _, a_loss, v_loss, value = self.SESS.run([self.update_a_op, self.update_c_op, self.a_loss, self.c_loss, self.v], feed_dict)  # local grads applies to global net
        # _, a_loss, v_loss, value = self.SESS.run([self.update_ac_op, self.a_loss, self.c_loss, self.v], feed_dict)  # local grads applies to global net
        return a_loss, v_loss, value 
        
    # pull function
    def pull_ac_global(self):  # run by a local
        self.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def pull_auto_global(self):  # run by a local
        self.SESS.run(self.encoder_params)

    def choose_action(self, s):  # run by a local
        prob_weights = self.SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


