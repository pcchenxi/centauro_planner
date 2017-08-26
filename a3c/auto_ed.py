import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2
from environment import centauro_env

import tensorflow as tf
import layers as lays

from sklearn.model_selection import train_test_split



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
    img = np.load('./data/auto/map.npy')
    # print(img)
    # plt.figure(1)
    # plt.imshow(img, cmap='gray')
    # # plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    # plt.show()

    counter = 0
    for x in np.arange(-2.5, 2.5, 0.03):
        for y in np.arange(-2.5, 2.5, 0.03):
            grid = get_subgrid(x, y, img)
            filename = str(counter)+'.npy'
            np.save('./data/auto/' + filename, grid)
            counter += 1
            # plt.clf()
            # plt.imshow(grid, cmap='gray')
            # # plt.show()
            # plt.pause(0.01)

def get_ds():
    list = os.listdir('./data/auto/') 
    ds_size = len(list) -1

    grid_set = []
    for i in range(ds_size):
        filename = './data/auto/' + str(i) + '.npy'
        grid = np.load(filename)
        grid_set.append(grid)
        # grid_set.append(grid.flatten())
        # plt.clf()
        # plt.imshow(grid, cmap='gray')
        # plt.show()
        # plt.pause(0.01)
    
    print(len(grid_set))
    return grid_set

def get_batch(data_set, batch_size):
    data_range = int(len(data_set))
    index = np.random.choice(data_range, batch_size)
    batch = []
    for i in index:
        batch.append(data_set[i])
    return batch

# create_ds()

#############################################################################333

def autoencoder(inputs):
    with tf.variable_scope('Global_Net'):
        with tf.variable_scope('auto'):
            with tf.variable_scope('encoder'):
                encoded = lays.conv2d(inputs, 32, [5, 5], stride=1, padding='SAME')
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

# upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return encoded_fc, decoded

# def spatial_softmax(encoded):
#     # shape = encoded.get_shape().as_list()

#     layer_flatten = tf.contrib.layers.flatten(encoded)
#     max_index = tf.argmax(layer_flatten, 1)

#     return max_index

def encoder_spatial_softmax(inputs):
    with tf.variable_scope('Global_Net'):
        with tf.variable_scope('auto'):
            with tf.variable_scope('encoder'):
                conv1 = lays.conv2d(inputs, 64, [7, 7], stride=1, padding='SAME', activation_fn=tf.nn.relu)
                conv2 = lays.conv2d(conv1,32, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.relu)
                conv3 = lays.conv2d(conv2, 32, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.relu)
                # conv3_normalized = conv3/max_v
                arg_max, softmax = lays.spatial_softmax(conv3)  # 16 number
                shape = conv3.get_shape().as_list()

                row_index = (arg_max // shape[2]) / shape[2] *2 -1 #tf.divide(arg_max, shape[2])
                col_index = (arg_max % shape[2]) / shape[2] *2 -1 #tf.mod(arg_max, shape[2])

                expected_xy = tf.concat([row_index, col_index], 2)

                arg_max_reshape = tf.reshape(arg_max, shape=[-1, 16])  
                expected_xy_reshape = tf.reshape(expected_xy, shape=[-1, 64])  

    return conv3, softmax

img_size = 15
def decoder(spatial_softmax):
    with tf.variable_scope('Global_Net'):
        with tf.variable_scope('auto'):
            with tf.variable_scope('decoder'):
                decoded = tf.layers.dense(inputs=spatial_softmax, units=img_size*img_size*1, activation=None, name = 'decoded_fc')
                decoded = tf.reshape(decoded, shape=[-1, img_size, img_size, 1])  
                # decoded = lays.conv2d_transpose(decoded, 32, [3, 3], stride=1, padding='SAME')
                # decoded = lays.conv2d_transpose(decoded, 64, [3, 3], stride=1, padding='SAME')
                # decoded = lays.conv2d_transpose(decoded, 1, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.tanh)                

    return decoded

####################################################################################

def draw_keypoint(key_points):
    img_size = 60
    img = np.zeros((img_size, img_size), np.float32)
    for i in range(0, len(key_points), 2):
        key_col = (key_points[i] + 1)/2
        key_row = (key_points[i+1] + 1)/2
        row = int(key_col * img_size)
        col = int(key_row * img_size)
        # print(row, col)
        # print (key_points[i], key_points[i+1])
        img[row, col] = 1
    
    return img


batch_size = 500  # Number of samples in each batch
epoch_num = 9999999     # Number of epochs to train the network
lr = 0.0001        # Learning rate

ae_inputs = tf.placeholder(tf.float32, (None, 60, 60))  # input to the network (MNIST images)
ae_inputs_reshape = tf.reshape(ae_inputs, shape=[-1, 60, 60, 1])  
resize_input = tf.image.resize_images(ae_inputs_reshape, size=(img_size,img_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# resize_input = tf.layers.max_pooling2d(inputs=ae_inputs_reshape, pool_size=[4, 4], strides=4)

key_points_input = tf.placeholder(tf.float32, (None, 64))

conv3, key_points = encoder_spatial_softmax(ae_inputs_reshape)  # create the Autoencoder network
decoded = decoder(key_points_input)

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(decoded - resize_input))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

saving_model = True

with tf.Session() as sess:
    sess.run(init)

    # saver = tf.train.import_meta_graph('./model/grid_feature_model.cptk.meta')
    # saver.restore(sess,tf.train.latest_checkpoint('./model'))
    # graph = tf.get_default_graph()

    # img_input = graph.get_tensor_by_name("Placeholder:0")
    # reconstruct_loss = graph.get_tensor_by_name("Mean:0")
    # reconstruct_op = graph.get_tensor_by_name("Conv2d_transpose_2/Tanh:0")

    summary_writer = tf.summary.FileWriter('data/log', sess.graph)

    saver = tf.train.Saver()
    # print ('Loading Model...')
    # ckpt = tf.train.get_checkpoint_state('./model/')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     print ('loaded')
    # else:
    #     print ('no model file')  

    grid_set = get_ds()

    # split trianing and testing data setD
    data_train, data_test, _, _ = train_test_split(grid_set, grid_set, test_size=0.1)

    # for ep in range(epoch_num): 
    #     batch_test = get_batch(data_test, 1)
    #     loss_test, decoded_img = sess.run([reconstruct_loss, reconstruct_op], feed_dict={img_input: batch_test})
    #     print(loss_test)
    #     plt.clf()
    #     plt.figure(1)
    #     plt.imshow(batch_test[0], cmap='gray')
    #     plt.figure(2)
    #     plt.imshow(decoded_img[0,:,:,0], cmap='gray')
    #     plt.pause(2)

    for ep in range(epoch_num):  # epochs loop
        batch_img = get_batch(data_train, batch_size)
        batch_test = get_batch(data_test, batch_size)
        conv3_img, points = sess.run([conv3, key_points], feed_dict={ae_inputs: batch_img})
        decoded_img, loss_v, _ = sess.run([decoded, loss, train_op], feed_dict={ae_inputs: batch_img, key_points_input: points})

        # loss_test = sess.run(loss, feed_dict={ae_inputs: batch_test})
        # print(encoded_softmax)
        # print(arg_max)
        # print(encoded_img[0,:,:,0].max(), encoded_img[0,:,:,0].shape)
        if ep % 50 == 0:
            img = draw_keypoint(points[0])
            plt.clf()
            plt.figure(1)
            plt.imshow(batch_img[0], cmap='gray')
            plt.figure(2)
            plt.imshow(conv3_img[0,:,:,0], cmap='gray')
            plt.figure(3)
            plt.imshow(decoded_img[0,:,:,0], cmap='gray')    
            plt.figure(4)
            plt.imshow(img, cmap='gray')                       
            plt.pause(0.01)
            # plt.show()
            if saving_model:
                saver.save(sess, './model/spatial_softmax.cptk') 

        print('Epoch: {} - cost= {:.5f}'.format((ep + 1), loss_v))
        summary = tf.Summary()
        summary.value.add(tag='Loss/loss Train', simple_value=float(loss_v))
        # summary.value.add(tag='Loss/loss Test', simple_value=float(loss_test))
        summary_writer.add_summary(summary, ep)
        summary_writer.flush() 
