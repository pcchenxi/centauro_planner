import numpy as np 
import os
import matplotlib.pyplot as plt
import cv2
from environment import centauro_env

import tensorflow as tf
import tensorflow.contrib.layers as lays

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
    # # plt.imshow(img, cmap='gray')
    # plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    # plt.show()

    counter = 0
    for x in np.arange(-2.5, 2.5, 0.1):
        for y in np.arange(-2.5, 2.5, 0.1):
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
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  4 x 4 x 8
    encoded = lays.conv2d(inputs, 32, [5, 5], stride=1, padding='SAME')
    encoded = lays.conv2d(encoded, 64, [4, 4], stride=1, padding='SAME')
    encoded = lays.conv2d(encoded, 64, [3, 3], stride=1, padding='SAME')
    encoded_flat = tf.contrib.layers.flatten(encoded)  # 32*32*64
    encoded_fc = tf.layers.dense(inputs=encoded_flat, units=60, activation=tf.nn.relu6, name = 'encoded_fc')

    # decoder
    # 4 x 4 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    decoded_fc = tf.layers.dense(inputs=encoded_fc, units=60*60*64, activation=tf.nn.relu6, name = 'decoded_fc')
    encoded_reshape = tf.reshape(decoded_fc, shape=[-1, 60, 60, 64])  
    decoded = lays.conv2d_transpose(encoded_reshape, 64, [3, 3], stride=1, padding='SAME')
    decoded = lays.conv2d_transpose(decoded, 32, [4, 4], stride=1, padding='SAME')
    decoded = lays.conv2d_transpose(decoded, 1, [5, 5], stride=1, padding='SAME', activation_fn=tf.nn.tanh)

# upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return encoded_fc, decoded


####################################################################################

batch_size = 500  # Number of samples in each batch
epoch_num = 9999999     # Number of epochs to train the network
lr = 0.0001        # Learning rate

# ae_inputs = tf.placeholder(tf.float32, (None, 60, 60))  # input to the network (MNIST images)
# ae_inputs_reshape = tf.reshape(ae_inputs, shape=[-1, 60, 60, 1])  

# encoded_fc, decoded = autoencoder(ae_inputs_reshape)  # create the Autoencoder network

# # calculate the loss and optimize the network
# loss = tf.reduce_mean(tf.square(decoded - ae_inputs_reshape))  # claculate the mean square error loss
# train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()

saving_model = False

with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.import_meta_graph('./model/grid_feature_model.cptk.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model'))
    graph = tf.get_default_graph()

    img_input = graph.get_tensor_by_name("Placeholder:0")
    reconstruct_loss = graph.get_tensor_by_name("Mean:0")
    reconstruct_op = graph.get_tensor_by_name("Conv2d_transpose_2/Tanh:0")

    summary_writer = tf.summary.FileWriter('data/log', sess.graph)
    saver = tf.train.Saver(max_to_keep=5)

    grid_set = get_ds()

    # split trianing and testing data setD
    data_train, data_test, _, _ = train_test_split(grid_set, grid_set, test_size=0.3)

    for ep in range(epoch_num): 
        batch_test = get_batch(data_test, 1)
        loss_test, decoded_img = sess.run([reconstruct_loss, reconstruct_op], feed_dict={img_input: batch_test})
        print(loss_test)
        plt.clf()
        plt.figure(1)
        plt.imshow(batch_test[0], cmap='gray')
        plt.figure(2)
        plt.imshow(decoded_img[0,:,:,0], cmap='gray')
        plt.pause(1)

    # for ep in range(epoch_num):  # epochs loop
    #     batch_img = get_batch(data_train, batch_size)
    #     batch_test = get_batch(data_test, batch_size)
    #     _, loss_v, decoded_img = sess.run([train_op, loss, decoded], feed_dict={ae_inputs: batch_img})
    #     loss_test = sess.run(loss, feed_dict={ae_inputs: batch_test})
    #     # print(encoded_img[0,:,:,0].max(), encoded_img[0,:,:,0].shape)
    #     if ep % 100 == 0:
    #         plt.clf()
    #         plt.figure(1)
    #         plt.imshow(batch_img[0], cmap='gray')
    #         plt.figure(2)
    #         plt.imshow(decoded_img[0,:,:,0], cmap='gray')
    #         plt.pause(0.01)
    #         if saving_model:
    #             saver.save(sess, './model/grid_feature_model.cptk') 

    #     print('Epoch: {} - cost= {:.5f}'.format((ep + 1), loss_v), loss_test)
    #     summary = tf.Summary()
    #     summary.value.add(tag='Loss/loss Train', simple_value=float(loss_v))
    #     summary.value.add(tag='Loss/loss Test', simple_value=float(loss_test))
    #     summary_writer.add_summary(summary, ep)
    #     summary_writer.flush() 
