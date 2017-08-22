
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.layers as lays

from sklearn.model_selection import train_test_split

from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 500  # Number of samples in each batch
epoch_num = 5     # Number of epochs to train the network
lr = 0.001        # Learning rate

    # # encoder
    # # 32 x 32 x 1   ->  16 x 16 x 32
    # # 16 x 16 x 32  ->  8 x 8 x 16
    # # 8 x 8 x 16    ->  4 x 4 x 8
    # conv1 = lays.conv2d(inputs, 32, [3, 3], stride=1, padding='SAME')
    # maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # conv2 = lays.conv2d(maxpool1, 16, [3, 3], stride=1, padding='SAME')
    # encoded = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # # conv3 = lays.conv2d(maxpool2, 8, [3, 3], stride=1, padding='SAME')
    # # encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # # decoder
    # # 4 x 4 x 8    ->  8 x 8 x 16
    # # 8 x 8 x 16   ->  16 x 16 x 32
    # # 16 x 16 x 32  ->  32 x 32 x 1
    # # upsample1 = tf.image.resize_images(encoded, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # # conv4 = lays.conv2d_transpose(upsample1, 16, [3, 3], stride=1, padding='SAME')
    # upsample2 = tf.image.resize_images(encoded, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # conv5 = lays.conv2d_transpose(upsample2, 32, [3, 3], stride=1, padding='SAME')
    # upsample3 = tf.image.resize_images(conv5, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # decoded = lays.conv2d_transpose(upsample3, 1, [3, 3], stride=1, padding='SAME', activation_fn=tf.nn.tanh)



def autoencoder(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    encoded = lays.conv2d(inputs, 1, [5, 5], stride=2, padding='SAME')
    encoded = lays.conv2d(encoded, 16, [5, 5], stride=2, padding='SAME')
    encoded = lays.conv2d(encoded, 8, [5, 5], stride=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    encoded = lays.conv2d_transpose(encoded, 16, [5, 5], stride=4, padding='SAME')
    encoded = lays.conv2d_transpose(encoded, 32, [5, 5], stride=2, padding='SAME')
    decoded = lays.conv2d_transpose(encoded, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)

    return decoded

def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network
# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# initialize the network
init = tf.global_variables_initializer()



# read MNIST dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# calculate the number of batches per epoch
batch_per_ep = mnist.train.num_examples // batch_size
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):  # epochs loop
        for batch_n in range(batch_per_ep):  # batches loop
            batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
            batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
            batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
            _, c, decoded_img = sess.run([train_op, loss, ae_outputs], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
            print(batch_img[0,:,:,0].min(), batch_img[0,:,:,0].max())
            # plt.clf()
            # # plt.figure(1)
            # # plt.imshow(encoded_img[0,:,:,0], cmap='gray')
            # plt.figure(2)
            # plt.imshow(decoded_img[0,:,:,0], cmap='gray')
            # plt.figure(3)
            # plt.imshow(batch_img[0,:,:,0], cmap='gray')
            # plt.pause(0.01)

    # test the trained network
    batch_img, batch_label = mnist.test.next_batch(50)
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    # plot the reconstructed images and their ground truths (inputs)
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images')
    for i in range(50):
        plt.subplot(5, 10, i+1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    plt.show()
