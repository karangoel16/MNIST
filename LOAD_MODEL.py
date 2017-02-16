
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np;
import pandas as pd;
BATCH_SIZE=50;
LEARNING_RATE = 1e-4
image_size=784
label=10;
image_width=28;
image_height=28;


# In[2]:

# weight initialization
def w_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0)
    return tf.Variable(initial)
def b_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def mpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise


# In[3]:

LOAD_FILE=input("Enter the name of the model to load the tensorflow model");
test=pd.read_csv('~/Documents/GIT_HUB/MNIST/test.csv').values
test=test.astype(np.float);
test=np.multiply(test,1.0/255.0);
std=tf.placeholder('float')
x = gaussian_noise_layer(tf.placeholder('float', shape=[None, image_size]),std);
#W=tf.placeholder('float',shape=[images.shape[1],10])
y_=tf.placeholder('float',shape=[None,label])
W_conv1 = w_variable([5, 5, 1, 32])
b_conv1 = b_variable([32])
image = tf.reshape(x, [-1,image_width , image_height,1])
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = mpool(h_conv1)
W_conv2 = w_variable([5, 5, 32, 64])
b_conv2 = b_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = mpool(h_conv2)
W_fc1 = w_variable([7 * 7 * 64, 1024])
b_fc1 = b_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#this is the output layer
W_fc2 = w_variable([1024, label])
b_fc2 = b_variable([label])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# optimisation function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,1)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
init = tf.initialize_all_variables()


# In[4]:

try:
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(LOAD_FILE+'.meta');
        saver.restore(sess,LOAD_FILE)
        predicted_values=np.zeros(shape=(test.shape[0]));
        for i in range(0,test.shape[0]//BATCH_SIZE):
            predicted_values[i*(BATCH_SIZE):(i+1)*BATCH_SIZE]=predict.eval(feed_dict={x:test[(i*BATCH_SIZE):(i+1)*BATCH_SIZE],keep_prob:1.0});
        print(predicted_values[10])
    np.savetxt('submission.csv', 
               np.c_[range(1,len(test)+1),predicted_values], 
               delimiter=',', 
               header = 'ImageId,Label', 
               comments = '', 
               fmt='%d')
except OSError as e:
    print('FILE DO NOT EXIST')


# In[ ]:




# In[ ]:



