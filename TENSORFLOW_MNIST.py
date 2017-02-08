
# coding: utf-8

# In[1]:

import pandas as pd
import tensorflow as tf
import numpy as np

DROPOUT=0.5

TRAINING_ITERATIONS =200
LEARNING_RATE = 1e-4
Validation_size=1000
BATCH_SIZE=50




# In[2]:

data=pd.read_csv('~/Documents/GIT_HUB/MNIST/train.csv')
data.head()


# In[3]:

images=data.iloc[:,1:].values.astype(np.float);
images=np.multiply(images,1.0/255.0);
#this is done to change the shape of the array
print(images.shape);
image_size=images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print(image_width)


# In[4]:

labels_count=data[[0]].values.ravel()
label = np.unique(labels_count).shape[0]
print(label);


# In[5]:

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    res=np.zeros(shape=(num_labels,num_classes));
    for i in range (0,num_labels):
        res[i,labels_dense[i]]=1;
    return res;

print(labels_count.shape);
hot_label=dense_to_one_hot(labels_count,label);
hot_label = hot_label.astype(np.uint8)
print(hot_label[10])
print(labels_count[10])


# In[6]:

validation_images=images[:Validation_size]
validation_labels=hot_label[:Validation_size]

train_images=images[Validation_size:]
train_labels=hot_label[Validation_size:]
print(validation_images.shape)
print(train_images.shape)


# In[7]:

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[8]:

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


# In[9]:

x = tf.placeholder('float', shape=[None, image_size])
#W=tf.placeholder('float',shape=[images.shape[1],10])
y_=tf.placeholder('float',shape=[None,label])


# In[10]:

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

image = tf.reshape(x, [-1,image_width , image_height,1])


h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
#print (h_conv1.get_shape()) # => (40000, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (40000, 14, 14, 32)



# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#this is the output layer
W_fc2 = weight_variable([1024, label])
b_fc2 = bias_variable([label])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# optimisation function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,1)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
saver = tf.train.Saver()


# In[11]:

check=input('Enter do you wanna load file or run a new model');
if check=='yes' or check=='YES':
    LOAD_FILE=input("Enter the name of the model to load the tensorflow model");
    try:
        new_saver.restore(sess, LOAD_FILE)
    except :
        print('File Do not exist')
else:
    sess.run(init)
    SAVING_FILE=input("Enter the name of the model to save the tensorflow");
# evaluation


# In[12]:

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(50) 
    if(i%100==0):
        print(i)
        #print(sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1.0}))
        if(Validation_size):#this is done to make sure that in case we make validation size 0 , we can simply avoid printing it
            print(sess.run(accuracy,feed_dict={x:validation_images,y_:validation_labels,keep_prob:1.0}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
if check=='Yes' or check=='yes':
    saver.save(sess,LOAD_FILE);
else:
    saver.save(sess, SAVING_FILE);#this is where we save our training model which can be used later


# In[ ]:



