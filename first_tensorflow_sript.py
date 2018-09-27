import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import minmax_scale

# Default
df = pd.read_csv('./Data/converted_database.csv',)

# Custom Names
#df = pd.read_csv('./Data/converted_database.csv', names=['Name', 'Tester', 'Trial', 'Length', 'Right Hand',
#                                                         'Data - Values', 'Noise', 'Bias'])
# No Headers
#df = pd.read_csv('./Data/converted_database.csv', header=None)
#df

# attempting to convert data column(bytes) to  str using utf-8
# df['data'].str.decode('str-8')

#print(df['data'].dtype)
#print(df)


# This is preparing the data for the neural network
# the database conatins 20 motion gestures
#
gesture_names = ['v_shape', 'x_shape', 'horizontal_circle', 'vertical_circle']
# Pull out the columns for the x (data to train with) and y (value to predict)
X_training = df.drop('data', axis=1).values
Y_training = df.drop['name'].values


# laod testing data set from the csv
test_data_df = pd.read_csv('cir_ver_cclk.csv')  # circle vertical counter clockwise


X_testing = test_data_df('data', axis=1).values
Y_testing = test_data_df['name'].values

# all data needs to be scled to a small range like 0 to 1
# for neural network to work well. Create scalars for the input and outputs
# this is the veactor normalisation that we need to do
X_scalar = minmax_scale(feature_range=(0,1))
Y_scalar = minmax_scale(feature_range=(0,1))

# scale both the training inputs and outputs
X_scaled_training = X_scalar.fit_transform(X_training)
Y_scaled_training = Y_scalar.fit_transform(Y_training)

# it's very important that the training and test data are scaled with the same scalar
X_scaled_testing = X_scalar.transform(X_testing)
Y_scaled_testing = Y_scalar.transfrom(Y_testing

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

print("Note: Y values were scaled by multiplying {0.10f} and adding {:.4f".format(Y_scalar.scale_[0], Y_scalar.scale_[0]))


# creating the model
learning_rate = 0.001
learning_epochs = 100
display_ste = 5

# Define how many inputs and outputs in our neural network
# This is a many to many problem
number_of_input = 10 # placeholder
number_of_output = 4 # 4 motion gesture to recognise

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

#input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_input)) #none = batches of any size

# Layer 1
with tf.variable_scope('layer'):
    weights = tf.get_variable(name="weights1", shape=[number_of_input, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer )
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer)
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# layer 2

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer )
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer )
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer)
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# output layer
with tf.variable_scope('outoput'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_output], initializer=tf.contrib.layers.xavier_initializer )
    biases = tf.get_variable(name="biases4", shape=[layer_3_nodes], initializer=tf.zeros_initializer)
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# training time
# define teh cost function of the neural network that will measure prediction
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# define the optimiser function that will be run to optimise the neural network
with tf.varible_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialise a session so that we can run tensorflow operations
with tf.Session as session:
    # run the global variable initialiser to initialise all variables and layers of the model
    session.run(tf.global_variables_initializer())

    # run the optimiser over and over to train the network
    # one epoch is in one full run through the training data set.
    for epoch in range(learning_epochs):
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        #print the current training status to the screen
        print("Training pass: {}".format(epoch))

    # Training is now complete
    print("Training is now complete")