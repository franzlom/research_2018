# # First, let's define a RNN Cell, as a layer subclass.
# import tensorflow as tf
# from tensorflow import keras
#
#
# # First, let's define a RNN Cell, as a layer subclass.
#
# class MinimalRNNCell(keras.layers.Layer):
#
#     def __init__(self, units, **kwargs):
#         self.units = units
#         self.state_size = units
#         super(MinimalRNNCell, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
#                                       initializer='uniform',
#                                       name='kernel')
#         self.recurrent_kernel = self.add_weight(
#             shape=(self.units, self.units),
#             initializer='uniform',
#             name='recurrent_kernel')
#         self.built = True
#
#     def call(self, inputs, states):
#         prev_output = states[0]
#         h = K.dot(inputs, self.kernel)
#         output = h + K.dot(prev_output, self.recurrent_kernel)
#         return output, [output]
#
# # Let's use this cell in a RNN layer:
#
# cell = MinimalRNNCell(32)
# x = keras.Input((None, 5))
# layer = RNN(cell)
# y = layer(x)
#
# # Here's how to use the cell to build a stacked RNN:
#
# cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
# x = keras.Input((None, 5))
# layer = RNN(cells)
# y = layer(x)

from IPython.display import Image
from IPython.core.display import HTML

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
Image(url= "https://cdn-images-1.medium.com/max/1600/1*UkI9za9zTR-HL8uM15Wmzw.png")


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)
pd.set_option('display.height', 500)

# copying the dataframe so I have a before and after
df = pd.read_csv('c:/Users/Franz Lom/Documents/TensorFlow/gesture_recognition/Data/extracted_data/cir_hor_cclk.csv') # saving circle horizontal counter clockwise csv as df
dataframe = df.copy()

k = ['tester', 'timestamp', 'trial', 'posX', 'posY', 'posZ']
# dataframe = dataframe[k]
df_array = dataframe.values
# df_array


grouped_df = dataframe.groupby(['tester'])
gb = grouped_df.groups

for key, val in gb.items():
    print(df.ix[val], '\n\n')

    if key == 'B2': break

dfs = dict(tuple(dataframe.groupby('tester')))
grouped = dict()

for k, v in dfs.items():
    d = dict(tuple(v.groupby(['trial', 'tester'])))
    grouped.update(d)

    # print(v, '\n\n')
    print(k, v['trial'].unique())

# so this is the batch generating I guess
# they are also were ordered alphabetically
# as you can see they are grouped by ['tester'] B1, ..., Y3
# each tester have 10 ['trial']

# there is an interesting issue when turning this to dict()(grouped)
# they are being appended to grouped
# BUT THEY ALL HAVE THE SAME KEYS SO IT IS OVERRIDING THE DICTIONARY WITH THE SAME KEYS
# FFS

print(grouped.keys(), '\n')
print(grouped.__len__(), '\n')
print(grouped[1, 'B2'])


# hyperparams

num_epoch = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4 # number of layers
number_classes = 2
echo_step = 3
batch_size = 5
number_classes = total_series_length // batch_size // truncated_backprop_length
timestep = 1000


# data = tf.placeholder(tf.float32, [timestep, batch_size])

def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.float32)
    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size

    data = tf.reshape(raw_data[0:batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps: (i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])

    y = data[:, i * num_steps + 1: (i + 1) * num_steps]
    y.set_Shape([batch_size, num_steps])

    return x, y

x_shape = tf.placeholder(tf.float32, shape=[ None, truncated_backprop_length])
batch_size = tf.shape(x_shape)[0]
print(x_shape.get_shape()[0])

i_x, i_y = batch_producer(grouped, batch_size, num_steps=8)

print(dtype(i_x), '\n')
print(dtype(i_y))



