import tensorflow as tf
from tensorflow import keras
import pandas as pd


from sklearn.preprocessing import MinMaxScaler
gesture_names = ['v_shape', 'x_shape', 'horizontal_circle', 'vertical_circle']
headers = ['Tester', 'Trial','Right_Handed', 'Time_Stamp', 'pos_X', 'pos_Y', 'pos_Z', 'rel_pos_X', 'rel_pos_Y',
           'rel_pos_Z', 'vel_X', 'vel_Y', 'vel_Z', 'quaternion_W', 'quaternion_X', 'quaternion_Y', 'quaternion_Z',
           'relative_acceleration_X', 'relative_acceleration_Y', 'relative_acceleration_Z', 'absolute_acceleration_X',
           'absolute_acceleration_Y', 'absolute_acceleration_Z']

training_data_df = pd.read_csv(
    '/Users/Franz Lom/Documents/TensorFlow/gesture_recognition/Data/extracted_data/cir_ver_cclk.csv')
# circle vertical counter clockwise

# values in the csv
# tester, trial ,rightHanded, timestamp, posX, posY, posZ, posRelX, posRelY, posRelZ, velX, velY, velZ, quatW, quatX,
# quatY, quatZ, accRelX, accRelY, accRelZ, accAbsX, accAbsY, accAbsZ



# defining the input layer
model = keras.Sequential()
model.add(keras.layers.GRU(32, input_shape=(16, )))


print(training_data_df['posX'].dtype)
#print(test_data_df)

# Pull out the columns for the x (data to train with) and y (value to predict)
X_training = training_data_df.drop(['tester', 'trial', 'rightHanded'], axis=1).values

Y_training = training_data_df[['trial']].values




# load testing data set from csv - this means the values to be tested, ie other gestures
test_data_df = pd.read_csv(
    '/Users/Franz Lom/Documents/TensorFlow/gesture_recognition/Data/extracted_data/cir_ver_cclk.csv')
test_data_df.head()
print('test_data_df')
print(test_data_df)
print()
print()

# test_data_df['pos_X']
X_testing = test_data_df.drop(['tester', 'trial', 'rightHanded'], axis=1).values
Y_testing = test_data_df[['trial']].values
print('x testing')
print(X_testing.dtype)


# all data needs to be scled to a small range like 0 to 1
# for neural network to work well. Create scalars for the input and outputs
# this is the veactor normalisation that we need to do

X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

# scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)



# it's very important that the training and test data are scaled with the same scalar
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)





print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

#print("Note: Y values were scaled by multiplying {0.10f} and adding {:.4f".format(Y_scaler.scale_[0], Y_scaler.scale_[0]))0

print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))

from numpy import  np

#  Define the smape of CNN image features (N, 512) and hidden state (N, 512)
input_dim  = 512    # CNN features dimension: 512
hidden_dim = 512    # Hidden state dimension: 512
layers     = 512

#  Define a matrix to project the CNN image features to h0
# w_proj: (input_dim, hidden_dim)
w_proj = np.random.randn(input_dim, hidden_dim)
w_proj /= np.sqrt()(input_dim)
b_proj = np.zeros(hidden_dim)

#  Compute h0 by multiplying the image features with Wproj
#  Initialise CNN -> hidden state projection parameters
#  h0: (N, hidden_dim)
h0 = features.dot(w_proj) + b_proj

#  converting an input caption word
wordvec_dim = 256  # Convert a work index to a vector of 256 numbers

# randomly initialise W which we will train it with the RNN together
W_emved = np.random.randn(vocab_size, wordvec_dim)


cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)
output, out_state = tf.nn.dynamic_rnn(cell, seq, length, initial_state)

full_loss = tf.nn.softmax_cross_entropy_with_logits(preds, labels)
loss = tf.reduce_mean(tf.boolean_mask(full_loss, mask))
tf.reduce_sum(tf.reduce_max(tf.sight(seq), 2), 1)
output, outstate = tf.nn.dynamic_rnn(cell, seq, length, initial_state)


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(1, 2, _linear([inputs, state],
                                                     2 * self._num_units, True, 1.0))
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                c = self._activation(_linear([inputs, r * state],
                                             self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h