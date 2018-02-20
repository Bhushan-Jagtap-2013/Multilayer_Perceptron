import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# read data set from cvs file

def get_dataset():
    df = pd.read_csv("dataset.cvs")

    x = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    # Encode the dependant variables

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    _y = one_hot_encode(y)
    print(x.shape)
    return (x, _y)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encoded_matrix = np.zeros((n_labels, n_unique_labels))
    one_hot_encoded_matrix[np.arange(n_labels), labels] = 1
    return one_hot_encoded_matrix


# Read processed data set from file
X, Y = get_dataset()

# shuffle row wise to mix up as its in order
X, Y = shuffle(X, Y, random_state=1)

# print shapes
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=42)

print("Train data shape : \t", train_x.shape)
print("Train labels shape : \t", train_y.shape)
print("Test data shape : \t", test_x.shape)
print("Test data shape : \t", test_y.shape)

# Define the parameter

lr = 0.3
train_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
each_sample_dim = X.shape[1]

print("Each sample is of size : ", each_sample_dim)

n_output_classes = test_y.shape[1]

print("Number of output classes : ", n_output_classes)

model_path = "model\\NMI"

# Define number of hidden layer and number of neurons for each layer

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

# define your model

# last layer
x = tf.placeholder(tf.float32, [None, each_sample_dim])
_y = tf.placeholder(tf.float32, [None, n_output_classes])


# Hidden layer

def multilayer_perceptron(x, weights, biases):
    # Hidden layer + sigmoid
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)

    # Hidden layer + sigmoid
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)

    # Hidden layer + sigmoid
    layer3 = tf.add(tf.matmul(layer2, weights['h3']), biases['b3'])
    layer3 = tf.nn.sigmoid(layer3)

    # Hidden layer + RELU
    layer4 = tf.add(tf.matmul(layer3, weights['h4']), biases['b4'])
    layer4 = tf.nn.relu(layer4)

    # last layer
    out_layer = tf.matmul(layer4, weights['out']) + biases['out']
    return out_layer


# define the weights and biases
weights = {
    'h1': tf.Variable(tf.truncated_normal([each_sample_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_output_classes])),
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_output_classes])),
}

# Initialize variables
init = tf.global_variables_initializer()
save_model = tf.train.Saver()

# call model
our_model = multilayer_perceptron(x, weights, biases)

# loss function and back propagation
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=our_model, labels=_y))
back_prop = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(init)

# Maintain loss and accuracy at each epoch

mse_history = []
accuracy_history = []

for epoch in range(train_epochs):
    sess.run(back_prop, feed_dict={x: train_x, _y: train_y})
    cost = sess.run(loss, feed_dict={x: train_x, _y: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(our_model, 1), tf.argmax(_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # run on test data
    pred_y = sess.run(our_model, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)

    accuracy = sess.run(accuracy, feed_dict={x: train_x, _y: train_y})
    accuracy_history.append(accuracy)

    print('epoch ', epoch, '\t\t- ', 'cost', cost, '\t\t- mse_: ', mse_, '\t\t- Train Accuracy : ', accuracy)

save_path = save_model.save(sess, model_path)
print("Model is saved at %s" % save_path)

plt.plot(mse_history[5:], 'r')
plt.show()

plt.plot(accuracy_history)
plt.show()

# print the final accuracy

correct_prediction = tf.equal(tf.argmax(our_model, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy : ", sess.run(accuracy, feed_dict={x: test_x, _y: test_y}))

# Print the final mean square error

pred_y = sess.run(our_model, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print('MSE : %.f' % sess.run(mse))
