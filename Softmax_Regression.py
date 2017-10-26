# The following code was used as a tutorial to learn tensorflow and can be found on TensorFlow's website

import tensorflow as tf
sess = tf.InteractiveSession()

# Load the data:
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Placeholders:
x = tf.placeholder(tf.float32, shape=[None,784]) # None for unspecified amount of evidence, imported as 32x32 pixel matrices (784 pixels total)
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # Target output classes

# Variables:
W = tf.Variable(tf.zeros([784, 10])) # shape as this in order to multiply it by x to output produce a 10-dim vector for each class
b = tf.Variable(tf.zeros([10])) # 10 for the output classes. this is the bias

sess.run(tf.global_variables_initializer())


# Defining the model:
y = tf.matmul(x, W) + b

# Defining the loss function:
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) # tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes


# Train the model:
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #0.5 is the step size
    # here, TensorFlow added new operations to the computation graph, including ones to compute gradients, compute parameter update steps, apply update steps to the parameter
    # when run, this operation will apply the gradient descent updates to the parameters
        # so training the model can therefore be acccomplished by repeatedly running train_step:
    
for i in range(1000):
    batch = mnist.train.next_batch(100) # load 100 training examples
    train_step.run(feed_dict={x: batch[0], y_: batch[1]}) # run train_step, using feed_dict to replace the placeholder tensors x and y_ with the training examples


# Evaluate the model:
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # gives us a list of booleans
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # converts booleans to floating point numbers and takes the mean
print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
    
