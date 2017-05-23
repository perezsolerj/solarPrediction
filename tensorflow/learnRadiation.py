import loadData
import tensorflow as tf
import numpy as np
import scipy.io as sio

windowSize=151
predictTime=5
trainInitDate=[2016, 1, 1]
trainEndDate=[2016, 1, 11]
dataFolder='../data/-0.06-39.99/'

validationInitDate = [2015, 1, 1]
validationEndDate = [2015, 2, 1]

# Import data
dataset=loadData.loadRadiationData(dataFolder,trainInitDate,trainEndDate,windowSize,predictTime)

# Create the model
x = tf.placeholder(tf.float32, [None, windowSize , windowSize,])
layer1 = tf.layers.dense(tf.reshape(x,[-1, windowSize*windowSize]), 100, activation=tf.nn.relu)
y = tf.layers.dense(layer1, predictTime)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, predictTime])

loss = tf.losses.mean_squared_error(y_,y)
train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(10):
  batch_xs, batch_ys = dataset.next_batch(200)
  _train, lossValue = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
  print(str(dataset.epochs_completed) + ' -> ' + str(lossValue))



# Test trained model
valSet=loadData.loadRadiationData(dataFolder,validationInitDate,validationEndDate,windowSize,predictTime)
result=np.empty((0,predictTime))
labels=np.empty((0,predictTime))
while(valSet.epochs_completed!=1):
  batch_xs, batch_ys = valSet.next_Valbatch(50)
  predict = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
  result=np.concatenate((result,predict),axis=0)
  labels=np.concatenate((labels,batch_ys),axis=0)
sio.savemat('predict.mat', {'predict':result})
sio.savemat('labels.mat', {'labels':labels})

"""correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))"""
