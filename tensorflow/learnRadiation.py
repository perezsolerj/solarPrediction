import loadData
import model
import tensorflow as tf
import numpy as np
import scipy.io as sio

import datetime

""" PARAMETER AREA"""

##training parameters
runUntil=datetime.datetime(2017,5,25,16,00)
max_epochs=1000
miniBatchSize=200

##Data Input parameters
windowSize=151
predictTime=4
previousImages=5
trainInitDate=[2016, 1, 1]
trainEndDate=[2016, 1, 11]
dataFolder='../data/UJI/'

##Validation parameters
validationInitDate = [2015, 1, 1]
validationEndDate = [2015, 1, 11]

##Summary parameters
summariesDIR='/tmp/solarRad'

# Import data
dataset=loadData.loadRadiationData(dataFolder,trainInitDate,trainEndDate,windowSize,predictTime,previousImages)

# Create the model
x = tf.placeholder(tf.float32, [None, windowSize , windowSize, previousImages])
copernicus = tf.placeholder(tf.float32, [None, predictTime+previousImages])

y = model.basicCopInference(x, windowSize, predictTime, previousImages,copernicus)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, predictTime])
loss = model.basicLoss(y_,y)

train_step = model.training(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

##launch logger
logger = model.trainLogger(summariesDIR, sess, miniBatchSize, dataset.num_examples)

##Merge summaries for tensorboard and saver for checkpoints
merged = tf.summary.merge_all()
saver = tf.train.Saver()

# Train
print("\n -> Starting to train until "+ str(max_epochs) + " epochs or " +  str(runUntil) + " <-\n")
now = datetime.datetime.now()
while (dataset.epochs_completed < max_epochs and now < runUntil):
  batch_xs, batch_ys, batch_x2s = dataset.next_batch(miniBatchSize)
  _train, summary,  lossValue = sess.run([train_step, merged, loss], feed_dict={x: batch_xs, y_: batch_ys, copernicus: batch_x2s})

  logger.addMiniBatchResults(lossValue, dataset.epochs_completed, summary)
  if (logger.newEpoch()):
    saver.save(sess,'models/model.ckpt')

  logger.showResults()

  now=datetime.datetime.now()

saver.save(sess,'models/FINALmodel.ckpt')

print("\n -> Finished Training, evaluating with validation Data <- \n")

# Test trained model
valSet=loadData.loadRadiationData(dataFolder,validationInitDate,validationEndDate,windowSize,predictTime,previousImages)
result=np.empty((0,predictTime))
labels=np.empty((0,predictTime))
while(valSet.epochs_completed!=1):
  batch_xs, batch_ys, batch_x2s = valSet.next_Valbatch(50)
  predict = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, copernicus: batch_x2s})
  result=np.concatenate((result,predict),axis=0)
  labels=np.concatenate((labels,batch_ys),axis=0)
sio.savemat('predict.mat', {'predict':result})
sio.savemat('labels.mat', {'labels':labels})

print("\n -> Finished evaluating, saving evaluated training data <- \n")

## Test training set
result=np.empty((0,predictTime))
labels=np.empty((0,predictTime))
aux_epochs=dataset.epochs_completed
while(dataset.epochs_completed!=aux_epochs+1):
  batch_xs, batch_ys, batch_x2s = dataset.next_Valbatch(50)
  predict = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, copernicus: batch_x2s})
  result=np.concatenate((result,predict),axis=0)
  labels=np.concatenate((labels,batch_ys),axis=0)
sio.savemat('predictTRAIN.mat', {'predict':result})
sio.savemat('labelsTRAIN.mat', {'labels':labels})

