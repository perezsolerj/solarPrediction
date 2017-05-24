import tensorflow as tf
import time


""" ==============================================================================
Creates the basic model used for dehazing used in other training & dehazing files (Not meant to be executed).
  Required functions:

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.
============================================================================== """



""" ==============================================================================

				INFERENCE FUNCTIONS

============================================================================== """

def basicInference(images, windowSize, predTime, prevImg):
  """ Builds the model and returns a prediction according to predTime"""
  """ Parameters:
	images: the placeholder of the images (input for the inference)
	windowSize: the images size
	predTime: the amount of time we want to predict
	prevImg: the images received"""

  conv1 = tf.layers.conv2d(images, 20, [5,5], padding='same', activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(pool1, 35, [5,5], padding='same', activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  dense1 = tf.layers.dense(tf.reshape(pool2,[-1, (windowSize/4)*(windowSize/4)*35]), 300, activation=tf.nn.relu)
  return tf.layers.dense(dense1, predTime)

def basicCopInference(images, windowSize, predTime, prevImg, copernicus):
  conv1 = tf.layers.conv2d(images, 20, [5,5], padding='same', activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(pool1, 35, [5,5], padding='same', activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, (windowSize/4)*(windowSize/4)*35])
  pool2_flatCop = tf.concat([pool2_flat, copernicus], 1)

  dense1 = tf.layers.dense(pool2_flatCop, units=300, activation=tf.nn.relu)
  return tf.layers.dense(dense1, predTime)


""" ==============================================================================

				LOSS FUNCTIONS

============================================================================== """

def basicLoss(labels,predict):
  """Uses the labels (groundtruth) and predicted values to obtain a loss to minimize"""

  loss=tf.losses.mean_squared_error(labels, predict)
  tf.summary.scalar('MSE', loss) ##To control it in tensorboard!

  return loss

""" ==============================================================================

				TRAIN FUNCTIONS

============================================================================== """

def training(loss):
  """Training parameters"""

  optimizer = tf.train.AdamOptimizer().minimize(loss)
  return optimizer


""" ==============================================================================

				TRAIN LOGGER

============================================================================== """

class trainLogger:
  """ This class holds the information about the loss and provides functions to display it
      while training and store it in drive to obtain graphs, it also manages tensorboard"""


  def __init__(self,summariesDIR, sess, miniBatchSize, instances):
    ## Error vars
    self._minibatch = 0
    self._epoch = 0
    self._epochLoss= 0
    self._miniBatchSize=miniBatchSize
    self._instances=instances
    self._Nminibatches=int(instances/miniBatchSize)
    self._showEpoch=False
    self._lastLoss=0

    ## Time vars
    self._batchTime=time.time()
    self._epochTime=time.time()
    self._batchElapsed=0
    self._epochElapsed=0

    ## Create new summary vars
    self._sess=sess
    self._summDIR = summariesDIR
    self._train_writer = tf.summary.FileWriter(summariesDIR + '/train', sess.graph)
    self._TFEpochLoss = tf.Variable(500, name="EpochLoss")
    self._TFEpochLoss_summ = tf.summary.scalar("EpochLoss", self._TFEpochLoss)
    sess.run(self._TFEpochLoss.assign(500))

  def addMiniBatchResults(self, loss, epoch, summary):
    tick=time.time()
    self._train_writer.add_summary(summary, (self._minibatch + self._Nminibatches * self._epoch))
    self._minibatch = self._minibatch + 1
    self._lastLoss=loss
    self._epochLoss= loss + self._epochLoss
    self._batchElapsed = round(tick - self._batchTime,2)
    self._batchTime=tick
    if(epoch != self._epoch):
      self._showEpoch=True
      self._epochElapsed= round(tick - self._epochTime,2)
      self._epochTime=tick
      self._sess.run(self._TFEpochLoss.assign(self._epochLoss/self._Nminibatches))
    self._epoch=epoch

  def newEpoch(self):
    return self._showEpoch

  def showResults(self):
    print('********* Epoch: ' + str(self._epoch) + ' Minibatch:' + str(self._minibatch) +'/' + str(self._Nminibatches) + ' ***********')
    print('Loss: ' + str(self._lastLoss) + ' elapsed: ' + str(self._batchElapsed))
    if (self._showEpoch):
      self._minibatch=0
      print('')
      print('=============== Epoch ' + str(self._epoch) + ' ============')
      print('Epoch loss: ' + str(round(self._epochLoss/self._Nminibatches,2))+ ' elapsed: ' + str(self._epochElapsed))
      print('')
      self._showEpoch=False
      self._epochLoss=0








