from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.io as sio

def loadMatFile(file):
  """ Reads a .mat file and returns a numpy array with the matrix"""
  mat=sio.loadmat(file)
  return np.array(mat['data'])

def nextDay(startDay):
  """Returns the next day from an array [year, month, day]"""
  monthDays=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  startDay[2]=startDay[2]+1

  if (startDay[2] > monthDays[startDay[1]-1]):  ##Check next month
    if(startDay[1] != 2  or startDay[0]%4 !=0 or startDay[2]>29):  ##Check if it is a leap year
      startDay[2]=1
      startDay[1]=startDay[1]+1

      if(startDay[1]>12): ##Check next year
        startDay[0]=startDay[0]+1
	startDay[1]=1

  return startDay

class RadiationDataSet(object):
  """Class that holds a radiation dataset and offers batch functions to provide data to train"""
  def __init__(self, images, labels, size):
    assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
    self._num_examples = images.shape[0]
    self._perm=np.arange(self._num_examples)
    np.random.shuffle(self._perm)

    self._images = images
    self._labels = labels
    self._size = size
    
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._val_index=0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def createBatch(self,start,end):
    """Creates a batch with the images in the permutation self._perm"""
    batch=np.empty((0,self._size,self._size)) 
    batchLabels=[]

    for i in xrange(start,end):
      batch=np.append(batch,np.reshape(self._images[self._perm[i],:,:],[1,self._size,self._size]),axis=0)
      batchLabels.append(self._labels[self._perm[i]])

    return batch, np.array(batchLabels).reshape((-1,1))
      

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed =  self._epochs_completed + 1
      # Shuffle the permutation
      np.random.shuffle(self._perm)

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self.createBatch(start,end)

  def next_Valbatch(self, batch_size):
    """Return the next `batch_size` examples from this data set without permuting."""  
    start = self._val_index
    self._val_index += batch_size
    if self._val_index > self._num_examples:
      # Finished epoch
      # Start next epoch
      self._epochs_completed =  self._epochs_completed + 1
      start = 0
      self._val_index = batch_size
      assert batch_size <= self._num_examples
    end = self._val_index
    return self._images[start:end,:,:], self._labels[start:end]

def loadRadiationData(dataDir,startDate,endDate,size):
  """ Reads the files from the different folders and loads a RadiationDataset ready to use it in the training"""
  
  images=np.empty((0,size,size))

  while(startDate!=endDate):
    filePath = dataDir+'/'+str(startDate[0])+'/'+str(startDate[1])+'/'+str(startDate[2])+'.mat'
    if os.path.isfile(filePath):
	image=np.swapaxes(loadMatFile(filePath),0,2)
	images=np.append(images,image,axis=0)
    else:
	print(str(startDate) + ' not available')

    
    nextDay(startDate)

  labels=images[:,int((size+1)/2),int((size+1)/2)]
  return RadiationDataSet(images,labels,size)



"""### DEBUGGING TESTS
day=loadMatFile('../data/-0.06-39.99/2016/1/1.mat')
print(day)
print(day.shape)

dataset=loadRadiationData('../data/-0.06-39.99/',[2016, 1, 1],[2016, 1, 11],151)
rad,label=dataset.next_Valbatch(300)
rad,label=dataset.next_Valbatch(300)
rad,label=dataset.next_Valbatch(300)
rad,label=dataset.next_Valbatch(300)
rad,label=dataset.next_Valbatch(300)
rad,label=dataset.next_Valbatch(300)
print(dataset.epochs_completed)
print(rad.shape)
print(label.shape)
print(label) 
##print(rad)"""


