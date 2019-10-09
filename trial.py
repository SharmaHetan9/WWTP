from __future__ import print_function
from os import listdir
from os.path import isfile , join


mypath = 'C://Users//PRACHI//Desktop//AI//dataset//img'

filename = [f for f in listdir(mypath) if isfile(join(mypath , f))]
print(str(len(filename)))

import cv2
import numpy as np
import sys 
import os 
import shutil

no_count = 0
yes_count = 0
size = 150
yes_train = 'C://Users//PRACHI//Desktop//AI//dataset//yes_train'
no_train = 'C://Users//PRACHI//Desktop//AI//dataset//no_train'
no_test = 'C://Users//PRACHI//Desktop//AI//dataset//no_test'
yes_test = 'C://Users//PRACHI//Desktop//AI//dataset//yes_test'

def make_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    
make_dir(yes_train)
make_dir(no_train)
make_dir(no_test)
make_dir(yes_test)

print('om')


def getzeros(no):
    if(no > 10 and no < 100):
        return '0'
    if(no < 10):
        return '00'
    else:
        return ""
    
training_img = []
training_lab = []
test_img = []
test_lab = []

for i,file in enumerate(filename):
  if filename[i][0] == 'y':
    yes_count += 1
    img  = cv2.imread(mypath+ "//" +  file)
    img = cv2.resize(img ,(size , size), interpolation = cv2.INTER_AREA)
    if yes_count <= 17:
      training_img.append(img)
      training_lab.append(1)
      zeros = getzeros(yes_count)
    if yes_count > 17 :
      test_img.append(img)
      test_lab.append(1)

  elif filename[i][0] == 'N':
      no_count += 1
      imgs = cv2.imread(mypath + '//' + file)
      imgs = cv2.resize(imgs ,(size , size), interpolation = cv2.INTER_AREA)
      if no_count <= 3:
        training_img.append(imgs)
        training_lab.append(0)
      if no_count > 3 :
        test_img.append(imgs)
        test_lab.append(0)

np.savez('pest_train_data.npz' , np.array(training_img))
np.savez('pest_test_data.npz' , np.array(test_img))
np.savez('pest_train_lab.npz' , np.array(training_lab))
np.savez('pest_test_loab.npz' , np.array(test_lab))

def ld_data():
  npzfile = np.load( 'pest_train_data.npz')
  train = npzfile['arr_0']

  npzfile = np.load('pest_test_data.npz')
  test = npzfile['arr_0']

  npzfile = np.load('pest_training_lab.npz')
  train_lab = npzfile['arr_0']

  npzfile = np.load('pest_test_loab.npz')
  test_loab = npzfile['arr_0']

  return(train , train_lab) , (test , test_loab)

import random

for i in range (1,11):
  random = np.random.randint(0 , len(training_img))
  cv2.imshow('image' + str(i) , training_img[random])
  if training_lab[random] == 0:
    print('no pest detected')
  elif training_lab[random] == 1:
    print('pest detected')
  cv2.waitKey(0)
cv2.destroyAllWindows()

(x_train , y_train) , (x_test , y_test)  = ld_data()
y_train = y_train.reshape(y_train.shape[0] , 1)
y_test = y_test.reshape(y_test.shape[0] , 1)

x_test = x_test.astype('float32')
x_train = x_train.astype('float32')

x_test /=255
x_train /= 255
print(x_train[0].shape[0])
print(x_train[1].shape[0])


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Dropout , Activation , Flatten
from keras.layers import Conv2D , MaxPooling2D
import os

batch_size = 16
epochs = 50

model = Sequential()
model.add(Conv2D(32, (3 , 3) , input_shape = ( 150 , 150 , 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3 , 3) , input_shape = (150 , 150 , 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3 , 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3 , 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss= 'binary_crossentropy',
              optimizer= 'rmsprop',
              metrics = ['accuracy'])

print(model.summary())

history = model.fit(x_train , y_train,
                    batch_size=batch_size,
                    epochs = epochs,
                    validation_data=(x_test , y_test),
                    shuffle = True)

model.save(mypath + '//' + 'i_kisan.h5')
scores = model.evaluate(x_test , y_test , verbose=1)
print (scores)
                    
import cv2
import numpy as np
from keras.models import load_model

md = load_model(mypath + '//' + 'i_kisan.h5')
ran = np.random.randint(1,10)
def test(nm , pred  , input_in):
  BLACK  =  [0,0,0]
  if pred == '[0]':
    pred == 'No pests'
  elif pred == '[1]':
    pred == 'Pests found'
  cv2.imshow('re' , input_in )
  print(pred)

res = str(md.predict_classes(x_test[ran] , 1 , verbose = 0)[0])
test('Prediction ' , res , x_test[ran])




