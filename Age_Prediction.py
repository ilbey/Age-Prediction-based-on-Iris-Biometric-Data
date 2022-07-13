# -*- coding: utf-8 -*-

# -- Sheet --

import numpy as np
import pandas as pd
import os 
import random
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import copy
from sklearn.model_selection import train_test_split

class DataParser:
  def __init__(self,path,typ):
    self.path = path
    self.feature_type = typ
    self.x_train = None
    self.y_train = None
    self.x_test = None
    self.y_test = None
    self.train = None
    self.test = None
    self.corr = None
    self.extract_features = None
    self.get_data()
    
  def get_data(self):
    self.x_train,self.y_train = self.load_data('Training')
    self.x_test,self.y_test = self.load_data('Testing')

     
  def load_data(self,mode):
    pth = f'{self.path}Iris{self.feature_type}Features_{mode}Set.txt'
    dataset = open(pth, "r")
    raw_dataset = dataset.read()

    sections = raw_dataset.split("\n\n")
    
    data = sections[2].split("\n")
    data = data[1:]
    test_list = []
    for row in data:
        test_list.append(row.split(","))
    x_list = test_list[:-1]

    df = pd.DataFrame(test_list)
    df = df.iloc[:-1,:]
    df = df.astype(np.float)
    colnames = []
    for i in range(df.shape[1]):
      if i == df.shape[1]-1:
        colnames.append("label")
      else:
        colnames.append(f"feature{i+1}")
    df.columns = colnames
    if mode == 'Training':
      self.train = copy.deepcopy(df)
    else:
      self.test = copy.deepcopy(df)
    #Changing pandas dataframe to numpy array
    x_list = df.iloc[:, : -1].values

    y_list = df.iloc[:, -1].values
    
    x_list = x_list.astype(np.float32)
    y_list = y_list.astype(np.int)
    y_list = y_list-1 # to match class labels
    return x_list, y_list

def merge_features(first,second):
  parser = copy.deepcopy(first)
  parser.x_train = np.concatenate([first.x_train,second.x_train],axis=1)
  parser.x_test = np.concatenate([first.x_test,second.x_test], axis=1)
  print(parser.x_train.shape)
  return parser
def encode_labels(y):
  encoder = OneHotEncoder()
  y = y.reshape(-1,1)

  y = encoder.fit_transform(y).toarray()
  y = np.array(y)
  return y

path = "./"
feature1 = "Texture"
feature2 = "Geometic"
parsed_data1 = DataParser(path,feature1)
parsed_data2 = DataParser(path,feature2)

#parsed_data1.y_train = encode_labels(parsed_data1.y_train)
#parsed_data1.y_test = encode_labels(parsed_data1.y_test)

parsed_data = merge_features(parsed_data1,parsed_data2)
print(parsed_data.x_train)

parsed_data.y_train = encode_labels(parsed_data.y_train)
parsed_data.y_test = encode_labels(parsed_data.y_test)

parsed_data.x_train, x_val, parsed_data.y_train, y_val = train_test_split(parsed_data.x_train, parsed_data.y_train, test_size=0.20,stratify=parsed_data.y_train )

def build_model(build):
  METRICS = [
      keras.metrics.CategoricalAccuracy(name='accuracy'),
  ]

  model = Sequential()
  for i in range(build['layers']):
    if i == 0:
      model.add(Dense(build['neurons'][i],input_dim=build['dim'],activation=build['activations'][i]))
    else:
      model.add(Dense(build['neurons'][i],activation=build['activations'][i]))
    if (build['dropout'][i] != None):
      model.add(Dropout(build['dropout'][i]))
  model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=METRICS)
  return model

build1 = {
    'dim' : 9605,
    'layers' : 1,
    'neurons': [3],
    'activations' : ['softmax'],
    'dropout' : [None]
}
build2 = {
    'dim' : 9605,
    'layers' : 2,
    'neurons': [1024,3],
    'activations' : ['relu','softmax'],
    'dropout' : [0.25, None]
}
build3 = {
    'dim' : 9605,
    'layers' : 3,
    'neurons': [512,256,3],
    'activations' : ['relu','relu','softmax'],
    'dropout' : [0.25,0.25, None]
}
build4 = {
    'dim' : 9605,
    'layers' : 4,
    'neurons': [512,256,128,3],
    'activations' : ['relu','relu','relu','relu','softmax'],
    'dropout' : [0.25,0.25,0.25, None]
}


model1 = build_model(build3)
K.set_value(model1.optimizer.learning_rate, 0.01)

model1.summary()

CALLBACKS = [
      tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5),
      tf.keras.callbacks.ModelCheckpoint("model2.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1),
      tf.keras.callbacks.TensorBoard(log_dir='./logs'),
      #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
]

class_weight = {0: 82/50,
                1: 82/82,
                2: 82/11}
model1.fit(parsed_data.x_train,parsed_data.y_train, epochs=50, batch_size=64, validation_data = (x_val,y_val), shuffle=True)#,callbacks=CALLBACKS)#,class_weight=class_weight)

# 


y_pred = model1.predict(parsed_data.x_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(parsed_data.y_test)):
    test.append(np.argmax(parsed_data.y_test[i]))

labels = ['loss','accuracy']
predic=model1.evaluate(parsed_data.x_test,parsed_data.y_test,verbose=1)
print(predic)

from sklearn.metrics import classification_report
Y_test = np.argmax(parsed_data.y_test, axis=1)
y_pred = model1.predict_classes(parsed_data.x_test)
print(classification_report(Y_test, y_pred))

