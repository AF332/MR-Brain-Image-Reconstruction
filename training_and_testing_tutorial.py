import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import os
import h5py
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('//content//drive')
#GPU hardware
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!nvidia-smi
#execute if called from google colab
! git clone https://github.com/ISMRM-MIT-CMR/CMR-DL-challenge.git
! mv ./CMR-DL-challenge/* .

#===========Read all data from files contained in folder from path
def read_data(path, keyname):
    data = []
    h5name = os.listdir(path) # Read all the names of data in file (path) and store as a list

    for i in range(len(h5name)): # Repeat for every fiel in folder from path
        temp_path = path + '//' + str(h5name[i]) # Create a path for each file in the folder
        tempdata = h5py.File(temp_path, "r") # Open h5 file as "read" and save as data
        tempkeydata = tempdata[keyname] # Store data from 'kspace' key as kspacedata
        data.append(tempkeydata) # Add data from key to list
    data = np.array(data) # Turn list into numpy array
    return data

#==========Read Target Images
def read_target_images(path):
    data = read_data(path, 'reconstruction_rss') # Read dataset called reconstruction_rss in the h5 file
    # Data is a numpy array: (4,18,256,256) (files, acquisitions, images pixels)
    data_length = len(data) # store how many files have been read
    images = [] # Generate empty variable
    for i in range(data_length): # Repeat for the number of files in folder
        d = data[i] # Save d as (18,256,256)
        num_acq = d.shape[0] # Number of acquisitions in each file
        for j in range(num_acq):  # Repeat for number of acquisitions
            acquisition = d[j, ...] # Seperate image based on loop number
            images.append(acquisition) # Add image to lits of images
    images = np.array(images) # At the end of function turn list of images into numpu array
    return images

#==========Read Kspace Images and combine images from each coil fro each acquisition
def read_source_images(path):
    data = read_data(path, 'kspace') # Read dataset called reconstruction_rss in the h5 file
    # Data is a numpy array: (4, 18, 4, 256, 256) (files, acquisitions, coils, imagex, imagey)
    data_length = len(data) # store how many files have been read
    kspace_data = [] # Generate empty variable
    kspace_images = []
    for i in range(data_length): # Repeat for the number of files in folder
        d = data[i] # Save d as (18, 4, 256, 256)
        num_acq = d.shape[0] # Number of acquisitions in each file
        for j in range(num_acq):  # Repeat for number of acquisitions
            acquisition = d[j, ...] # Seperate image based on loop number
            kspace_data.append(acquisition) # Add image to lits of images
            #This is simplified to average for now-------
            coil0 = acquisition[0, ...]
            coil1 = acquisition[1, ...]
            coil2 = acquisition[2, ...]
            coil3 = acquisition[3, ...]
            kspace = (coil0+coil1+coil2+coil3)/4 # Coils added and averaged
            #--------------------------------------------
            kspace_images.append(kspace)
    kspace_data = np.array(kspace_data) # At the end of function turn list of images into numpu array
    kspace_images = np.array(kspace_images)
    return kspace_images

#==========Read Kspace Images and combine images from each coil fro each acquisition and add gaussian noise
def read_source_images_noise(path, mean, standard_deviation):
    data = read_data(path, 'kspace') # Read dataset called reconstruction_rss in the h5 file
    # Data is a numpy array: (4, 18, 4, 256, 256) (files, acquisitions, coils, imagex, imagey)
    data_length = len(data) # store how many files have been read
    kspace_data = [] # Generate empty variable
    kspace_images = []
    kspace_noise = []

    for i in range(data_length): # Repeat for the number of files in folder
        d = data[i] # Save d as (18, 4, 256, 256)
        num_acq = d.shape[0] # Number of acquisitions in each file
        for j in range(num_acq):  # Repeat for number of acquisitions
            acquisition = d[j, ...] # Seperate image based on loop number
            kspace_data.append(acquisition) # Add image to lits of images
            #This is simplified to average for now-------
            coil0 = acquisition[0, ...]
            coil1 = acquisition[1, ...]
            coil2 = acquisition[2, ...]
            coil3 = acquisition[3, ...]
            kspace = (coil0+coil1+coil2+coil3)/4 # Coils added and averaged
            noise = np.random.normal(mean, standard_deviation, (kspace.shape)).astype(np.uint8)
            im_noise = kspace + noise
            #--------------------------------------------
            kspace_images.append(kspace)
            kspace_noise.append(im_noise)
    kspace_data = np.array(kspace_data) # At the end of function turn list of images into numpu array
    kspace_images = np.array(kspace_images)
    kspace_noise = np.array(kspace_noise)
    return kspace_noise

train_kspace_images = read_source_images_noise('//content//drive//MyDrive//DeepLearningProject//Training', 0, 10)
validation_kspace_images = read_source_images('//content//drive//MyDrive//DeepLearningProject//Training')
test_images = read_source_images('//content//drive//MyDrive//DeepLearningProject//Testing')

#train_images = train_images.transpose(2, 1, 0)
#test_images = test_images.transpose(2, 1, 0)
#train_kspace_images = train_kspace_images.transpose(2, 1, 0)
#test_kspace_images = test_kspace_images.transpose(2, 1, 0)

print(train_kspace_images.shape) # Number of files(4)*acquisitions(18)
print(validation_kspace_images.shape)  # Number of files(2)*acquisitions(18)
print(test_images.shape) # Number of files(2)*acquisitions(18)

#training
train_set = train_kspace_images

#validation
val_set = validation_kspace_images

#test
test_set = test_images

print('Training batches to process:', len(train_set))
print('Validation batches to process:', len(val_set))
print('Test samples to process:', len(test_set))

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),      # used optimizer with chosen learning rate
              loss='mse',                                                   # loss function
              metrics=['mse', 'mae'])                                       # evaluation metrics (for training and validation set)

# define callbacks to monitor model
keras_callbacks = get_callbacks(val_test, model)#NEED TO ASK ABOUT THIS

keras_callbacks = model.get_callbacks(val_set, model)
model.fit(train_set,
          val_data = val_set,
          epochs = 3)
#train model prediction
predicted_output = model.predict(test_set)

#trained model evaluation
test_loss = model.evaluate(test_set)

#display
icase = 0
plt.imshow(np.squeeze(np.abs(predicted_output[icase,])), cmap=plt.gray())
plt.show()
