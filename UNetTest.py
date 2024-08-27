import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Conv2DTranspose, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Activation, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
#from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import os
import h5py
import matplotlib.pyplot as plt
#---
#from google.colab import drive
#drive.mount('//content//drive')
#---

path_validation = 'c://Users//2418015//OneDrive - University of Dundee//Documents//Validation'
path_training = 'c://Users//2418015//OneDrive - University of Dundee//Documents//Training'
path_testing = 'c://Users//2418015//OneDrive - University of Dundee//Documents//Testing'

#===========Read all data from files contained in folder from path
def read_data(path, keyname):
    data = [] # Create an empty variable
    h5name = os.listdir(path) # Read all the names of data in file (path) and store as a list
    for i in range(len(h5name)): # Repeat for every file in folder from path
        temp_path = path + '//' + str(h5name[i]) # Create a path for each file in the folder
        tempdata = h5py.File(temp_path, "r") # Open h5 file as "read" and save as data
        tempkeydata = tempdata[keyname] # Store data from 'kspace' key as kspacedata
        data.append(tempkeydata) # Add data from key to list
    data = np.array(data) # Turn list into numpy array
    return data

#==========Read Kspace Images and combine images from each coil fro each acquisition and add gaussian noise
def read_source_images_noise(path, mean, standard_deviation):
    data = read_data(path, 'kspace') # Read dataset called reconstruction_rss in the h5 file
    # Data is a numpy array: (4, 18, 4, 256, 256) (files, acquisitions, coils, imagex, imagey)
    data_length = len(data) # store how many files have been read
    kspace_images = [] # Generate empty variable
    kspace_noise = [] # Generate empty variable
    image_images = []
    image_noise = []
    for i in range(data_length): # Repeat for the number of files in folder
        d = data[i] # Save d as (18, 4, 256, 256)
        num_acq = d.shape[0] # Number of acquisitions in each file
        for j in range(num_acq):  # Repeat for number of acquisitions in each file
            acquisition = d[j, ...] # Seperate image based on loop number
            #This is simplified to average for now-------
            coil0 = acquisition[0, ...]
            coil1 = acquisition[1, ...]
            coil2 = acquisition[2, ...]
            coil3 = acquisition[3, ...]
            kspace = (coil0+coil1+coil2+coil3)/4 # Coils added and averaged
            noise = np.random.normal(mean, standard_deviation, (kspace.shape)).astype(np.uint8) # Change this into MRI related noise. Change into 8-bit values to keep values from 0-255
            im_noise = kspace + noise # Add noise to the kspace
            temp_im_noise = np.fft.ifft2(im_noise) # Translate into image domain
            temp_im_noise = np.fft.fftshift(temp_im_noise) # Reshape data into image
            im_image = np.fft.ifft2(kspace) # Translate into image domain
            im_image = np.fft.fftshift(im_image) # Reshape data into image
            kspace_images.append(kspace) # Adding kspace images into the list
            kspace_noise.append(im_noise) # Adding noisy kspace images into the list
            image_images.append(im_image)
            image_noise.append(temp_im_noise)
    kspace_images = np.array(kspace_images) # At the end of function turn list of images into numpy array
    kspace_noise = np.array(kspace_noise) # At the end of function turn list of images into numpy array
    image_images = np.array(image_images) # At the end of function turn list of images into numpy array
    image_noise = np.array(image_noise) # At the end of function turn list of images into numpy array
    return kspace_images, kspace_noise, image_images, image_noise, data

_, _, train_image_images, train_image_noise, data = read_source_images_noise(path_training, 0, 6)
_, _, validation_image_images, validation_image_noise, _  = read_source_images_noise(path_validation, 0 , 6)
_, _, test_image_images, _, _ = read_source_images_noise(path_testing, 0 , 6)

print('Size of training images')
print(train_image_images.shape) # Number of files(4)*acquisitions(18)
print('Size of noisy training images')
print(train_image_noise.shape) # Number of files(2)*acquisitions(18)
print('Size of validation images')
print(validation_image_images.shape)  # Number of files(2)*acquisitions(18)
print('Size of noisy validation images')
print(validation_image_noise.shape) # Number of files(2)*acquisitions(18)
print('Size of test images')
print(test_image_images.shape)


#training generator
train_set = (train_image_noise, train_image_images)

#validation generator
val_set = (validation_image_noise, validation_image_images)

#test generator
test_set = test_image_images

#batches to process
print('Training batches to process:', len(train_set))
print('Validation batches to process:', len(val_set))
print('Test samples to process:', len(test_set))


def conv_block(inputs = None, n_filters = 64, batch_norm = True, dropout_prob = 0.4):
     convolutional_1 = SeparableConv2D(n_filters, 3, padding = 'same')(inputs)
     if batch_norm:
          convolutional_1 = BatchNormalization(axis = -1)(convolutional_1)
     convolutional_1 = LeakyReLU(alpha = 0.2)(convolutional_1)

     convolutional_2 = SeparableConv2D(n_filters, 3, padding = 'same')(convolutional_1)
     if batch_norm:
          convolutional_2 = BatchNormalization(axis = -1)(convolutional_2)
     convolutional_2 = LeakyReLU(alpha = 0.2)(convolutional_2)

     if dropout_prob > 0:
          convolutional_2 = Dropout(dropout_prob)(convolutional_2)

     return convolutional_2


def encoder_block(inputs = None, n_filters = 64, batch_norm = True, dropout_prob = 0.4):

     skip_con = conv_block(inputs, n_filters, batch_norm, dropout_prob)
     # next_layer = SeparableConv2D(n_filters, 3, strides = 2, padding = 'same')(skip_con)
     next_layer = MaxPooling2D((2,2))(skip_con)

     return next_layer, skip_con


def decoder_block(expansive_input, skip_con, n_filters, batch_norm = True, dropout_prob = 0.4):

     up_samp = Conv2DTranspose(n_filters, 3, strides = 2, padding = 'same')(expansive_input)
     sum = concatenate([up_samp, skip_con], axis = -1)
     convolution = conv_block(sum, n_filters, batch_norm, dropout_prob)

     return convolution


def Unet(input_size = (256, 256, 1), n_filters = 64, n_classes = 2, batch_norm = True, dropouts = np.zeros(9)):

     inputs = Input(input_size)

     encoder_block_1 = encoder_block(inputs, n_filters, batch_norm, dropout_prob = dropouts[0])
     encoder_block_2 = encoder_block(encoder_block_1[0], n_filters * 2, batch_norm, dropout_prob = dropouts[1])
     encoder_block_3 = encoder_block(encoder_block_2[0], n_filters * 4, batch_norm, dropout_prob = dropouts[2])
     encoder_block_4 = encoder_block(encoder_block_3[0], n_filters * 8, batch_norm, dropout_prob = dropouts[3])

     bridge = conv_block(encoder_block_4[0], n_filters * 16, batch_norm, dropout_prob = dropouts[4])

     decoder_block_4 = decoder_block(bridge, encoder_block_4[1], n_filters * 8, batch_norm, dropout_prob = dropouts[5])
     decoder_block_3 = decoder_block(decoder_block_4, encoder_block_3[1], n_filters * 4, batch_norm, dropout_prob = dropouts[6])
     decoder_block_2 = decoder_block(decoder_block_3, encoder_block_2[1], n_filters * 2, batch_norm, dropout_prob = dropouts[7])
     decoder_block_1 = decoder_block(decoder_block_2, encoder_block_1[1], n_filters, batch_norm, dropout_prob = dropouts[8])

     if n_classes == 2:
          conv10 = SeparableConv2D(1, 1, padding = 'same')(decoder_block_1)
          output = Activation('sigmoid')(conv10)

     else:
          conv10 = SeparableConv2D(n_classes, 1, padding = 'same')(decoder_block_1)
          output = Activation('softmax')(conv10)

     #model = tf.keras.Model(input = inputs, outputs = output, name = 'Unet')
     model = Model(inputs=inputs, outputs=output, name='Unet')
     
     return model


if __name__ == '__main__':

     model = Unet()

     #print(model.summary())


#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),      # used optimizer with chosen learning rate
              loss='mse',                                                   # loss function
              metrics=['mse', 'mae'])                                       # evaluation metrics (for training and validation set)

def batch_data(loss_function, model, dataset_x, dataset_y, val_x, val_y, num_batches, epochs):
  progress_ims = [] # Create empty variable
  progress_val = []

  x = dataset_x
  x = x.reshape((num_batches, (int(len(dataset_x)/num_batches)), 30, 30, 1)) # Split the dataset into a number of batches
  y = dataset_y
  y = y.reshape((num_batches, (int(len(dataset_y)/num_batches)), 30, 30, 1)) # Split the dataset into a number of batches
  for i in range(num_batches): # Repeat for the number of batches
    print(f'Batch Number {i}/{num_batches}')
    history = model.fit(x = x[i], y = y[i], validation_data = (val_x, val_y), epochs = epochs) # Run the first batch of training and save the model training history
    output = model.predict(x[i]) # Predict the output of the model with the input xi
    loss = history.history['loss'] # Find the history of hte loss
    progress_val.append(loss[0]) # Add the first epoch loss of the batch to list
    progress_ims.append(output[0]) # Add the predicted image to list  
  print('Training phase complete.')
  progress_ims = np.array(progress_ims) # Make lsit into numpy array
  progress_val = np.array(progress_val) # Make lsit into numpy array
  return model, progress_ims, progress_val


model, progress_ims, progress_val = batch_data(loss_function = 'mse', model = model, epochs = 2, batch_number = 3) # Run training
