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
from scipy.ndimage import rotate
#'''
"""from google.colab import drive
drive.mount('//content//drive')"""
#'''

''' For data on the computer
path_training = 'C://Users//gabes//Desktop//Data//Training'
path_testing = 'C://Users//gabes//Desktop//Data//Testing'
'''
#'''
path_training = 'E:\Data\Training'
path_testing = 'E:\Data\Testing'
"""path_training = '/content/drive/MyDrive/4th Year/DeepLearningProject/Data (1)/Training'
path_testing = '/content/drive/MyDrive/4th Year/DeepLearningProject/Data (1)/Testing'"""
#'''


def undersample(kspace, undersamplefactor = 2):
  undersample_kspace = kspace
  undersample_kspace[::undersamplefactor, ::undersamplefactor] = 0
  return undersample_kspace


#===========Read all data from files contained in folder from path
def read_data(path, keyname):
  data = [] # Create an empty variable
  h5name = os.listdir(path) # Read all the names of data in file (path) and store as a list
  for i in range(int(len(h5name)/2)): # Repeat for every file in folder from path
    temp_path = path + '//' + str(h5name[i]) # Create a path for each file in the folder
    tempdata = h5py.File(temp_path, "r") # Open h5 file as "read" and save as data
    tempkeydata = tempdata[keyname] # Store data from 'kspace' key as kspacedata
    data.append(tempkeydata) # Add data from key to list
  data = np.array(data) # Turn list into numpy array
  return data

#==========Read target images and input Kspace images combining data from each coil, add gaussian noise and transform into the image domain
def read_source_images_noise(path, mean, standard_deviation):
  kdata = read_data(path, 'kspace') # Read dataset called reconstruction_rss in the h5 file
  data = read_data(path, 'reconstruction_rss') # Read dataset called reconstruction_rss in the h5 file
  data_length = len(kdata) # store how many files have been read
  assert(len(kdata) == len(data)) # Make sure the number of kspace images and regular images is equal
  image_input = [] # Create an empty list
  image_target = []
  for i in range(data_length): # Repeat for the number of files in folder
    kd = kdata[i] # Save the ith instance of kspace data as kd
    d = data[i] # save the ith instance of data as d
    num_acq = kd.shape[0] # Number of acquisitions in each file
    for j in range(int(num_acq/2)):  # Repeat for number of acquisitions in each file/2 (cutting out non brain structures in the images)
      image = d[j, ...] # Save the jth instance of d as image
      kspace_acquisition = kd[j, ...] # Seperate image based on loop number, this has 4 coil kspace images
      kspace = np.mean(kspace_acquisition, axis=0) # Find the mean average of all coil images
      noise = abs(np.random.normal(mean, standard_deviation, (kspace.shape)).astype(np.uint8)) # Change this into MRI related noise. Change into 8-bit values to keep values from 0-255
      im_noise = kspace + noise # Add noise to the kspace
      un_im_noise = undersample(im_noise)
      temp_im_noise = np.fft.ifft2(un_im_noise) # Translate into image domain
      temp_im_noise = np.fft.fftshift(temp_im_noise) # Reshape data into image
      image_input.append(abs(temp_im_noise)) # Add the resultant noisy image into the input list
      image_target.append(image) #  Add the complete image to the target list
  return np.array(image_target), np.array(image_input) # Change lists into numpy arrays


#==========Normalise data by applying the min-max scaling formula
def min_normalise_dataset(dataset): # Uisng min max scaling
  min_max_images = [] # Create an empty list
  for i in range(dataset.shape[0]): # Repeat for number of images in dataset
    im = dataset[i] # Isolate the ith image in 'dataset'
    min = np.min(im) # Find the minimum value of the image values
    max = np.max(im) # Find the maximum value of the image values
    min_max = (im-min)/(max-min) # Apply the mim-max scaling formuala to get normalised image
    min_max_images.append(min_max) # Add the image to the list
  return np.array(min_max_images) # Change the list into numy array



train_target, train_input = read_source_images_noise(path_training, 0, 0.01)
test_target, test_input = read_source_images_noise(path_testing, 0, 0.01)

train_input = min_normalise_dataset(train_input)
train_target = min_normalise_dataset(train_target)
test_input = min_normalise_dataset(test_input)
test_target = min_normalise_dataset(test_target)

def conv_block(inputs=None, n_filters=128, batch_norm=True, dropout_prob=0.2):
    convolutional_1 = SeparableConv2D(n_filters, 3, padding='same', kernel_initializer='HeNormal')(inputs)
    if batch_norm:
        convolutional_1 = BatchNormalization(axis=-1)(convolutional_1)
    convolutional_1 = LeakyReLU(alpha=0.2)(convolutional_1)

    convolutional_2 = SeparableConv2D(n_filters, 3, padding='same', kernel_initializer='HeNormal')(convolutional_1)
    if batch_norm:
        convolutional_2 = BatchNormalization(axis=-1)(convolutional_2)
    convolutional_2 = LeakyReLU(alpha=0.2)(convolutional_2)

    if dropout_prob > 0:
        convolutional_2 = Dropout(dropout_prob)(convolutional_2)

    return convolutional_2


def encoder_block(inputs=None, n_filters=128, batch_norm=True, dropout_prob=0.2):
    skip_con = conv_block(inputs, n_filters, batch_norm, dropout_prob)
    # next_layer = SeparableConv2D(n_filters, 3, strides = 2, padding = 'same')(skip_con)
    next_layer = MaxPooling2D((2, 2))(skip_con)

    return next_layer, skip_con


def decoder_block(expansive_input, skip_con, n_filters, batch_norm=True, dropout_prob=0.2):
    up_samp = Conv2DTranspose(n_filters, 3, strides=2, padding='same', kernel_initializer='HeNormal')(expansive_input)
    sum = concatenate([up_samp, skip_con], axis=-1)
    convolution = conv_block(sum, n_filters, batch_norm, dropout_prob)

    return convolution


def Unet(input_size=(256, 256, 1), n_filters=128, n_classes=2, batch_norm=True, dropouts= [0.2]*9):
    inputs = Input(input_size)

    encoder_block_1 = encoder_block(inputs, n_filters, batch_norm, dropout_prob=dropouts[0])
    encoder_block_2 = encoder_block(encoder_block_1[0], n_filters * 2, batch_norm, dropout_prob=dropouts[1])
    encoder_block_3 = encoder_block(encoder_block_2[0], n_filters * 4, batch_norm, dropout_prob=dropouts[2])
    encoder_block_4 = encoder_block(encoder_block_3[0], n_filters * 8, batch_norm, dropout_prob=dropouts[3])

    bridge = conv_block(encoder_block_4[0], n_filters * 16, batch_norm, dropout_prob=dropouts[4])

    decoder_block_4 = decoder_block(bridge, encoder_block_4[1], n_filters * 8, batch_norm, dropout_prob=dropouts[5])
    decoder_block_3 = decoder_block(decoder_block_4, encoder_block_3[1], n_filters * 4, batch_norm,
                                    dropout_prob=dropouts[6])
    decoder_block_2 = decoder_block(decoder_block_3, encoder_block_2[1], n_filters * 2, batch_norm,
                                    dropout_prob=dropouts[7])
    decoder_block_1 = decoder_block(decoder_block_2, encoder_block_1[1], n_filters, batch_norm,
                                    dropout_prob=dropouts[8])

    if n_classes == 2:
        conv10 = SeparableConv2D(1, 1, padding='same')(decoder_block_1)
        output = Activation('sigmoid')(conv10)
    else:
        conv10 = SeparableConv2D(n_classes, 1, padding='same')(decoder_block_1)
        output = Activation('softmax')(conv10)
    # model = tf.keras.Model(input = inputs, outputs = output, name = 'Unet')
    model = Model(inputs=inputs, outputs=output, name='Unet')
    return model


if __name__ == '__main__':
    model = Unet()
    # print(model.summary())

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # used optimizer with chosen learning rate
              loss='mse',  # loss function
              metrics=['mse', 'mae'])

"""The ephocs refer to the iterations. During each of this, the model processes the entire training dataset. """
history = model.fit(x = train_input, y = train_target, batch_size = 32, epochs = 6, verbose = 1, validation_split = 0.2)

result_images = model.predict(test_input) # Find the outputs of the model based on the test inputs

model.evaluate(test_input, test_target) # Find the loss for the test dataset

# Save the model with parameters after training
#model.save('E:') # Change the path to specify a place to save the trained model

#==========Plot the input, target and output
plt.subplot(2,3,1)
plt.imshow(abs(test_input[17]), cmap = 'gray')
plt.title('Network input')
plt.subplot(2,3,2)
plt.imshow(result_images[17], cmap = 'gray')
plt.title('Network output')
plt.subplot(2,3,3)
plt.imshow(test_target[17], cmap = 'gray')
plt.title('Target Image')
plt.subplot(2,3,4)
plt.imshow(abs(test_input[10]), cmap = 'gray')
plt.title('Network input')
plt.subplot(2,3,5)
plt.imshow(result_images[10], cmap = 'gray')
plt.title('Network output')
plt.subplot(2,3,6)
plt.imshow(test_target[10], cmap = 'gray')
plt.title('Target Image')
plt.tight_layout()

#==========Plot the loss convergence
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Value')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
