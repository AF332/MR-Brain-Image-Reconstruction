import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount('//content//drive')

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
    
undersamplefactor=0.4 # Can be adjusted with experimentation
undersamplefactor1D=0.4
undersamplefactor2D=0.4

def undersample(kspace, undersamplefactor):
    nlines=kspace.shape[0] # Total number of lines
    nlinesundersampled=int(nlines*(1-undersamplefactor)) # How many lines to keep
    
    randomlines=np.random.choice(nlines,nlinesundersampled,replace=False) # Select random lines

    undersampledkspace=np.zeros_like(kspace) # Create zero array
    undersampledkspace[randomlines, ...]=kspace[randomlines, ...] # Place undersampled lines in the array

    return undersampledkspace

def gaussian1Dundersample(kspace, undersamplefactor1D):
    gaussian1D=np.exp(-((np.arange(lines)-nlines/2)/(0.32*nlines/2))**2) #create a 1D gaussian distribution in the centre of kspace
    gaussian1D /=np.max(gaussian1D) #normalise the distribution to have a max value of 1

    nlinesundersampled1=int(nlines*1-undersamplefactor2) #lines to keep/lose
    randomlines2=np.random.choice(nlines,nlinesundersampled1,replace=False,p=gaussian1D)#based on 1D gaussian
    undersamplekspace1D=np.zeros_like(kspace)#create zero array with kspace shape
    undersamplekspace1D[random_lines,...]=kspace[random_lines]#place the lines in the zero array

    return undersampledkspace1D

    
def gaussian2Dundersample(kspace, undersamplefactor2D):
    nrows,ncolumns=kspace.shape #number of rows and columns in kspace array
    #create a 2D gaussian distribution centred in the middle of kspace (low frequencies)
    rows,columns=np.meshgrid(np.arange(nrows),np.arange(ncolumns)) 
    gaussian2D=np.exp(-((rows-nrows/2)**2+(columns-ncolumns/2)**2)/(0.3*rows/2)**2)

    gaussian2D/=np.max(gaussian2D) #normalize max value to 1
    nlinesundersampled2D=int(nrows*(1-undersamplefactor2D)) #how much to undersample
    random_rows=np.random.choice(nrows,nlinesundersampled2D,replace=False,p=gaussian2D.flatten())#choose lines based on 2D gaussian distribution
    undersamplekspace2D=np.zeros_like(kspace)#make zero array same as kspace
    undersamplekspace2D[random_rows,:]=kspace[random_rows,:]#add undersampled data

    return undersamplekspace2D

#==========Read Kspace Images and combine images from each coil for each acquisition and add gaussian noise
def read_source_images_noise(path, mean, standard_deviation): # mean of 0 and std of 6.5 for a good random noise
    data = read_data(path, 'kspace') # Read dataset called reconstruction_rss in the h5 file
    # Data is a numpy array: (4, 18, 4, 256, 256) (files, acquisitions, coils, imagex, imagey)
    data_length = len(data) # store how many files have been read
    kspace_images = [] # Generate empty variable
    kspace_noise = [] # Generate empty variable
    image_images = [] # Generate empty variable
    image_noise = [] # Generate empty variable
    for i in range(data_length): # Repeat for the number of files in folder
        d = data[i] # Save d as (18, 4, 256, 256)
        num_acq = d.shape[0] # Number of acquisitions in each file
        for j in range(num_acq):  # Repeat for number of acquisitions in each file
            acquisition = d[j, ...] # Seperate image based on loop number
            coil0 = acquisition[0, ...] # Save data in row 0 as 'coil0'
            coil1 = acquisition[1, ...] # Save data in row 1 as 'coil1'
            coil2 = acquisition[2, ...] # Save data in row 2 as 'coil2'
            coil3 = acquisition[3, ...] # Save data in row 3 as 'coil3'
            kspace = (coil0+coil1+coil2+coil3)/4 # Data from coils added and averaged
            
            undersampledkspace=undersample(kspace, undersamplefactor)  ####### USE THIS INSTEAD OF KSPACE
            undersampledkspace1D=gaussian1Dundersample(kspace, undersamplefactor1D)
            undersampledkspace2D=gaussian2Dundersample(kspace, undersamplefactor2D)

            noise = np.random.normal(mean, standard_deviation, (kspace.shape)).astype(np.uint8) # Change this into MRI related noise. Change into 8-bit values to keep values from 0-255
            im_noise = kspace + noise # Add noise to the kspace
            temp_im_noise = np.fft.ifft2(im_noise) # Translate into image domain
            temp_im_noise = np.fft.fftshift(temp_im_noise) # Reshape data into image
            im_image = np.fft.ifft2(kspace) # Translate into image domain
            im_image = np.fft.fftshift(im_image) # Reshape data into image
            kspace_images.append(kspace) # Adding kspace images into the list
            kspace_noise.append(im_noise) # Adding noisy kspace images into the list
            image_images.append(im_image) # Adding images to the list 
            image_noise.append(temp_im_noise) # Adding images with noise to list
    kspace_images = np.array(kspace_images) # At the end of function turn list of images into numpy array
    kspace_noise = np.array(kspace_noise) # At the end of function turn list of images into numpy array
    image_images = np.array(image_images) # At the end of function turn list of images into numpy array
    image_noise = np.array(image_noise) # At the end of function turn list of images into numpy array
    return kspace_images, kspace_noise, image_images, image_noise # Finnish the function with the four variables


def view_data(kspace_images, kspace_noise, display_acqisition): # Kspace_ image and kspace_noise should be inputted as (number of images, imageX, imageY), display acquisition is the numebr of acquisitions to be dispalyed by the function.
    assert(len(kspace_images) == len(kspace_noise)) # Make sure the size of the kspace image set is the same size as the noisey kspace image set
    for i in range(display_acqisition): # Repeat for the number of acquisitons selected
        kspace = kspace_images[i] # Save the ith row of 'kspace_images' as 'kspace'
        kspace_noise = kspace_noise[i] # Save the ith row of 'kspace_noise' as 'kspace_noise'
        imdata = kspace # Make a copy of the kspace data for data processing
        image = np.fft.ifft2(imdata) # Translate into image domain
        image = np.fft.fftshift(image) # Reshape data into image
        imdata_noise = kspace_noise # Make a copy of the kspace noisy data for data processing
        image_noise = np.fft.ifft2(imdata_noise) # Translate into image domain
        image_noise = np.fft.fftshift(image_noise) # Reshape data into image
        plt.subplot(2, display_acqisition, i + 1) # Create subplot to be scaled to number of acquisitions displayed
        plt.imshow(abs(image_noise), cmap='gray') # Display positive values of image as a grayscale image
        plt.title('Noisy Image, Acquisition:' + str(i)) # Make image title with images based on number of loop
        plt.subplot(2, display_acqisition, i + 1) # Create subplot to be scaled to number of acquisitions displayed
        plt.imshow(abs(image), cmap='gray')
        plt.title('Image, Acquisition:' + str(i))
        plt.subplot(2, display_acqisition, (i + display_acqisition+1))
        plt.imshow(abs(kspace_noise), cmap='gray')
        plt.title('Noisy Kspace, Acquisition:' + str(i))
        plt.subplot(2, display_acqisition, (i + display_acqisition+1))
        plt.imshow(abs(kspace), cmap='gray')
        plt.title('Kspace, Acquisition:' + str(i))
plt.tight_layout() # Evenly space the subplots on the figure
plt.show()

def view_image_data(image_images, image_noise, display_acqisition): # Kspace_ image and kspace_noise should be inputted as (number of images, imageX, imageY), display acquisition is the numebr of acquisitions to be dispalyed by the function.
    assert(len(image_images) == len(image_noise)) # Make sure the size of the kspace image set is the same size as the noisey kspace image set
    for i in range(display_acqisition): # Repeat for the number of acquisitons selected
        im = image_images[i] # Save the ith row of 'kspace_images' as 'kspace'
        im_noise = image_noise[i] # Save the ith row of 'kspace_noise' as 'kspace_noise'
        plt.subplot(2, display_acqisition, i + 1) # Create subplot to be scaled to number of acquisitions displayed
        plt.imshow(abs(im_noise), cmap='gray') # Display positive values of image as a grayscale image
        plt.title('Noisy Image, Acquisition:' + str(i)) # Make image title with images based on number of loop
        plt.subplot(2, display_acqisition, i + display_acqisition + 1) # Create subplot to be scaled to number of acquisitions displayed
        plt.imshow(abs(im), cmap='gray')
        plt.title('Image, Acquisition:' + str(i))
plt.tight_layout() # Evenly space the subplots on the figure
plt.show()

def view_epoch_image(image_images, image_noise, display_acquisition): # View images noisy images and images from epoch determined by the input 'display_acquisition'
    assert(len(image_images) == len(image_noise)) # Make sure the size of the kspace image set is the same size as the noisey kspace image set

    for i in range(len(data)): # Repeat for the number of acquisitons selected
        im = predicted_output[i] # Save the ith row of 'kspace_images' as 'kspace'
        im_noise = image_noise[i] # Save the ith row of 'kspace_noise' as 'kspace_noise'
        plt.subplot(2, display_acquisition, i + 1) # Create subplot to be scaled to number of acquisitions displayed
        plt.imshow(abs(im_noise), cmap='gray') # Display positive values of image as a grayscale image
        plt.title('Noisy Image, Acquisition:' + str(i)) # Make image title with images based on number of loop
        plt.subplot(2, display_acquisition, i + display_acquisition + 1) # Create subplot to be scaled to number of acquisitions displayed
        plt.imshow(abs(im), cmap='gray')
        plt.title('Image, Acquisition:' + str(i))
        i + (len(data)/display_acquisition)
plt.tight_layout() # Evenly space the subplots on the figure
plt.show()
