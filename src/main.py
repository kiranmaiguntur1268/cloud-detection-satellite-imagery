Cloud Detection in Satellite Imagery-Based on Deep Learning
Code:
#Import the all libraries

import numpy as np # NumPy is usually imported under the np alias
import os # os module to interact with the underlying operating system.0
import re # regular expressions
import matplotlib.pyplot as plt # is a collection of functions that make matplotlib work like MATLAB
from sklearn.model_selection import train_test_split #  splits arrays or matrices into random subsets for train and test data, respectively
from sklearn.metrics import classification_report # import metrics in SciKit Learn API to evaluate your machine learning algorithms
import random # random module, which contains a number of random number generation-related functions.
import pandas as pd # to use functions for working with date values

#Import images
dirname = os.path.join(os.getcwd(), 'ISatelitales')
imgpath = dirname + os.sep # "dirname" and "os.sep," to create a path to an image file
imag = [] # Load and store an image in the "imag" 
IDcat = []; 
IDim = []; idim = 0 # initializes two variables, "IDim" and "idim," in a programming language
for ruta, carpetas, filenames in os.walk(imgpath): # path, folder and file names:
    print(ruta,idim)# "ruta" and "idim" have been assigned values earlier in the code, the print statement would display those values.
    for filename in filenames: # looking at all files individually    
        if ruta == imgpath +'cloud':  # if the image is in the cloud folder
            IDim.append(idim) ; idim += 1
            IDcat.append(1) # if the image have clouds IDcat value is 1
            filepath = os.path.join(ruta, filename) #obtain the image direction 
            imagen = plt.imread(filepath) #obtain the image array.
            imag.append(imagen) #save the image in images list
        else: 
            IDim.append(idim) ; idim += 1
            IDcat.append(0) # if the image have clouds IDcat value is 0
            filepath = os.path.join(ruta, filename) #obtain the image direction 
            imagen = plt.imread(filepath) #obtain the image array.
            imag.append(imagen) #save the image in images list
print('Total images: ',idim)
print('Cloud images: ',sum(IDcat))
cat=IDcat
imagenes=imag# the variables "cat" and "imagenes" are being assigned values based on the variables "IDcat" and "imag
#Cloud-free images filtering function
def filter_proportion(dat_set_cat, dat_set_im, rat_acept): 
#dat_set_cat: A list or array containing categorical labels (0 for cloud-free, 1 for cloud).
#dat_set_im: A list or array containing images (assumed to be corresponding to the categorical labels).  
 #rat_acept: A floating-point number representing the acceptance ratio for cloud-free images. Cloud-free images will be included in the output with a probability less than rat_acept.
Cat = [] 
    Im = []
    con1= 0 #counter Cloud-free images
    con2= 0 #counter cloud images
    lim = len(dat_set_cat)
    for i in range(lim) :
        if dat_set_cat[i] == 0: # The function iterates through the input dataset (dat_set_cat and dat_set_im) using a loop.
            R=random.random()
            if R < rat_acept:
                con1 += 1
                Im.append(dat_set_im[i])
                Cat.append(dat_set_cat[i])
        elif dat_set_cat[i] == 1:
            con2 += 1
            Im.append(dat_set_im[i])
            Cat.append(dat_set_cat[i])
    Cat = np.array(Cat)  # The function converts the lists Cat and Im into NumPy arrays for more efficient handling.
    Im = np.array(Im, dtype=np.uint8)
    print ('Ratio cloud images: ',con2/(con1+con2), '%')
    del dat_set_im; del dat_set_cat  # The original input datasets (dat_set_cat and dat_set_im) are deleted from memory using del to free up memory.
    return Im, Cat
#Separate the data set into data subsets
# training set, validation set, and test set.
Im,valIm1,Cat,valCat1 = train_test_split(imagenes,cat,test_size=0.3) 
# train_test_split: This function is part of the scikit-learn library, which is used for machine learning tasks in Python. It takes input arrays or lists (imagenes and cat) and splits them into random train and validation sets based on the specified test_size. Here, test_size=0.3 indicates that 30% of the data will be used for validation, and the remaining 70% will be used for training.
del imagenes; del cat; # imagenes and cat are deleted from memory using the del keyword to free up memory space.
entrIm1,testIm1,entrCat1,testCat1 = train_test_split(Im,Cat,test_size=0.3) #subsets train test
del Im; del Cat
#Filtering unclouded images from subsets
valIm, valCat = filter_proportion(valCat1, valIm1, rat_acept=0.07)# filter_proportion function to filter the validation dataset (valCat1 and valIm1) based on a specified acceptance ratio (rat_acept=0.07)
 # Filter the validation dataset based on the acceptance ratio
plt.figure(figsize=(20,10))  # Plot the filtered images and their categorical labels
col = 5 # Number of columns in the plot grid
for i in range(col):
    plt.subplot(5/col+1,col,i+1) # This ensures that the images are displayed in a 5x5 grid layout.
    plt.imshow(valIm[i]) # This line displays the i-th filtered image from valIm.
    plt.title(valCat[i])

#Image processing
valIm = valIm / 255. # the data arrays valIm, entrIm, and testIm are normalized to have values between 0 and 1 by dividing each element by 255.
entrIm = entrIm / 255.
testIm = testIm / 255.
#Develop the CNN
import keras
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Sequential,Model
from tensorflow.keras.layers import InputLayer
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization #Batch normalization is a technique used to improve the convergence and training stability of neural networks.
from keras.layers import LeakyReLU # The Leaky ReLU is a variant of the Rectified Linear Unit (ReLU) activation function.
from keras.models import load_model 

#Create the CNN structure
modelo2 = Sequential()#A sequential model is a linear stack of layers, where you can add layers one by one.
modelo2.add(Conv2D(60, kernel_size=(3, 3),activation='relu',padding='same',input_shape=(256,256,3)))
modelo2.add(MaxPooling2D((4, 4)))# This adds a max-pooling layer with a pooling size of (4, 4).
modelo2.add(Conv2D(120, (3, 3), activation='relu')) #2D convolutional layer with 200 filters and a kernel size of (3, 3), followed by a ReLU activation function.
modelo2.add(MaxPooling2D((2, 2)))
modelo2.add(Dropout(0.5))
modelo2.add(Conv2D(200, (3, 3), activation='relu'))
modelo2.add(MaxPooling2D((2, 2)))
modelo2.add(Conv2D(250, (3, 3), activation='relu'))
modelo2.add(Dropout(0.5))
modelo2.add(Flatten())
modelo2.add(Dense(512, activation='relu')) #This adds a fully connected layer with 512 neurons and a ReLU activation function.
modelo2.add(Dense(1,activation='sigmoid'))
modelo2.summary() #This will display a concise overview of the layers in the model, the number of parameters, and the output shapes at each layer
#Training the CNN
modelo2.compile(optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc']) # This compiles the model "modelo2" before training. In this step, you specify the optimizer, loss function, and evaluation metrics for the model.
modelo2a = modelo2.fit(x=entrIm, y=entrCat, batch_size=32, epochs=30, verbose=1, validation_data=(testIm, testCat), shuffle=True)# The method adjusts the model's parameters to minimize the specified loss function while maximizing the accuracy on the training data.
modelo2.save("modF6.h5py")#saves the trained Keras model "modelo2" to a file named "modF6.h5py" in the Hierarchical Data Format (HDF5) format.

#CNN Evaluation
test_eval = modelo2.evaluate(valIm, valCat, verbose=1)# prints the validation loss and accuracy
print('Validation loss:', test_eval[0])
print('Validation accuracy:', test_eval[1])
val_loss = modelo2a.history['val_loss'] #loss value
val_acc = modelo2a.history['val_acc'] #acc value
loss = modelo2a.history['loss'] #historical loss array
acc = modelo2a.history['acc'] #historical acc array
X1a1 = range(1, len(acc)+1)
plt.plot(X1a1, acc,'b', label='Training accurarcy')# to plot the training accuracy and validation accuracy against the epoch number.
plt.plot(X1a1, val_acc,'r', label='Test accurarcy')
plt.title('Training and Test accurarcy')
plt.legend()
plt.figure()
plt.plot(X1a1, loss, 'b', label='Training loss')
plt.plot(X1a1, val_loss, 'r',label='Test loss')
plt.title('Training and Test loss')
plt.legend()
plt.show() #used to display the two plots of training and validation accuracy and loss.



#Validation and discussion
nub_pred = modelo2.predict(valIm, batch_size=32, verbose=1)# make predictions on the validation dataset "valIm". The predictions are then processed to get the final predicted class labels 
nub_predicted = np.argmax(nub_pred, axis=1)  


#ROC curve values

fpr, tpr, thresholds = roc_curve(valCat , nub_pred) # valCat and the predicted probabilities nub_pred as inputs. The function returns three arrays: fpr (false positive rate), tpr (true positive rate), and thresholds
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'-')
plt.xlabel('Ratio (FPR)')# This sets the label for the x-axis of the plot.
plt.ylabel('Sensibilidad (VPR)')# This sets the label for the y-axis of the plot.
plt.title('Curva ROC')
plt.show()#This displays the ROC curve plot.

AUC=roc_auc_score ( valCat ,  nub_pred) # valCat and the predicted probabilities nub_pred as inputs. The function calculates the AUC based on the ROC curve.
print("Area de la curva ROC=",AUC)# prints the calculated AUC value

dim = valCat.shape[0] # stores the number of samples in the validation dataset.
#nv, nf, dv, df, na, cn, and cd are initialized to count the occurrences of true positive (nv), false positive (nf), true negative (dv), false negative (df), and number of accurate (na) predictions, as well as the total number of positive samples (cn) and the total number of samples that were classified as negative (cd).
nv = 0; nf = 0; dv= 0; df = 0; na=0
cn = 0; cd= 0
inf=[]; idf=[]
for i in range(dim):
    if nub_pred[i] >= 0.5 and valCat[i] == 1:
        nv +=1
        cn +=1
    elif nub_pred[i] >= 0.5 and valCat[i] == 0:
        nf +=1
        inf.append(valIm[i])
        cd +=1
    elif nub_pred[i] < 0.5 and valCat[i] == 0:
        dv +=1
        cd +=1
    else :
        df +=1
        idf.append(valIm[i])
        cn +=1
    
print('True Positive TF=',nv) #The number of samples correctly classified as positive.
print('False Positive FP=',nf)# The number of samples incorrectly classified as positive
print('True Negative TN=',dv)# The number of samples correctly classified as negative.
print('False Negative FN=',df)# The number of samples incorrectly classified as negative.
print('Sensitivity VPR=',nv/cn)# True Positive Rate (TPR) or Recall, it is the proportion of positive samples correctly classified.
print('False positive rate FPR=',nf/cd) # The proportion of negative samples incorrectly classified as positive.
print('Accuracy  ACC=',(nv+dv)/(cn+cd)) # The overall proportion of correctly classified samples.
print('Specificity SPC=',dv/cd)
print('Positive Predictive Value PPV=',nv/(nv+nf))
print('Negative Predictive Value NPV=',dv/(dv+df))
print('Ratio FDR=',nf/(nf+nv))
