import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
from PIL import ImageFilter
from matplotlib import pyplot as plt
import colorsys
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from skimage.filters import gabor, gaussian

from pywt import dwt2
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow_io as tfio

import tensorflow_io as tfio
# Parameters Based on Paper
epsilon = 1e-7
m_plus = 0.8
m_minus = 0.5
lambda_ = 0.9
alpha = 0.001
epochs = 50
no_of_secondary_capsules = 2

optimizer = tf.keras.optimizers.Adam()

params = {
    "no_of_conv_kernels": 256,
    "no_of_primary_capsules": 32,
    "no_of_secondary_capsules": 2,
    "primary_capsule_vector": 32,
    "secondary_capsule_vector": 64,
    "r":5,
}
class CapsuleNetwork(tf.keras.Model):
    def __init__(self, no_of_conv_kernels, no_of_primary_capsules, primary_capsule_vector, no_of_secondary_capsules, secondary_capsule_vector, r):
        super(CapsuleNetwork, self).__init__()
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_capsules = no_of_primary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.secondary_capsule_vector = secondary_capsule_vector
        self.r = r
        
        
        with tf.name_scope("Variables") as scope:
            self.convolution = tf.keras.layers.Conv2D(self.no_of_conv_kernels, [9,9], strides=[1,1], name='ConvolutionLayer', activation='relu')
            self.primary_capsule = tf.keras.layers.Conv2D(self.no_of_primary_capsules * self.primary_capsule_vector, [9,9], strides=[2,2], name="PrimaryCapsule", padding= "same")
            self.w = tf.Variable(tf.random_normal_initializer()(shape=[1, 4608, self.no_of_secondary_capsules, self.secondary_capsule_vector, self.primary_capsule_vector]), dtype=tf.float32, name="PoseEstimation", trainable=True)
            self.dense_1 = tf.keras.layers.Dense(units = 256, activation='relu')
            self.dropout_1 = tf.keras.layers.Dropout(0.5)
            self.dense_2 = tf.keras.layers.Dense(units = 512, activation='relu')
            self.dropout_2 = tf.keras.layers.Dropout(0.5)
            self.dense_3 = tf.keras.layers.Dense(units = 1024 , activation='sigmoid', dtype='float32')
            
    def build(self, input_shape):
        pass
        
    def squash(self, s):
        with tf.name_scope("SquashFunction") as scope:
            s_norm = tf.norm(s, axis=-1, keepdims=True)
            return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + epsilon)
    
    
    def call(self, inputs):
        input_x, y = inputs
        
        x = self.convolution(input_x)
        x = self.primary_capsule(x) # x.shape: (None, 6, 6, 256) mais avec (64,64) , (None, 56,56,256)
    
        
        with tf.name_scope("CapsuleFormation") as scope:
            X_gabor = tfio.experimental.filter.gabor(x, freq=0.7 , theta=0.9 ).numpy()
            X_gabor = tf.cast(X_gabor, dtype="float32")
            
            u = tf.reshape(X_gabor, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 32)) # u.shape: (None, 1152, 8)
            u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
            u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
            u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 1152, 10, 16, 1)
            u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 1152, 10, 16)
            
        
        with tf.name_scope("DynamicRouting") as scope:
            
            b = tf.zeros((x.shape[0], 4608, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
         
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
                v = self.squash(s) # v.shape: (None, 1, 10, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                
                b += agreement
                
        with tf.name_scope("Masking") as scope:
            y = tf.expand_dims(y, axis=-1) # y.shape: (None, 10, 1)
            y = tf.expand_dims(y, axis=1) # y.shape: (None, 1, 10, 1)
            mask = tf.cast(y, dtype=tf.float32) # mask.shape: (None, 1, 10, 1)
            v_masked = tf.multiply(mask, v) # v_masked.shape: (None, 1, 10, 16)
            
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            self.dropout_1
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            self.dropout_2
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, 784)
        
        return v, reconstructed_image

   
    def predict_capsule_output(self, inputs):
        
        x = self.convolution(inputs)
        x = self.primary_capsule(x) # x.shape: (None, 6, 6, 256)
        
        with tf.name_scope("CapsuleFormation") as scope:
            x_g = tfio.experimental.filter.gabor(x, freq=0.7 , theta=0.9 ).numpy()
            X_gabor = tf.cast(x_g, dtype="float32")
            u = tf.reshape(X_gabor, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 32)) # u.shape: (None, 1152, 8)
            
            u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
            u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
            u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 1152, 10, 16, 1)
            u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 1152, 10, 16)
            
        
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((x.shape[0], 4608, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
            
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
                v = self.squash(s) # v.shape: (None, 1, 10, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                
                b += agreement
        return v

    def regenerate_image(self, inputs):
       
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            self.dropout_1
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            self.dropout_2
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, 784)
        return reconstructed_image

def Preprocessing(img):
    img = cv.resize(img, (64,64))
    #conversion vers Image
    #print("shape image ", img.shape)
    image = Image.fromarray(img)
    #print("image size : ", image.size)

    #Calcul des coordonnées pour couper le milieu de l'image
    width = 64 
    height = 64
    left = (width - 32) // 2
    top = (height - 32) // 2
    right = left + 32
    bottom = top + 32

    #Découpage de l'image
    sub_image = image.crop((left, top, right, bottom))

    #conversion vers numpy
    np_image = np.asarray(sub_image)
    #print("shape np image ", np_image.shape)
    np_image = cv.GaussianBlur(np_image, (3,3 ), 3)

    np_image = np_image / 255.0
    np_image = tf.cast(np_image, dtype=tf.float32)
    np_image = tf.expand_dims(np_image, axis=-1)
    return np_image


def safe_norm(v, axis=-1, epsilon=1e-7):
    v_ = tf.reduce_sum(tf.square(v), axis = axis, keepdims=True)
    return tf.sqrt(v_ + epsilon)

def predict(model, x):
    #print(type(x))
    #print(x.shape)
    pred = safe_norm(model.predict_capsule_output(x))
    pred = tf.squeeze(pred, [1])
    #print(pred)
    return np.argmax(pred, axis=1)[:,0]


def predictP(model, x):
    #print(type(x))
    pred = safe_norm(model.predict_capsule_output(x))
    pred = tf.squeeze(pred, [1])
    #print(pred)
    return pred


def predict_proba(model, x):
    #print(type(x))
    pred = safe_norm(model.predict_capsule_output(x))
    pred = tf.squeeze(pred, [1])
    return pred, np.argmax(pred, axis=1)[:,0]



def final_pred(imgX, imgY, imgZ, modelx, modely, modelz):
    imgX = Preprocessing(imgX)
    imgY = Preprocessing(imgY)
    imgZ = Preprocessing(imgZ)
    probaX, valX = predict_proba(modelx, tf.expand_dims(imgX, 0))
    probaY, valY = predict_proba(modely, tf.expand_dims(imgY, 0))
    probaZ, valZ = predict_proba(modelz, tf.expand_dims(imgZ, 0))
    somme = 0
    if (valX[0] == valY[0]) and (valY[0]== valZ[0]):
        return valX[0]
    else:
        #print("ici")
        somme  = (1.5* np.max(probaX)) + (1 * np.max(probaY)) + (2 * np.max(probaZ))
        #print(np.max(probaX))
        final_pred = somme / (1.5+1+2)
        #print(final_pred)
        #print(XM_trainY[i])
        #print(final_pred)
        if final_pred <=0.73:
            return 1
        else:
            return 0
        
    return valX[0]

def final_predMAJOR(imgX, imgY, imgZ, modelx, modely, modelz):
    
    imgX = Preprocessing(imgX)
    imgY = Preprocessing(imgY)
    imgZ = Preprocessing(imgZ)
    probaX, valX = predict_proba(modelx, tf.expand_dims(imgX, 0))
    probaY, valY = predict_proba(modely, tf.expand_dims(imgY, 0))
    probaZ, valZ = predict_proba(modelz, tf.expand_dims(imgZ, 0))
    somme = 0
    if (valX[0] == valY[0]) :
        #list_def_ref.append([i, imgX, imgY, None])
        return valX[0]
    else:
        if (valY[0]== valZ[0]):
            #list_def_ref.append([i, None,imgY, imgZ])
            return valY[0]
        else:
            if (valX[0]== valZ[0]):
                #list_def_ref.append([i, imgX,None, imgZ])
                return valX[0]
            
            
def final_predSTRICT(imgX, imgY, imgZ, modelx, modely, modelz):
    imgX = Preprocessing(imgX)
    imgY = Preprocessing(imgY)
    imgZ = Preprocessing(imgZ)
    probaX, valX = predict_proba(modelx, tf.expand_dims(imgX, 0))
    probaY, valY = predict_proba(modely, tf.expand_dims(imgY, 0))
    probaZ, valZ = predict_proba(modelz, tf.expand_dims(imgZ, 0))
    somme = 0
    if (valX[0] == valY[0]) and (valY[0]== valZ[0]):
        return valX[0]
    else:
        return 1
    
    
def calcEnergy(Image):

    img = Image
    img = np.expand_dims(img, 0)
    #print(img.shape)
    #img = np.expand_dims(img, -1)
    filtered_img = tfio.experimental.filter.gabor(img, freq=0.7 , theta=0.9 )
    #print(filtered_img.shape)
    #* reduire la dimmension de l'image
    filtered_img = tf.squeeze(filtered_img, 0)
    filtered_img = tf.cast(filtered_img, dtype="float32")
    return np.sum(np.abs(filtered_img))

from scipy.spatial.distance import directed_hausdorff
def calcHaussdorff(image_reference, image_test2):
    image_ref = tf.squeeze(image_reference, -1)
    image_test = tf.squeeze(image_test2, -1)


    return max(directed_hausdorff(image_test, image_ref)[0], directed_hausdorff(image_ref, image_test)[0])


import numpy.linalg as npl
import cv2
def calcRho(image_reference, image_test2):
    B = tf.squeeze(image_reference, -1)
    A = tf.squeeze(image_test2, -1)
    #AtA - BtB
    mat1 = np.subtract( np.dot(np.transpose(A),A) ,  np.dot(np.transpose(B),B))

    #rho(ATA)
    (valp,vecp)=npl.eig(mat1)
    
    return max(abs(valp))

#LA TEXTURE
from skimage.feature import greycomatrix, greycoprops

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def Texture(image_test2):
    # Lecture de l'image
    imgg = image_test2[:,:,0]
    img = (imgg * 255).astype(np.uint8)
    #print("hey",img.shape)

    # Calcul de la matrice de co-occurrence de niveaux de gris
    glcm = greycomatrix(img, [1], [0], levels=256)

    # Calcul des propriétés de texture à partir de la matrice de co-occurrence
    
    homogeneity = greycoprops(glcm, 'homogeneity')
    homogeneity = homogeneity[0][0]
    energy = greycoprops(glcm, 'energy')
    energy = energy[0][0]
    #print("energy",energy)

    # Affichage des résultats
    #print("Contrast: ", contrast)
    #print("Homogeneity: ", homogeneity)
    #print("Energy: ", energy)
    return energy,homogeneity
def Contours(image_test2):
    # Lecture de l'image
    imgg = image_test2[:,:,0]
    img = (imgg * 255).astype(np.uint8)

# Application du filtre de Sobel pour détecter les bords de l'image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.addWeighted(sobelx, 0.3, sobely, 0.3, 0)
    sobel = cv2.convertScaleAbs(sobel)

# Seuillage de l'image pour ne garder que les bords les plus significatifs
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Application d'un filtre morphologique pour supprimer les petits artefacts
    kernel = np.ones((7,7),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


    # Extraction des contours des vaisseaux sanguins
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #plt.imshow(thresh,'gray')
    #print(len(contours))
    return len(contours)

def divideReference(DataImages, DataLabels, dataReference):
    zero_list =[]
    one_list = []
    for i in range(dataReference.shape[0]):
        if DataLabels[dataReference[i]] == 0:
            zero_list.append(DataImages[dataReference[i]])
        else:
            one_list.append(DataImages[dataReference[i]])
    return np.array(zero_list),np.array(one_list)
import nashpy as nash
def thj_fonction( imgX, imgY, imgZ,valX,valY,valZ):
    
    X_refZ= np.load("THJdata/X_refZ.npy", allow_pickle=True)
    X_refO = np.load("THJdata/X_refO.npy", allow_pickle=True)
    Y_refZ = np.load("THJdata/Y_refZ.npy", allow_pickle=True)
    Y_refO = np.load("THJdata/Y_refO.npy", allow_pickle=True)
    Z_refZ = np.load("THJdata/Z_refZ.npy", allow_pickle=True)
    Z_refO = np.load("THJdata/Z_refO.npy", allow_pickle=True) 
    
    #--------------------------------------------------------------------------------    
    #Calcul de RHE pour X avec NORMALISATION:
    ## Energy max
    ListEnergy = []
 
    ## Hauss max
    ListHauss = []
    
    ## Max RHO
    ListRHO = []
    
    ## EnergyCo-occurence max
    ListEnergyCo = []
    
    if(valX == 0):
        # Boucle pour l'energie, hauss, rho
        #for i in range(X_refZ.shape[0]):
        #    ListEnergy.append(calcEnergy(X_refZ[i]))
        
        for i in range(X_refZ.shape[0]):
            ListHauss.append(calcHaussdorff(X_refZ[i], imgX))
            
        for i in range(X_refZ.shape[0]):
            ListRHO.append(calcRho(X_refZ[i], imgX))
            
        #for i in range(X_refZ.shape[0]):
         #   ListEnergyCo.append(Texture(X_refZ[i]))    
        # Fin
        #------------------------------------------------------------
        
        #Calcul du rho pour X avec X_refZ 
        
        RX = min(ListRHO)/max(ListRHO)
        #Calcul du dist Rhoss pour X avec X_refZ
        HX = min(ListHauss)/max(ListHauss)
        #Calcul du Energy pour X avec X_refZ
        #EX = calcEnergy(imgX)/max(ListEnergy)
        #Calcul du EnergyCo pour X avec X_refZ
        imgXX = imgX.numpy()
        EXC,OMX = Texture(imgXX)
        EXC = EXC.astype(np.float64)
        OMX = OMX.astype(np.float64)
        print("Rho X = ",min(ListRHO))
        print("La distance de Haussdorf X = ",min(ListHauss))
        print("Energie X = ",EXC)
        print("Homogénéité X = ",OMX)
        
        
        
        
    else:
        # Boucle pour l'energie, hauss, rho
        #for i in range(X_refO.shape[0]):
         #   ListEnergy.append(calcEnergy(X_refO[i]))
        
        for i in range(X_refO.shape[0]):
            ListHauss.append(calcHaussdorff(X_refO[i], imgX))
            
        for i in range(X_refO.shape[0]):
            ListRHO.append(calcRho(X_refO[i], imgX))
            
        #for i in range(X_refO.shape[0]):
         #   ListEnergyCo.append(Texture(X_refO[i]))     
        # Fin
        #------------------------------------------------------------
        
        #Calcul du rho pour X avec X_refO
        RX = min(ListRHO)/max(ListRHO)
        #Calcul du dist Rhoss pour X avec X_refO
        HX = min(ListHauss)/max(ListHauss)
        #Calcul du Energy pour X avec X_refO  
        #EX = calcEnergy(imgX)/max(ListEnergy)
        #Calcul du EnergyCo pour X avec X_refZ
        imgXX = imgX.numpy()
        EXC,OMX = Texture(imgXX)
        EXC = EXC.astype(np.float64)
        OMX = OMX.astype(np.float64)
        print("Rho X = ",min(ListRHO))
        print("La distance de Haussdorf X = ",min(ListHauss))
        print("Energie X = ",EXC)
        print("Homogénéité X = ",OMX)
     
    #--------------------------------------------------------------------------------  
    ## Energy max
    ListEnergy = []
 
    ## Hauss max
    ListHauss = []
    
    ## Max RHO
    ListRHO = [] 
    ## EnergyCo-occurence max
    ListEnergyCo = []
    #Calcul de RHE pour Y avec NORMALISATION:
    if(valY == 0):
        # Boucle pour l'energie, hauss, rho
        #for i in range(Y_refZ.shape[0]):
        #    ListEnergy.append(calcEnergy(Y_refZ[i]))
        
        for i in range(Y_refZ.shape[0]):
            ListHauss.append(calcHaussdorff(Y_refZ[i], imgY))
            
        for i in range(Y_refZ.shape[0]):
            ListRHO.append(calcRho(Y_refZ[i], imgY))
            
        #for i in range(Y_refZ.shape[0]):
         #   ListEnergyCo.append(Texture(Y_refZ[i]))     
        # Fin
        #------------------------------------------------------------
        #Calcul du rho pour Y avec Y_refZ
        RY = min(ListRHO)/max(ListRHO)
        #Calcul du dist Rhoss pour Y avec Y_refZ
        HY = min(ListHauss)/max(ListHauss)
        #Calcul du Energy pour Y avec Y_refZ
        #EY = calcEnergy(imgY)/max(ListEnergy)
        #Calcul du EnergyCo pour X avec X_refZ
        imgYY = imgY.numpy()
        EYC,OMY = Texture(imgYY)
        EYC = EYC.astype(np.float64) 
        OMY = OMY.astype(np.float64) 
        print("Rho Y = ",min(ListRHO))
        print("La distance de Haussdorf Y = ",min(ListHauss))
        print("Energie Y = ",EYC)
        print("Homogénéité Y = ",OMY)
        
    else:
        
        #for i in range(Y_refO.shape[0]):
         #  ListEnergy.append(calcEnergy(Y_refO[i]))
    
        for i in range(Y_refO.shape[0]):
            ListHauss.append(calcHaussdorff(Y_refO[i], imgY))
            
        for i in range(Y_refO.shape[0]):
            ListRHO.append(calcRho(Y_refO[i], imgY))
            
        #for i in range(Y_refO.shape[0]):
         #   ListEnergyCo.append(Texture(Y_refO[i])) 
        # Fin
        #------------------------------------------------------------
        #Calcul du rho pour Y avec Y_refO
        RY = min(ListRHO)/max(ListRHO)
        #Calcul du dist Rhoss pour Y avec Y_refO
        HY = min(ListHauss)/max(ListHauss)
        #Calcul du Energy pour Y avec Y_refO  
        #EY = calcEnergy(imgY)/max(ListEnergy)
        #Calcul du EnergyCo pour X avec X_refZ
        imgYY = imgY.numpy()
        EYC,OMY = Texture(imgYY)
        EYC = EYC.astype(np.float64) 
        OMY = OMY.astype(np.float64)
        print("Rho Y = ",min(ListRHO))
        print("La distance de Haussdorf Y = ",min(ListHauss))
        print("Energie Y = ",EYC)
        print("Homogénéité Y = ",OMY)
        
    #--------------------------------------------------------------------------------        
    #!Calcul de RHE pour Z avec NORMALISATION:
    ## Energy max
    ListEnergy = []
 
    ## Hauss max
    ListHauss = []
    
    ## Max RHO
    ListRHO = []
    ## EnergyCo-occurence max
    ListEnergyCo = []
    
    if(valZ == 0):
        #for i in range(Z_refZ.shape[0]):
         #  ListEnergy.append(calcEnergy(Z_refZ[i]))
    
        for i in range(Z_refZ.shape[0]):
            ListHauss.append(calcHaussdorff(Z_refZ[i], imgZ))
            
        for i in range(Z_refZ.shape[0]):
            ListRHO.append(calcRho(Z_refZ[i], imgZ))
            
        #for i in range(Z_refZ.shape[0]):
         #   ListEnergyCo.append(Texture(Z_refZ[i]))     
        # Fin
        #------------------------------------------------------------
        #Calcul du rho pour Z avec Z_refZ 
        RZ = min(ListRHO)/max(ListRHO)
        #Calcul du dist Rhoss pour Z avec Z_refZ
        HZ = min(ListHauss)/max(ListHauss)
        #Calcul du Energy pour Z avec Z_refZ
        #EZ = calcEnergy(imgZ)/max(ListEnergy) 
        #Calcul du EnergyCo pour X avec X_refZ
        imgZZ = imgZ.numpy()
        EZC,OMZ = Texture(imgZZ)
        EZC = EZC.astype(np.float64)
        OMZ = OMZ.astype(np.float64)
        print("Rho Z = ",min(ListRHO))
        print("La distance de Haussdorf Z = ",min(ListHauss))
        print("Energie Z = ",EZC)
        print("Homogénéité Z = ",OMZ)
    else:
    
        #for i in range(Z_refO.shape[0]):
         #  ListEnergy.append(calcEnergy(Z_refO[i]))
    
        for i in range(Z_refO.shape[0]):
            ListHauss.append(calcHaussdorff(Z_refO[i], imgZ))
            
        for i in range(Z_refO.shape[0]):
            ListRHO.append(calcRho(Z_refO[i], imgZ))
            
        #for i in range(Z_refO.shape[0]):
         #   ListEnergyCo.append(Texture(Z_refO[i]))     
        # Fin
        #------------------------------------------------------------
        #Calcul du rho pour Z avec Z_refO 
        RZ = min(ListRHO)/max(ListRHO)
        #Calcul du dist Rhoss pour Z avec Z_refO
        HZ = min(ListHauss)/max(ListHauss)
        #Calcul du Energy pour Z avec Z_refO
        #EZ = calcEnergy(imgZ)/max(ListEnergy) 
        #Calcul du EnergyCo pour X avec X_refZ
        imgZZ = imgZ.numpy()
        EZC,OMZ = Texture(imgZZ)
        EZC = EZC.astype(np.float64)
        OMZ = OMZ.astype(np.float64)
        print("Rho Z = ",min(ListRHO))
        print("La distance de Haussdorf Z = ",min(ListHauss))
        print("Energie Z = ",EZC)
        print("Homogénéité Z = ",OMZ)
        
    #--------------------------------------------------------------------------------
    BeninListe =[]
    MalinListe=[]
    
    if(valX == 0):
        BeninListe.append(RX)
        BeninListe.append(HX)
        #BeninListe.append(EX)
        BeninListe.append(EXC)
        BeninListe.append(OMX)
    else:
        MalinListe.append(RX)
        MalinListe.append(HX)
        #MalinListe.append(EX)
        MalinListe.append(EXC)
        MalinListe.append(OMX)
        
    if(valY == 0):
        BeninListe.append(RY)
        BeninListe.append(HY)
        #BeninListe.append(EY)
        BeninListe.append(EYC)
        BeninListe.append(OMY)
    else:
        MalinListe.append(RY)
        MalinListe.append(HY)
        #MalinListe.append(EY)
        MalinListe.append(EYC)  
        MalinListe.append(OMY)  
         
        
    if(valZ == 0):
        BeninListe.append(RZ)
        BeninListe.append(HZ)
        #BeninListe.append(EZ)
        BeninListe.append(EZC)
        BeninListe.append(OMZ)
    else:

        MalinListe.append(RZ)
        MalinListe.append(HZ)
       #MalinListe.append(EZ)
        MalinListe.append(EZC)
        MalinListe.append(OMZ)
    
    #return BeninListe, MalinListe
    #-------------------------------------------------------------------------------- 
    #Construction de la Matrice du jeu
    MatJeu = []
    vecBenin = np.array(BeninListe)
    vecMalin = np.array(MalinListe)

    for i in range(vecBenin.shape[0]):
        v1=[]
        for j in range(vecMalin.shape[0]):
            #Fonction d'utilité
            ut = vecBenin[i] - vecMalin[j]
            v1.append(ut)

        MatJeu.append(v1)

    #-------------------------------------------------------------------------------- 
    #Simulation du jeu avec nashpy
    MatJeu = np.array(MatJeu)
    #print("matjeu ",MatJeu)
    #print("somme ",np.sum(MatJeu))
    jeu = nash.Game(MatJeu)
    #print("jeu ",jeu)
    eqs = jeu.support_enumeration()
    a, g = next(eqs)
    ligne = np.argmax(a)
    col = np.argmax(g)
    valeur_nash = MatJeu[ligne, col]
    print("La valeur de nash = ",valeur_nash)
    if valeur_nash > 0:
        return 0
    elif valeur_nash < 0:
        return 1
    else:
        print("val nulle")
        # Majorite
        if (valX == valY) :
            
            return valZ[0]
        else:
            if (valY== valZ):
                
                return valX[0]
            
            else:
                if (valX== valZ):
                    
                    return valY[0]
                
def thj_pred(imgX, imgY, imgZ, modelx, modely, modelz):
    # Fonction de prediction avec utilisation du model de théorie des jeux
    imgX = Preprocessing(imgX)
    imgY = Preprocessing(imgY)
    imgZ = Preprocessing(imgZ)
    probaX, valX = predict_proba(modelx, tf.expand_dims(imgX, 0))
    probaY, valY = predict_proba(modely, tf.expand_dims(imgY, 0))
    probaZ, valZ = predict_proba(modelz, tf.expand_dims(imgZ, 0))
    
    if (valX[0] == valY[0]) and (valY[0]== valZ[0]):
        return valX[0]
    
    else:
        return thj_fonction(imgX, imgY, imgZ,valX[0],valY[0],valZ[0])                