#!/usr/bin/env python
# coding: utf-8

# In[8]:


import idx2numpy
import numpy as np
import os
import pickle
from PIL import Image
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def load_data(data_directory, train = True):
    if train:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'train-images.idx3-ubyte'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'train-labels.idx1-ubyte'))
    else:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 't10k-images.idx3-ubyte'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 't10k-labels.idx1-ubyte'))

    vdim = images.shape[1] * images.shape[2]
    vectors = np.empty([images.shape[0], vdim])
    for imnum in range(images.shape[0]):
        imvec = images[imnum, :, :].reshape(vdim, 1).squeeze()
        vectors[imnum, :] = imvec
    
    return vectors, labels


# In[2]:


train_images, train_labels = load_data('')


# In[3]:


pca = PCA(70)
pca.fit(train_images)


# In[4]:


pca_matrix = pca.components_


# In[6]:


pca_matrix.shape


# In[14]:


for i in range(10):
    pc = pca_matrix[i]
    print(str(i) +'th top principle component')
    plt.imshow(pc.reshape(28,28), cmap='gray')
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





