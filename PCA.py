#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


# In[11]:


def export_image(img_arr, name='test.tiff'):
    Image.fromarray(img_arr).save(name)


# In[20]:


train_images, train_labels = load_data('')


# In[23]:


idx = random.randint(0, len(train_images) - 1)


# In[24]:


image = train_images[idx]
label = train_labels[idx]


# In[25]:


label


# In[26]:


export_image(image.reshape(28,28), name = 'PCA_original_image.tiff')


# In[ ]:





# In[27]:


pca_dimensions = [2, 10, 50, 100, 200, 784]


# In[32]:


for p in pca_dimensions:
    pca = PCA(p)
    pca.fit(train_images)
    projected_train_images = pca.transform(train_images)
    projected_image = projected_train_images[idx]
    print('projected_dimension:', projected_image.shape)
    recon_train = pca.inverse_transform(projected_train_images)
    recon_image = recon_train[idx]
    print('reconstructed_dimension:', recon_image.shape)
    plt.imshow(recon_image.reshape(28,28), cmap='gray', vmin=0, vmax=255)
    plt.show()



