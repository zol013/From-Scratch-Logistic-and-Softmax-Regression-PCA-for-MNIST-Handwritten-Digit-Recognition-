#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import sm_network
import sm_data as data
import image
import numpy as np
from sklearn.decomposition import PCA


# In[103]:


import matplotlib.pyplot as plt


# In[90]:


def softmax(hyperparemeters):
    network = sm_network.Network(hyperparameters, sm_network.softmax, sm_network.multiclass_cross_entropy, 10)
    val_acc = []
    val_loss = np.zeros((10,100))
    train_loss = np.zeros((10,100))
    fold_idx = 0
    for train_set, val_set in data.generate_k_fold_set((train_images,train_labels), 10):
        
        loss = np.Inf
         
        for epoch in range(100): #hyperparameters.epoch
            
            #train_set = data.shuffle(train_set)
            # stochastic gradient descent on every mini-batch
            tot_batch = 0
            for minibatch in data.generate_minibatches(train_set, 512):
                train_loss_value, train_acc = network.train(minibatch)
                train_loss[fold_idx][epoch] += train_loss_value
                tot_batch += 1
            # test performance on validation
            train_loss[fold_idx][epoch] /= tot_batch
            average_loss, acc = network.test(val_set)
            val_loss[fold_idx][epoch] = average_loss
            val_acc.append(acc)
            
            #if average_loss > loss:
            #    break
            loss = average_loss
        
        fold_idx += 1
    
    val_loss_avg = np.sum(val_loss, axis = 0) / 10
    train_loss = np.sum(train_loss, axis = 0) / 10
    for i in range(100):
        print(i,'th epoch average validation loss','   ',val_loss_avg[i])
    
    test_loss, test_acc = network.test((test_images, test_labels))
     
    return network.weights, train_loss, val_acc, val_loss, val_loss_avg, test_loss, test_acc


# In[ ]:





# In[112]:


train_images, train_labels = data.load_data('')
test_images, test_labels = data.load_data('', train = False)


# In[83]:


train_images, train_labels = data.shuffle((train_images, train_labels))


# In[84]:


#train_images, train_max, train_min = data.min_max_normalize(train_images)
#test_images, test_max, test_max = data.min_max_normalize(test_images)


train_images, train_mean, train_std = data.z_score_normalize(train_images)
test_images, test_mean, test_std = data.z_score_normalize(test_images)


# In[96]:


parser = argparse.ArgumentParser(description = 'CSE151B PA1')
parser.add_argument('--batch_size', type = int, default = 256,
        help = 'input batch size for training (default: 1)')
parser.add_argument('--epochs', type = int, default = 100,
        help = 'number of epochs to train (default: 100)')
parser.add_argument('--learningrate', type = float, default = 0.01,
        help = 'learning rate (default: 0.001)')
parser.add_argument('--zscore', dest = 'normalization', action='store_const', 
        default = data.min_max_normalize, const = data.z_score_normalize,
        help = 'use z-score normalization on the dataset, default is min-max normalization')
parser.add_argument('--folds', type = int, default = 10,
        help = 'number of folds for cross-validation')
parser.add_argument('--p', type = int, default = 70,
        help = 'number of principal components')

hyperparameters, unknown = parser.parse_known_args()


# In[7]:


hyperparameters.epochs


# In[113]:


pca = PCA(hyperparameters.p)
pca.fit(train_images)
train_images = pca.transform(train_images)
test_images = pca.transform(test_images)


# In[87]:


train_labels = data.onehot_encode(train_labels)
test_labels = data.onehot_encode(test_labels)


# In[88]:


train_images = data.append_bias(train_images)
test_images = data.append_bias(test_images)


# In[35]:


train_images.shape


# In[97]:


weights, train_loss, val_acc, val_loss, val_loss_avg, test_loss, test_acc = softmax(hyperparameters)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###PCA Weights visualization


# In[104]:


weights.shape


# In[114]:


weights


# In[115]:


weights_no_bias = np.delete(weights, 0, axis = 0)


# In[ ]:


pca_matrix = pca.components_


# In[119]:


weights_full_dim = pca_matrix.T @ weights_no_bias


# In[120]:


weights_full_dim.shape


# In[126]:


for i in range(10):
    weight_arr = weights_full_dim.T[i]
    print('digit_class:' + str(i + 1))
    plt.imshow(weight_arr.reshape(28,28))
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[118]:


pca_matrix.shape


# In[101]:


#batch_size = 512, learningrate = 0.01, PCA = 70, z-score normalization, TEST_LOSS =0.29427990497800804
#TEST_ACC = 0.9166


x = [i for i in range(100)]
plt.xlabel('# of epochs')
plt.ylabel('average validation loss per epoch')
plt.plot(x, val_loss_avg)
plt.grid(True)
plt.title('number of epochs vs validation loss during training')


# In[102]:


x = [i for i in range(100)]
plt.xlabel('# of epochs')
plt.ylabel('average training loss per epoch')
plt.plot(x, train_loss)
plt.grid(True)
plt.title('number of epochs vs training loss during training')


# In[ ]:





# In[98]:


val_acc


# In[93]:


train_loss[-100:]


# In[99]:


test_loss


# In[100]:


test_acc


# In[ ]:


BACTH = 256, LEARNING = 0.001, PCA = 100
VAL_LOSS = 1.0230239108555335
TEST_LOSS = 0.29742139263819406
TEST_ACC = 0.9158


# In[ ]:


BACTH = 256, LEARNING = 0.01, PCA = 100
VAL_LOSS = 0.30724605907271396
TEST_LOSS =0.28714344948988124
TEST_ACC = 0.9158


# In[ ]:


BACTH = 512, LEARNING = 0.01, PCA = 70
VAL_LOSS = 0.30718411118340444
val_acc = 0.9095
TEST_LOSS =0.29427990497800804
TEST_ACC = 0.9166


# In[ ]:


BACTH = 128, LEARNING = 0.001, PCA = 100
VAL_LOSS = 0.3040733906843014
TEST_LOSS = 0.2972136601130659

TEST_ACC = 0.915


# In[ ]:


BACTH = 256, LEARNING = 0.001, PCA = 70
VAL_LOSS = 0.10229202596667566
TEST_LOSS = 0.3051745952884528
TEST_ACC = 0.9158

