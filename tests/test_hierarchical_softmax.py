
from hierarchical_softmax import HierarchicalSoftmax, hs_categorical_crossentropy
from keras.utils.generic_utils import Progbar
from keras.models import Sequential, Graph
from keras.layers import Activation
from keras.layers.core import TimeDistributedDense

import numpy as np
from numpy.linalg import norm

def test_hierarchical_softmax(timesteps = 15, input_dim = 50, batch_size = 32,
                              output_dim = 3218, batches = 300, epochs = 30):
    
    model = Graph()
    model.add_input(name='real_input', batch_input_shape=(batch_size, timesteps, input_dim))
    model.add_input(name='train_input', batch_input_shape=(batch_size, timesteps), dtype='int32')
    model.add_node(HierarchicalSoftmax(output_dim, input_dim = input_dim, input_length = timesteps), 
                   name = 'hs', inputs=['real_input','train_input'], 
                   merge_mode = 'join', create_output=True)
    
    
    model.compile(loss={'hs':hs_categorical_crossentropy}, optimizer='adam')
    print("hs model compiled")     
    
    model2 = Sequential()
    model2.add(TimeDistributedDense(output_dim, 
                    batch_input_shape=(batch_size, timesteps, input_dim)))
    model2.add(Activation('softmax'))    
    model2.compile(loss='categorical_crossentropy', optimizer='adam')
    print("softmax model compiled") 
    
    learn_f = np.random.normal(size = (input_dim, output_dim))
    learn_f = np.divide(learn_f, norm(learn_f, axis=1)[:,None])
    print("learn_f generated")
    
    
    for j in range(epochs):    
      
      
        batch_data= generate_batch(learn_f, batch_size, 
                                   timesteps, input_dim, output_dim, batches)
            
        print("Epoch", j, "data genrated")
         
        p = Progbar(batches * batch_size)
        for b in batch_data:
            data_train = {'real_input': b[0], 'train_input': b[1], 'hs':b[2]}
            loss =  float(model.train_on_batch(data_train)[0])
            p.add(batch_size,[('hs_loss', loss)])
        p2 = Progbar(batches * batch_size)
        for b in batch_data:
            loss, acc  =  model2.train_on_batch(b[0], b[3], accuracy=True)
            p2.add(batch_size,[('softmax_loss', loss),('softmax_acc', acc)])
           
    
    test_data = generate_batch(learn_f, batch_size, 
                                   timesteps, input_dim, output_dim, batches)
                                   
    p = Progbar(batches * batch_size)
    for b in test_data:
        data_test = {'real_input': b[0], 'train_input': b[1], 'hs':b[3]}
        loss =  float(model.test_on_batch(data_test)[0])
        p.add(batch_size,[('hs__test_loss', loss)])
        
    p2 = Progbar(batches * batch_size)
    for b in batch_data:
        loss =  float(model2.train_on_batch(b[0], b[3])[0])
        p2.add(batch_size,[('softmax_loss', loss)])
        
    
        
def generate_batch(learn_f, batch_size, timesteps, input_dim, output_dim, batches):    
    batch_data = []        
    for i in range(batches):
        ri = np.random.normal(size = (batch_size, timesteps, input_dim))
        ri = np.divide(ri, norm(ri, axis=2)[:,:,None])
        ti = np.argmax(np.dot(ri, learn_f), axis = 2)
       
        ones = np.ones_like(ti, dtype = 'float32')[:,:,None]
        
        one_hot = np.zeros((batch_size, timesteps, output_dim))
        one_hot[np.arange(batch_size)[:,np.newaxis],np.arange(timesteps),ti] = 1.0
        batch_data.append((ri,ti,ones,one_hot))
    return batch_data
