import numpy as np
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
import csv

def make_fixed_embeddings(glove, seq_len):
    glove_mat = np.array(glove.values())
    return Embedding(input_dim = glove_mat.shape[0], output_dim = glove_mat.shape[1], 
                       weights = [glove_mat], trainable = False, input_length  = seq_len)
                       
   
class CsvHistory(Callback):
    
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.writer = csv.writer(self.file)
        self.header = True
 
    def on_epoch_end(self, epoch, logs={}):
        if self.header:
            self.writer.writerow(logs.keys())
            self.header = False
        self.writer.writerow(logs.values())

    def on_train_end(self, logs={}):
        self.file.close()  
