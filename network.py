import layers as l
from dataset import dataset
import numpy as np
import matplotlib.pyplot as plt
import logging

class LeNet():
    def __init__(self):
        # Layers
        self.conv1 = l.Convolution("conv1",3,6,5,1)
        self.conv2 = l.Convolution("conv2",6,16,5,1)
        self.relu = l.ReLU("relu")
        self.pool = l.Maxpooling("pooling",2,2)
        self.dense1 = l.Dense("dense1",16*5*5,120)
        self.dense2 = l.Dense("dense1",120,84)
        self.dense3 = l.Dense("dense1",84,10)
    def forward(self,x):
        

        # Feature extractor
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1)

        # Fully connected layers
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x

    def __call__(self,x):
        return self.forward(x)
    
    def load_params(self,dict_params):
        # Loading parameters

        logging.info('Loading net parameters')
        self.conv1.load_weight(dict_params['conv1_weight'])
        self.conv1.load_bias(dict_params['conv1_bias'])

        self.conv2.load_weight(dict_params['conv2_weight'])
        self.conv2.load_bias(dict_params['conv2_bias'])

        self.dense1.load_weight(dict_params['dense1_weight'])
        self.dense1.load_bias(dict_params['dense1_bias'])

        self.dense2.load_weight(dict_params['dense2_weight'])
        self.dense2.load_bias(dict_params['dense2_bias'])

        self.dense3.load_weight(dict_params['dense3_weight'])
        self.dense3.load_bias(dict_params['dense3_bias'])

    def show(self,name,x,col=False,line=False,block=False):
        c,_,_ = x.shape
        if col == False: col = 2 
        if line == False: line = c//col 
        assert col*line >= c, logging.critical('Could not display with this configuration col = {} and line = {}'.format(col,line))
        plt.figure()
        plt.suptitle(name)
        for i in range(c):
            sub = plt.subplot(line,col,i+1)
            sub.set_title(i)
            sub.axis('off')
            plt.imshow(x[i])
        plt.savefig(name)
        plt.show(block=block)