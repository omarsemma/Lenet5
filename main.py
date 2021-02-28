from dataset import dataset
from network import LeNet
import numpy as np
import logging
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='Debug option',action='store_true')
parser.add_argument('--nb_imgs', help='Number of picture to use in db < 10000',type=int,default=10000)
args = parser.parse_args()

if args.debug:
    logging.basicConfig(filename='debug.log',level=logging.INFO,format='%(levelname)s %(asctime)s %(message)s',filemode='w')

# dataset object creation
data = dataset(nb_imgs=args.nb_imgs)

net = LeNet()

# Dictionnary for network parameters
params = {
    "conv1_weight" : np.load('params/conv1.weight.save'),
    "conv1_bias" : np.load('params/conv1.bias.save'),

    "conv2_weight" : np.load('params/conv2.weight.save'),
    "conv2_bias" : np.load('params/conv2.bias.save'),

    "dense1_weight" : np.load('params/fc1.weight.save'),
    "dense1_bias" : np.load('params/fc1.bias.save'),

    "dense2_weight" : np.load('params/fc2.weight.save'),
    "dense2_bias" : np.load('params/fc2.bias.save'),

    "dense3_weight" : np.load('params/fc3.weight.save'),
    "dense3_bias" : np.load('params/fc3.bias.save')
}

# Loading params
net.load_params(params)

positive = 0
# Testing
for i,(img,label) in tqdm(enumerate(data),initial=0,total=len(data),desc="Classification of CIFAR-10",ascii=True,unit='it', unit_scale=True):
    out = net( ( (img/255) - 0.5) / 0.5 )
    idx_max = np.argmax(out)
    positive += idx_max == label
    logging.info('CURRENT POSITIVE : {} %'.format( positive*100/(i+1)) )
    if idx_max == label:
        logging.info("Classe predite : {}".format(data.classes[idx_max]))
    else:
        logging.info("Classe predite : {} insted of {}".format(data.classes[idx_max],data.classes[label]))