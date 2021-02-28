import numpy as np
import logging

class Convolution():
    def __init__(self,name,in_channels,nb_filtre,kernel_shape,stride):
        logging.info("CREATE CONV LAYER : {} : WITH IN {} CHANNELS, {} FILTERS, STRIDE {}, KERNEL SHAPE ({},{})".format(name,in_channels,nb_filtre,stride,kernel_shape,kernel_shape))
        self.name = name
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.nb_filtre = nb_filtre
        self.in_channels = in_channels
        self.weight = np.random.rand(nb_filtre,in_channels,kernel_shape,kernel_shape)
        self.bias   = np.random.rand(nb_filtre) 
    def __call__(self,x):
        in_c, in_h, in_w = x.shape
        out_h  = int( (in_h - (self.kernel_shape -1) - 1 ) / self.stride) + 1
        out_w  = int( (in_w - (self.kernel_shape -1) - 1 ) / self.stride) + 1
        out = []
        for f in range(self.nb_filtre):
            for h in range(0,out_h,self.stride):
                for w in range(0,out_w,self.stride):
                    acc = 0
                    for channel in range(self.in_channels):
                        for m in range(self.kernel_shape):
                            for n in range(self.kernel_shape):
                                acc += x[channel][h+m][w+n] * self.weight[f][channel][m][n]
                    acc += self.bias[f]
                    out.append(acc)
        return np.array(out).reshape(self.nb_filtre,out_h,out_w)

    def load_weight(self,new_weight):
        logging.info('loading weight with shape : {}'.format(new_weight.shape))
        assert self.weight.shape == new_weight.shape, logging.critical('Wrong dimension {} != {}'.format(self.weight.shape,new_weight.shape))
        self.weight = new_weight
    def load_bias(self,new_bias):
        logging.info('loading bias with shape : {}'.format(new_bias.shape))
        assert self.bias.shape == new_bias.shape, logging.critical('Wrong dimension {} != {}'.format(self.bias.shape,new_bias.shape))
        self.bias = new_bias
    def load_params(self,dict_params):
        self.load_weight(dict_params['weight'])
        self.load_bias(dict_params['bias'])

class Dense():
    def __init__(self,name,dim_in,dim_out):
        logging.info('CREATE DENSE LAYER : {} :  {} TO {}'.format(name,dim_in,dim_out))

        self.name = name
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.weight = np.random.rand(dim_out,dim_in)
        self.bias   = np.random.rand(dim_out)

    def __call__(self,x):
        out = np.dot(x,np.transpose(self.weight))
        out += self.bias
        return out

    def load_weight(self,new_weight):
        logging.info('loading weight with shape : {}'.format(new_weight.shape))
        assert self.weight.shape == new_weight.shape, log.critical('Wrong dimension {} != {}'.format(self.weight.shape,new_weight.shape))
        self.weight = new_weight
    def load_bias(self,new_bias):
        logging.info('loading bias with shape : {}'.format(new_bias.shape))
        assert self.bias.shape == new_bias.shape, log.critical('Wrong dimension {} != {}'.format(self.bias.shape,new_bias.shape))
        self.bias = new_bias
        
    def load_params(self,dict_params):
        self.load_weight(dict_params['weight'])
        self.load_bias(dict_params['bias'])


class ReLU():
    def __init__(self,name):
        logging.info('CREATE ReLU LAYER : {}'.format(name))
        self.name = name
    def __call__(self,x):
        x[x<0] = 0 
        return x


class Maxpooling():
    def __init__(self,name,kernel_shape,stride):
        logging.info('CREATE MaxPooling LAYER : {} WITH STRIDE {} AND KERNEL SHAPE ({},{})'.format(name,stride,kernel_shape,kernel_shape))
        self.name = name 
        self.kernel_shape = kernel_shape
        self.stride = stride
    def __call__(self,x):
        channels, height,width = x.shape
        out = []
        out_shape_x = int((width - self.kernel_shape ) / self.stride)+1
        out_shape_y = int((height - self.kernel_shape) / self.stride)+1
        for channel in range(channels):
            for h in range(0,height-self.kernel_shape+1,self.stride):
                for w in range(0,width-self.kernel_shape+1,self.stride):
                    maximum = x[channel][h][w]
                    for m in range(self.kernel_shape):
                        for n in range(self.kernel_shape):
                            if x[channel][h+m][w+n] > maximum : maximum = x[channel][h+m][w+n]
                    out.append(maximum)
        out = np.array(out).reshape((channels,out_shape_y,out_shape_x))
        return out