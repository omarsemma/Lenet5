import requests as req
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from PIL import Image 
import argparse
import logging 






class dataset:
    
    def __init__(self,file_name,root = os.getcwd()):
       self.file_name = file_name
       self.root = root
       self.__labels = []
       self.__labels_name = []
       self.__images = []
       self.database_url = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
       self.download_base()
       self.__labeling() 
        
    def __len__(self):
        return len(self.__images)
    def __getitem__(self,idx):
        return self.__images[idx],self.__labels[idx]
    
    def show(self,idx):
        # Plots image (RGB + channels) and its label
        
        # We can also use numpy.moveaxis()
        # Arrays of different images
        image_red_channel_array = self.__images[idx][0]
        image_green_channel_array = self.__images[idx][1]
        image_blue_channel_array = self.__images[idx][2]
        image_rgb = np.dstack((image_red_channel_array,image_green_channel_array,image_blue_channel_array))
        
        image_red_channel = Image.fromarray(image_red_channel_array)
        image_green_channel = Image.fromarray(image_green_channel_array)
        image_blue_channel = Image.fromarray(image_blue_channel_array)
        image_rgb_mode = Image.fromarray(image_rgb)
        
        # plotting
        fig, axs = plt.subplots(2, 2)
        
        fig.suptitle('Label : {}'.format(self.__labels_name[self.__labels[idx]]))
        axs[0, 0].imshow(image_rgb_mode)
        axs[0, 0].set_title('RGB Mode')
        axs[0, 1].imshow(image_red_channel,cmap='gray')
        axs[0, 1].set_title('Red Channel')
        axs[1, 0].imshow(image_green_channel,cmap='gray')
        axs[1, 0].set_title('Green Channel')
        axs[1,1].imshow(image_blue_channel,cmap='gray')
        axs[1, 1].set_title('Blue Channel')
        for axes in axs :
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()
    
    def __labeling(self):
        # create label_names 
        with open('{}/binary_cifar/batches.meta.txt'.format(self.root),'r') as file:
            for line in file:
                if line.rstrip():
                    self.__labels_name.append(line.rstrip('\n'))

    def read_dataset(self):
        with open('{}/binary_cifar/{}'.format(self.root,self.file_name),"rb") as file :
            for i in range(10000):
                byte = file.read(1)
                self.__labels.append(int.from_bytes(byte,byteorder='big')) # byte read the label
                byte_array = file.read(3072)
                image = [byte for byte in byte_array] # convert and store image data from array byte 
                self.__images.append(np.array(image,'uint8').reshape(3,32,32)) 
    
    def download_base(self):
        # download the database if it dosen't exists
        request = req.get(self.database_url,stream=True)
        content_length = int(request.headers.get('content-length'))
        tar_gz = 'cifar-10.tar.gz'
        
        if not os.path.isfile(tar_gz):
            with open(tar_gz,'wb') as file:
                with tqdm(total=content_length, unit='it', unit_scale=True,desc=tar_gz,initial=0,ascii=True) as pbar: # progress bar
                    for chunk in request.iter_content(chunk_size=1024):
                        if chunk :    
                            file.write(chunk)
                            pbar.update(len(chunk)) 

                
if __name__ == '__main__':
    # to test data 
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx','--index',type=int, required = False ,help = 'Index of the image you want to display')
    parser.add_argument('--db','--database_file',type=str,help = 'Name of the database binary file')
    args = parser.parse_args()

    # Error Logging (I used logger.exception to keep the traceback...You can change that with error)
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler('dataset_errors.log',mode='w')
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(name)s:%(message)s'))
    logger.addHandler(handler)
    try:
        db = dataset(args.db)
        db.read_dataset()
        # db.show(args.idx)
    except (IndexError,FileNotFoundError,Exception) as e:
        logger.exception(e)
    
        
    

   
    
   


        
