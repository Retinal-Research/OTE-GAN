import os 
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class EyeQ_Dataset(Dataset):
    def __init__(self,mode,transform_HQ=None, transform_PQ = None):
        
        self.image_list = glob.glob("dataset/preprocessed/referencetest/pre/*.*")
        self.original_root = 'dataset/preprocessed/referencetest/pre/'
        self.degregation_root = 'dataset/preprocessed/referencetest/deg/'
        
        self.transform_HQ = transform_HQ
        self.transform_PQ = transform_PQ
        self.mode = mode

    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):

        original_path = self.image_list[idx]

        image_dir = os.path.split(original_path)[-1]
        image_root = os.path.splitext(image_dir)[0] + '_'+self.mode+'.jpeg'
        #print(image_name)

        degregation_path = self.degregation_root + image_root



        
        original_image = Image.open(original_path)

        
        if self.transform_HQ is not None:
            ori = self.transform_HQ(original_image)


        deg_image = Image.open(degregation_path)


        if self.transform_PQ is not None:

            deg = self.transform_PQ(deg_image)



        return deg,ori,image_dir