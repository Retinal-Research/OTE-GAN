import os 
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class EyeQ_Dataset(Dataset):
    def __init__(self, root, file_dir, transform_HQ=None, transform_PQ = None):
        
        dataset = pd.read_csv(file_dir)

        #image_id = dataset.iloc[:,0].values

        image_dir = dataset.iloc[:,1].values

        image_quality = dataset.iloc[:,2].values
        
        image_Grading_label = dataset.iloc[:,3].values


        # select number of image    0->Hight Quality    1->usable   2->Poor quality
        #index_HQ =  np.where(image_quality == 0)[0]
        #random_HQ_select = np.random.randint(low=0,high=int(len(index_HQ) - 1),size=select_number)
        #index_HQ_select = np.take(index_HQ,random_HQ_select)

        # Hight Quality 
        #self.HQ_images = np.take(image_dir,index_HQ)
        #self.HQ_DRgrading_labels = np.take(image_Grading_label,index_HQ)


        index_PQ = np.where(image_quality == 2)[0]
        #random_PQ_select = np.random.randint(low=0,high=int(len(index_PQ) - 1),size=select_number)
        #index_PQ_select = np.take(index_PQ,index_PQ)
        self.len = len(index_PQ)

        # Poor Quality 
        self.PQ_images = np.take(image_dir,index_PQ)
        self.PQ_DRgrading_labels = np.take(image_Grading_label,index_PQ)
        #labels = dataset.iloc[:, 1].values
        #image_ids = dataset.iloc[:, 0].values
        self.transform_PQ = transform_PQ
        self.root = root

    def __len__(self):
        return self.len


    def __getitem__(self, idx):

        # High Quality Image 
        # Generate the corresponding index of the label with hight quality image
        dr_label = self.PQ_DRgrading_labels[idx]
        #print(dr_label)
        #name = os.path.splitext(self.PQ_images[idx])[0]
            # mask = transformed["mask"]
        PQ_file = os.path.splitext(self.PQ_images[idx])[0] + '.png'
        PQ_path = os.path.join(self.root,PQ_file)
        PQ_image = Image.open(PQ_path)
        #PQ_image = cv2.imread(PQ_path)
        #PQ_image = cv2.cvtColor(PQ_image, cv2.COLOR_BGR2RGB)

        if self.transform_PQ is not None:
            # a 
            # transform_PQ = self.transform_PQ(image=PQ_image)
            # qp = transform_PQ["image"]
            pq = self.transform_PQ(PQ_image)


        # if self.augmentation is not None: 
        #     augmentation = self.augmentation(image=ori_image)
        #     aug = augmentation["image"]

        #     return image, aug,label
        # else:
        return pq,dr_label,PQ_file