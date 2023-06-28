import os 
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import random

class EyeQ_Dataset(Dataset):
    def __init__(self, rootHQ,rootPQ, file_dir, select_number, transform_HQ=None, transform_PQ = None):
        
        dataset = pd.read_csv(file_dir)

        #image_id = dataset.iloc[:,0].values

        image_dir = dataset.iloc[:,1].values

        image_quality = dataset.iloc[:,2].values
        
        image_Grading_label = dataset.iloc[:,3].values

        print(random.randint(99999))
        split = StratifiedShuffleSplit(n_splits=1,test_size=0.5,random_state=random.randint(99999))

        for a, b in split.split(image_dir,image_Grading_label):

            all_HQ_images = image_dir[a]
            all_HQ_labels = image_Grading_label[a]
            all_PQ_images = image_dir[b]
            all_PQ_labels = image_Grading_label[b]

        # select number of image    0->Hight Quality    1->usable   2->Poor quality
        #index_HQ =  np.where(image_quality == 0)[0]
        #random_HQ_select = np.random.randint(low=0,high=int(len(all_HQ_images) - 1),size=select_number)
        #index_HQ_select = np.take(all_HQ_images,random_HQ_select)

        # Hight Quality 
        self.HQ_images =  all_HQ_images
        self.HQ_DRgrading_labels = all_HQ_labels


        #index_PQ = np.where(image_quality == 2)[0]
        random_PQ_select = np.random.randint(low=0,high=int(len(all_PQ_images) - 1),size=select_number)
        #index_PQ_select = np.take(all_PQ_images,random_PQ_select)


        # Poor Quality 
        self.PQ_images = np.take(image_dir,random_PQ_select)
        self.PQ_DRgrading_labels = np.take(all_PQ_labels,random_PQ_select)
        #labels = dataset.iloc[:, 1].values
        #image_ids = dataset.iloc[:, 0].values
        self.transform_HQ = transform_HQ
        self.transform_PQ = transform_PQ
        self.rootHQ = rootHQ
        self.rootPQ = rootPQ
        self.select_number = select_number

    def __len__(self):
        return self.select_number


    def __getitem__(self, idx):

        # High Quality Image 
        # Generate the corresponding index of the label with hight quality image
        dr_label = self.PQ_DRgrading_labels[idx]
        #print(dr_label)
        all_HQ_index_with_same_label = np.where(self.HQ_DRgrading_labels == dr_label)[0]
        #print(all_HQ_index_with_same_label)
        random_HQ_index = np.random.randint(low=0,high=int(len(all_HQ_index_with_same_label) - 1),size=1)[0]
        generate_HQ_index = all_HQ_index_with_same_label[random_HQ_index]

        #print(random_HQ_index[0])
        hq_dr_label = self.HQ_DRgrading_labels[generate_HQ_index]

        HQ_file = os.path.splitext(self.HQ_images[generate_HQ_index])[0] + '.jpeg'
        #print(hq_path)
    
        HQ_path = os.path.join(self.rootHQ,HQ_file)
        
        HQ_image = Image.open(HQ_path)

        #HQ_image = cv2.imread(HQ_path)
        #HQ_image = cv2.cvtColor(HQ_image, cv2.COLOR_BGR2RGB)
        
        if self.transform_HQ is not None:
            # transform_HQ = self.transform_HQ(image=HQ_image)
            # hq = transform_HQ["image"]
            hq = self.transform_HQ(HQ_image)

        #muti_label = load_image_label_from_xml(label)

            # mask = transformed["mask"]
        PQ_file = os.path.splitext(self.PQ_images[idx])[0] + '_111.jpeg'
        PQ_path = os.path.join(self.rootPQ,PQ_file)
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
        return pq,hq,dr_label,hq_dr_label