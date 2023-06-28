import os 
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class EyeQ_Dataset(Dataset):
    def __init__(self, root, file_dir, select_number, transform_HQ=None, transform_PQ = None):
        
        dataset = pd.read_csv(file_dir)

        #image_id = dataset.iloc[:,0].values

        image_dir = dataset.iloc[:,1].values

        image_quality = dataset.iloc[:,2].values
        
        #image_Grading_label = dataset.iloc[:,3].values

        # select number of image    0->Hight Quality    1->usable   2->Poor quality
        index_HQ =  np.where(image_quality == 0)[0]
        random_HQ_select = np.random.randint(low=0,high=int(len(index_HQ) - 1),size=select_number)
        index_HQ_select = np.take(index_HQ,random_HQ_select)

        # Hight Quality 
        self.HQ_images = np.take(image_dir,index_HQ_select)


        index_PQ = np.where(image_quality == 2)[0]
        random_PQ_select = np.random.randint(low=0,high=int(len(index_PQ) - 1),size=select_number)
        index_PQ_select = np.take(index_PQ,random_PQ_select)


        # Poor Quality 
        self.PQ_images = np.take(image_dir,index_PQ_select)

        #labels = dataset.iloc[:, 1].values
        #image_ids = dataset.iloc[:, 0].values
        self.transform_HQ = transform_HQ
        self.transform_PQ = transform_PQ
        self.root = root
        self.select_number = select_number

    def __len__(self):
        return self.select_number


    def __getitem__(self, idx):
        # image_name = self.image_ids[idx] + '.tiff'
        # imagepath = os.path.join(self.image_dir,image_name)

        # image = cv2.imread(imagepath)

        # ori_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # label = self.labels[idx]

        # High Quality Image 
        HQ_file = os.path.splitext(self.HQ_images[idx])[0] + '.png'
        #print(hq_path)
        HQ_path = os.path.join(self.root,HQ_file)
        
        HQ_image = Image.open(HQ_path)

        #HQ_image = cv2.imread(HQ_path)
        #HQ_image = cv2.cvtColor(HQ_image, cv2.COLOR_BGR2RGB)
        
        if self.transform_HQ is not None:
            # transform_HQ = self.transform_HQ(image=HQ_image)
            # hq = transform_HQ["image"]
            hq = self.transform_HQ(HQ_image)

        #muti_label = load_image_label_from_xml(label)

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
        return pq,hq