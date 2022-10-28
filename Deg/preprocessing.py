from util import *
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.pool import Pool
import cv2 as cv
import pandas as pd
import csv

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dsize = (512,512)

def process(image_list):  
    for image_path in image_list: 
        
        name = os.path.splitext(image_path)[0] + '.jpeg'
        dst_image_path = os.path.join('dataset/degratation/pre', name)
        dst_mask_path = os.path.join('dataset/degratation/mask', name)
        try:
            img = imread('dataset/original/test/'+name)
            img, mask = preprocess(img)
            img = cv.resize(img, dsize)
            mask = cv.resize(mask, dsize)
            imwrite(dst_image_path, img)
            imwrite(dst_mask_path, mask)
        except:
            print(name)
            continue

if __name__=="__main__":
    
    #image_list = glob.glob(os.path.join('dataset/degratationpreprocessed/referencetest/ori', '*.jpeg'))
    data = pd.read_csv('dataset/all_data.csv')
    image_list = np.array(data['image'])
    #print(len(all_images))

    patches = 16
    patch_len = int(len(image_list)/patches)
    filesPatchList = []
    for i in range(patches-1):
        fileList = image_list[i*patch_len:(i+1)*patch_len]
        filesPatchList.append(fileList)
    filesPatchList.append(image_list[(patches-1)*patch_len:])


    pool = Pool(patches)
    pool.map(process, filesPatchList)
    pool.close()