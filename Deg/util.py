import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


def get_transform(resize_or_crop,loadSizeX,loadSizeY,fineSize):
    transform_list = []
    if resize_or_crop == 'resize_and_crop':
        osize = [loadSizeX, loadSizeY]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop =='scale':
        osize = [loadSizeX,loadSizeY]

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

transform = get_transform('scale',loadSizeX =512, loadSizeY=512, fineSize=512)

def imread(file_path,c=None):
    if c is None:
        im=cv2.imread(file_path)
    else:
        im=cv2.imread(file_path,c)
    
    if im is None:
        raise 'Can not read image'

    if im.ndim==3 and im.shape[2]==3:
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im

def imwrite(file_path,image):
    if image.ndim==3 and image.shape[2]==3:
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path,image)   

def remove_back_area(img,bbox=None,border=None):
    image=img
    if border is None:
        border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=np.int)
    image=image[border[0]:border[1],border[2]:border[3],...]
    return image,border

def get_mask_BZ(img):
    if img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-7
    _, mask = cv2.threshold(gray_img, max(0,threhold), 1, cv2.THRESH_BINARY)
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask

def _get_center_radius_by_hough(mask):
    circles= cv2.HoughCircles((mask*255).astype(np.uint8),cv2.HOUGH_GRADIENT,1,1000,param1=5,param2=5,minRadius=min(mask.shape)//4, maxRadius=max(mask.shape)//2+1)
    center = circles[0,0,:2]
    radius = circles[0,0,2]
    return center,radius

def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    return center_mask    

def get_mask(img):
    if img.ndim ==3:
        g_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        g_img =img.copy()
    else:
        raise 'image dim is not 1 or 3'
    h,w = g_img.shape
    shape=g_img.shape[0:2]
    g_img = cv2.resize(g_img,(0,0),fx = 0.5,fy = 0.5)
    tg_img=cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask=get_mask_BZ(tg_img)
    center, radius = _get_center_radius_by_hough(tmp_mask)
    #resize back
    center = [center[1]*2,center[0]*2]
    radius = int(radius*2)
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    tmp_mask=_get_circle_by_center_bbox(shape,center,bbox,radius)
    return tmp_mask,bbox,center,radius

def mask_image(img,mask):
    img[mask<=0,...]=0
    return img

def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)    
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border    

def preprocess(img):
    # preprocess images 
    #   img : origin image
    # return:
    #   result_img: preprocessed image 
    #   mask: mask for preprocessed image
    mask,bbox,center,radius=get_mask(img)
    r_img=mask_image(img,mask)
    r_img,r_border=remove_back_area(r_img,bbox=bbox)
    mask,_=remove_back_area(mask,border=r_border)
    r_img,sup_border=supplemental_black_area(r_img)
    mask,_=supplemental_black_area(mask,border=sup_border)
    return r_img,(mask*255).astype(np.uint8)