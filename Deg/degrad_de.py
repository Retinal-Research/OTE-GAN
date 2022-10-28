import random
import math
import torchvision.transforms.functional as F
import numpy as np
import cv2
import json
from PIL import Image
from util import transform

def DE_COLOR(img, brightness=0.3, contrast=0.4, saturation=0.4):
    """Randomly change the brightness, contrast and saturation of an image"""
    
    if brightness > 0:
        brightness_factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness-0.1) # brightness factor
        img = F.adjust_brightness(img, brightness_factor)
    if contrast > 0:
        contrast_factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast) # contrast factor
        img = F.adjust_contrast(img, contrast_factor)
    if saturation > 0:
        saturation_factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation) # saturation factor
        img = F.adjust_saturation(img, saturation_factor)

    img = transform(img)
    img = img.numpy()

    color_params = {}
    color_params['brightness_factor'] = brightness_factor
    color_params['contrast_factor'] = contrast_factor
    color_params['saturation_factor'] = saturation_factor

    return img, color_params

def DE_HALO(img, h, w, brightness_factor, center=None, radius=None):
    '''
    Defined to simulate a 'ringlike' halo noise in fundus image
    :param weight_r/weight_g/weight_b: Designed to simulate 3 kinds of halo noise shown in Kaggle dataset.
    :param center_a/center_b:          Position of each circle which is simulated the ringlike shape
    :param dia_a/dia_b:                Size of each circle which is simulated the ringlike noise
    :param weight_hal0:                Weight of added halo noise color
    :param sigma:                      Filter size for final Gaussian filter
    '''
    
    weight_r = [251/255,141/255,177/255]
    weight_g = [249/255,238/255,195/255]
    weight_b = [246/255,238/255,147/255]
    # num
    if brightness_factor >= 0.2:
        num = random.randint(1, 2)
    else:
        num = random.randint(0, 2)
    w0_a = random.randint(w/2-int(w/8),w/2+int(w/8))
    h0_a = random.randint(h/2-int(h/8),h/2+int(h/8))
    center_a = [w0_a, h0_a]

    wei_dia_a =0.75 + (1.0-0.75) * random.random()
    dia_a = min(h,w)*wei_dia_a
    Y_a, X_a = np.ogrid[:h, :w]
    dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
    circle_a = dist_from_center_a <= (int(dia_a / 2))

    mask_a = np.zeros((h, w))
    mask_a[circle_a] = np.mean(img) #np.multiply(A[0], (1 - t))

    center_b =center_a
    Y_b, X_b = np.ogrid[:h, :w]
    dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)

    dia_b_max =2* int(np.sqrt(max(center_a[0],h-center_a[0])*max(center_a[0],h-center_a[0])+max(center_a[1],h-center_a[1])*max(center_a[1],w-center_a[1])))/min(w,h)
    wei_dia_b = 1.0+(dia_b_max-1.0) * random.random()

    if num ==0:
        # if halo tend to be a white one, set the circle with a larger radius.
        dia_b = min(h, w) * wei_dia_b + abs(max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h)*2 / 3)
    else:
        dia_b =min(h,w)* wei_dia_b +abs(max(center_b[0]-w/2,center_b[1]-h/2)+max(w,h)/2)

    circle_b = dist_from_center_b <= (int(dia_b / 2))

    mask_b = np.zeros((h, w))
    mask_b[circle_b] = np.mean(img)

    weight_hal0 = [0, 1, 1.5, 2, 2.5]
    delta_circle = np.abs(mask_a - mask_b) * weight_hal0[1]
    dia = max(center_a[0],h-center_a[0],center_a[1],h-center_a[1])*2
    gauss_rad = int(np.abs(dia-dia_a))
    sigma = 2/3*gauss_rad

    if(gauss_rad % 2) == 0:
        gauss_rad= gauss_rad+1
    delta_circle = cv2.GaussianBlur(delta_circle, (gauss_rad, gauss_rad), sigma)

    delta_circle = np.array([weight_r[num]*delta_circle,weight_g[num]*delta_circle,weight_b[num]*delta_circle])
    img = img + delta_circle

    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    halo_params = {}
    halo_params['num'] = num
    halo_params['center_a'] = center_a
    halo_params['dia_a'] = dia_a
    halo_params['center_b'] = center_b
    halo_params['dia_b'] = dia_b

    return img, halo_params

def DE_HOLE(img, h, w, region_mask, center=None, diameter=None):
    '''
    :param diameter_circle:     The size of the simulated artifacts caused by non-uniform lighting
    :param center:              Position
    :param brightness_factor:   Weight utilized to adapt the value of generated non-uniform lighting artifacts.
    :param sigma:               Filter size for final Gaussian filter
    :return:
    '''
    # if radius is None: # use the smallest distance between the center and image walls
    # diameter_circle = random.randint(int(0.3*w), int(0.5 * w))
    #  define the center based on the position of disc/cup
    diameter_circle = random.randint(int(0.4 * w), int(0.7 * w))

    center =[random.randint(w/4,w*3/4),random.randint(h*3/8,h*5/8)]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    circle = dist_from_center <= (int(diameter_circle/2))

    mask = np.zeros((h, w))
    mask[circle] = 1

    num_valid = np.sum(region_mask)
    aver_color = np.sum(img) / (3*num_valid)
    if aver_color>0.25:
        brightness = random.uniform(-0.26,-0.262)
        brightness_factor = random.uniform((brightness-0.06*aver_color), brightness-0.05*aver_color)
    else:
        brightness =0
        brightness_factor =0
    # print( (aver_color,brightness,brightness_factor))
    mask = mask * brightness_factor

    rad_w = random.randint(int(diameter_circle*0.55), int(diameter_circle*0.75))
    rad_h = random.randint(int(diameter_circle*0.55), int(diameter_circle*0.75))
    sigma = 2/3 * max(rad_h, rad_w)*1.2

    if (rad_w % 2) == 0: rad_w = rad_w + 1
    if(rad_h % 2) ==0 : rad_h =rad_h + 1

    mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
    mask = np.array([mask, mask, mask])
    img = img + mask
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    hole_params = {}
    hole_params['center'] = center
    hole_params['diameter_circle'] = diameter_circle
    hole_params['brightness_factor'] = brightness_factor
    hole_params['rad_w'] = rad_w
    hole_params['rad_h'] = rad_h
    hole_params['sigma'] = sigma

    return img, hole_params

def DE_ILLUMINATION(img, region_mask, h=512, w=512):
    img, color_params = DE_COLOR(img)
    img, halo_params = DE_HALO(img, h, w, color_params['brightness_factor'])
    img, hole_params = DE_HOLE(img, h, w, region_mask)

    illum_params = {}
    illum_params['color'] = color_params
    illum_params['halo'] = halo_params
    illum_params['hole'] = hole_params

    return img, illum_params


def DE_SPOT(img, h, w, center=None, radius=None):
    '''
    :param s_num:  The number of the generated artifacts spot on the fundus image
    :param radius: Define the size of each spot
    :param center: Position of each spot on the fundus image
    :param K:      Weight of original fundus image value
    :param beta:   Weight of generated artifacts(spots) mask value (The color is adapted based on the size(radius) of each spot)
    :param sigma:  Filter size for final Gaussian filter
    '''
    spot_params = []
    s_num =random.randint(5,10)
    mask0 =  np.zeros((h, w))
    for i in range(s_num):
        # if radius is None: # use the smallest distance between the center and image walls
            # radius = min(center[0], center[1], w-center[0], h-center[1])
        radius = random.randint(math.ceil(0.01*h),int(0.05*h))

        # if center is None: # in the middle of the image
        center  = [random.randint(radius+1,w-radius-1),random.randint(radius+1,h-radius-1)]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        circle = dist_from_center <= (int(radius/2))

        k =(14/25) +(1.0-radius/25)
        beta = 0.5 + (1.5 - 0.5) * (radius/25)
        A = k *np.ones((3,1))
        d =0.3 *(radius/25)
        t = math.exp(-beta * d)

        mask = np.zeros((h, w))
        mask[circle] = np.multiply(A[0],(1-t))
        mask0 = mask0 + mask
        mask0[mask0 != 0] = 1

        sigma = (5 + (20 - 0) * (radius/25))*2
        rad_w = random.randint(int(sigma / 5), int(sigma / 4))
        rad_h = random.randint(int(sigma / 5), int(sigma / 4))
        if (rad_w % 2) == 0: rad_w = rad_w + 1
        if (rad_h % 2) == 0: rad_h = rad_h + 1

        mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
        mask = np.array([mask,mask,mask])
        img = img + mask
        img = np.maximum(img,0)
        img = np.minimum(img,1)

        spot_dict = {}
        spot_dict['radius'] = radius
        spot_dict['center'] = center
        spot_dict['rad_w'] = rad_w
        spot_dict['rad_h'] = rad_h
        spot_dict['sigma'] = sigma
        spot_params.append(spot_dict)

    return img, spot_params


def DE_BLUR(img, h, w, center=None, radius=None):
    '''
    :param sigma: Filter size for Gaussian filter
    '''
    img = (np.transpose(img, (1, 2, 0)))
    sigma = 5+(15-5) * random.random()
    rad_w = random.randint(int(sigma/3), int(sigma/2))
    rad_h = random.randint(int(sigma/3), int(sigma/2))
    if (rad_w % 2) == 0: rad_w = rad_w + 1
    if(rad_h % 2) ==0 : rad_h =rad_h + 1

    img = cv2.GaussianBlur(img, (rad_w,rad_h), sigma)
    img = (np.transpose(img, (2, 0, 1)))

    img = np.maximum(img, 0)
    img= np.minimum(img, 1)

    blur_params = {}
    blur_params['sigma'] = sigma
    blur_params['rad_w'] = rad_w
    blur_params['rad_h'] = rad_h

    return img, blur_params


def DE_process(img, mask, h, w, de_type):
    params = {}
    if de_type == '001':
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
    elif de_type == '010':
        img = transform(img)
        img = img.numpy()
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
    elif de_type == '011':
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
    elif de_type == '100':
        img = transform(img)
        img = img.numpy()
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    elif de_type == '101':    
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    elif de_type == '110':
        img = transform(img)
        img = img.numpy()
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    elif de_type == '111':
        img, illum_params = DE_ILLUMINATION(img, mask, h, w)
        params['illumination'] = illum_params
        img, spot_params = DE_SPOT(img, h, w)
        params['spots'] = spot_params
        img, blur_params = DE_BLUR(img, h, w)
        params['blur'] = blur_params
    else:
        raise ValueError('Wrong type')

    img = (np.transpose(img*mask, (1,2,0))*255).astype(np.uint8)

    return img, params