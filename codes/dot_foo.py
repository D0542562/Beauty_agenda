import numpy as np
import cv2
import time
from PIL import Image
from time import sleep
import gc
import random, shutil
import math
import matplotlib.pyplot as plt
from skimage import measure

from foo import *

def imadjust(x,a,b,c,d,gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def bwareaopen(image,size):
    # image 是單通道二值圖 uint8
    # size 黑底上的白區域
    output=image.copy()
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype('uint8'))
    #stats x y w h area(不規則的面積)
    #扣掉第一個(全圖)
    q1 = np.quantile(stats[:,4],0.25)
    q3 = np.quantile(stats[:,4],0.75)
    iqr = q3 - q1
    con1 = q3 + 1.5*iqr
    con2 = q1 - 1.5*iqr
    for i in range(1,nlabels-1):
        regions_size=stats[i,4]
        if regions_size<size or regions_size > con1 or regions_size < con2:
            x0=stats[i,0]
            y0=stats[i,1]
            x1=stats[i,0]+stats[i,2]
            y1=stats[i,1]+stats[i,3]
            for row in range(y0,y1):
                for col in range(x0,x1):
                    if labels[row,col]==i:
                        output[row,col]=0
    
    return output

def dot(filename,select_region,BTh = 0.75,AreaMin = 30,AreaMax = 6000):
    im = crop_pic(filename,select_region)
    imG = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    imN = norm(imG)

    imHsv = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)

    V = imHsv[:,:,2] # brightness plane
    V = (V - np.amin(V)) / (np.amax(V) - np.amin(V))

    imGoff = (V - imN)
    imGoff = imadjust(imGoff,imGoff.min(),imGoff.max(),0,1)

    imB = imGoff > BTh
    imB = np.asarray([[1 if v else 0 for v in row] for row in imB])

    imB = bwareaopen(imB,AreaMin)
    label = measure.label(imB)
    CC = np.argwhere(10 == label)
    S = measure.regionprops(label)

    count = len(S)
    return count

def little_dot(file):
    im = crop_pic(file,'all')
    imG = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    imN = norm(imG)
    imHsv = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    S = imHsv[:,:,1] # brightness plane
    V = imHsv[:,:,2]
    S = norm(S)
    V = norm(V)
    imGoff = (S - imN)
    im_v = (V - imN)
    imGoff = imadjust(imGoff,imGoff.min(),imGoff.max(),0,1)
    im_v = imadjust(im_v,im_v.min(),im_v.max(),0,1)
    plot = np.zeros(imHsv.shape[:2])

    con1 = np.logical_and(imGoff > 0.5,imGoff < 1)
    con2 = np.logical_and(im_v > 0.5,im_v < 1)
    tmp = np.where(np.logical_and(con1,con2))
    plot[tmp] = 1
    
    a, cnts = cv2.findContours(plot.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    area_list = [cv2.contourArea(a1) for a1 in a if cv2.contourArea(a1) > 0]
    
    a = [a1 for a1 in a if (cv2.contourArea(a1) < (im.shape[0]*im.shape[1])*0.00001)]

    little_count = len(a)
    
    
    return little_count