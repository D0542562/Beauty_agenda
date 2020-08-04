from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils

import math
import numpy as np
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import copy
import pandas as pd
def check(filename):
    org,_,_,_ = detect_face_landmarks(filename)
    if org is None:
        no_face = True
    else : no_face = False
    return no_face

blur = lambda X:cv2.blur(X,(3,3))
norm = lambda X:(X-np.min(X))/(np.max(X)-np.min(X))

#轉座標
def rotate_box(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        v = [coord[0],coord[1],1]
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb

#轉圖
def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))

face_cascade = cv2.CascadeClassifier('./face/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./face/shape_predictor_81_face_landmarks.dat")

# 讀圖 轉灰階 依照特徵資料在臉上瞄點 將點的座標放入list
def detect_face_landmarks(img):
    # img = cv2.imread(filename)
    if img.shape[0] > 600 or img.shape[1] > 400 :
        min_ratio = min(600/img.shape[0],400/img.shape[1])
        img = cv2.resize(img,(int(img.shape[1]*min_ratio),int(img.shape[0]*min_ratio)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 圖 upsample的次數(把圖變大 去更好的偵測臉)
    faces = detector(gray, 1)
    
    if len(faces) == 0:
        return (None,) * 4
    
    face = faces[0]
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)
    #shape2 把13個點 換順序
    shape2 = copy.deepcopy(shape)
    shape2[68] = shape[77]
    shape2[69:71] = shape[75:77]
    shape2[71:75] = shape[68:72]
    shape2[75] = shape[80]
    shape2[76:78] = shape[72:74]
    shape2[78] = shape[79]
    shape2[79] = shape[74]
    shape2[80] = shape[78]
    
#     for (x, y) in shape:
#         cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    ##圖轉正
    # 取眼睛的中心
    left_eye = np.array([shape[36], shape[39]])
    right_eye = np.array([shape[42], shape[45]])
    l_mean = np.mean(left_eye, axis=0).astype("int")
    r_mean = np.mean(right_eye, axis=0).astype("int")

    r_angle = math.atan2(r_mean[1] - l_mean[1], r_mean[0] - l_mean[0])
    rotate_angle = r_angle * 180 / math.pi

    #取圖中心點當旋轉中心
    img_h, img_w = img.shape[0:2]
    dst = rotate_bound(img, rotate_angle)

    #框圖
    #把座標旋轉
    after_coord = rotate_box(shape2,img_w/2,img_h/2,img_h,img_w,rotate_angle)
    x1, y1, w1, h1 = cv2.boundingRect(np.array(after_coord).astype('int'))
    if x1 < 0:
        after_coord = [(x + abs(x1) ,y) for (x,y) in after_coord]
        x1 = 0
    if y1 < 0:
        after_coord = [(x ,y + abs(y1)) for (x,y) in after_coord]
        y1 = 0

    #切圖
    new_dst = dst[y1:(y1+h1),x1:(x1+w1)]
    #拿到轉完的原圖 切完的圖 旋轉完的座標 最左上角的座標
    return dst,new_dst,after_coord,(x1, y1, w1, h1)

##分區
def face_region(img,point,left_up,region):
    #用框過的圖 所以要減掉左上角的點
    point = [(x-left_up[0],y-left_up[1]) for x,y in point]
    #記錄每個部位的清單
    face_dict = {
        'fronthead_range':[68,80,74],
        'between_eyebrow':[20,23,27,74],
        'left_eye_left':[0,1,36,17],
        'right_eye_right':[15,16,45,26],
        'left_eye_down':[0,1,36,39,29],
        'right_eye_down':[15,16,42,45],
        'left_cheek':[1,28,8],
        'right_cheek':[15,28,8],
        'nose':[48,54,41,46],
        'all':[0,8,16,74],
        'left_eye':[36,37,39,41],
        'right_eye':[42,43,44,45,46,47],
        'mouth':[48,50,52,54,56,57]
    }
    points = [point[idx] for idx in face_dict[region]]
    
    x1, y1, w1, h1 = cv2.boundingRect(np.array(points).astype('int'))
    new_dst = img[y1:(y1+h1),x1:(x1+w1)]
    return new_dst,(x1, y1)

def region(new_dst,new_left_up,point,left_up,region):
    w = new_dst.shape[0]
    h = new_dst.shape[1]
    points = [(x - left_up[0] - new_left_up[0] ,y - left_up[1] - new_left_up[1]) for x,y in point]
    if region == 'fronthead_range':
        poly_point = [points[idx] for idx in list([72,19,24,76])]
        poly_format = [(x*0.9+0.1*w,y*0.8+0.05*h) for x,y in poly_point]
    
    elif region == 'between_eyebrow':
        poly_point = [points[idx] for idx in list([20,27,23,74])]
        poly_format = [(x*0.8+0.05*w,y*0.7+0.3*h) for x,y in poly_point]
        
    elif region == 'left_eye_left':
        poly_point = [points[idx] for idx in list([1,41,36,0])]
        poly_point[3] = (poly_point[3][0]+(poly_point[2][0] - poly_point[3][0])*0.3,poly_point[3][1]+(poly_point[2][1] - poly_point[3][1])*0.3)
        poly_point[0] = (poly_point[0][0]+(poly_point[1][0] - poly_point[0][0])*0.3,poly_point[0][1]+(poly_point[1][1] - poly_point[0][1])*0.3)
        poly_format = poly_point
        
    elif region == 'right_eye_right':
        poly_point = [points[idx] for idx in list([45,46,15,16])]
        poly_point[3] = (poly_point[3][0]+(poly_point[2][0] - poly_point[3][0])*0.3,poly_point[3][1]+(poly_point[2][1] - poly_point[3][1])*0.3)
        poly_point[0] = (poly_point[0][0]+(poly_point[1][0] - poly_point[0][0])*0.3,poly_point[0][1]+(poly_point[1][1] - poly_point[0][1])*0.3)
        poly_format = poly_point
        
    elif region == 'left_eye_down':
        poly_point = [points[idx] for idx in list([36,41,40,39])]
        points_2 = [(x,y*5+0.2*h) for x,y in ([poly_point[-1]]+[poly_point[0]])]
        poly_point.extend(points_2)
        poly_format = poly_point
        
    elif region == 'right_eye_down':
        poly_point = [points[idx] for idx in list([42,47,46,45])]
        points_2 = [(x,y*5+0.5*h) for x,y in ([poly_point[-1]]+[poly_point[0]])]
        poly_point.extend(points_2)
        poly_format = poly_point 
        
    elif region == 'left_cheek':
        poly_point = [points[idx] for idx in list([36,41,40,39,31,48])]
        poly_format = poly_point
        
    elif region == 'right_cheek':
        poly_point = [points[idx] for idx in list([42,47,46,45,54,35])]
        poly_format = poly_point
        
    elif region == 'nose':
        poly_point = np.array([(0,0),
                   (w+new_left_up[0],0),
                   (w+new_left_up[0],h+new_left_up[1]),
                   (0,h+new_left_up[1])])
        poly_format = poly_point
        
    elif region == 'all':
        poly_point = [points[idx] for idx in list(list(range(0,17)) + list(range(80,67,-1)))]
        poly_format = poly_point
    elif region == 'left_eye':
        poly_point = [points[idx] for idx in list(range(36,42))]
        poly_format = poly_point
    elif region == 'right_eye':
        poly_point = [points[idx] for idx in list(range(42,48))]
        poly_format = poly_point
    elif region == 'mouth':
        poly_point = [points[idx] for idx in list(range(48,59))]
        poly_format = poly_point

    return poly_point,poly_format

def no_eye(img,point,left_up):
    new_dst,new_left_up = face_region(img,point,left_up,'all')
    plot = np.zeros(new_dst.shape[:2], np.uint8)
    
    for re in list(['left_eye','right_eye','mouth']):
        poly_point1,poly_format1 = region(new_dst,new_left_up,point,left_up,re)
        points = poly_point1
        cv2.fillPoly(plot,[np.array(poly_format1).astype("int")],255)

    masked = cv2.bitwise_and(new_dst, new_dst, mask=(255-plot))

    ##把邊框捕掉
    masked2 = masked.copy()
    for i in range(3):
        masked2[:,:,i] = np.where(masked[:,:,i]==0, np.mean(masked[:,:,i]), masked[:,:,i])
    masked2 = cv2.blur(masked2,(15,15))

    for i in range(3):
        masked[:,:,i] = np.where(masked[:,:,i]==0, masked2[:,:,i], masked[:,:,i])
    masked = np.array(masked, dtype=np.uint8)
    return masked

def crop_pic(img,select_region):
    org,img,point,left_up = detect_face_landmarks(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if org is None:
    #     print("there is no face.")
    new_dst,new_left_up = face_region(img,point,left_up,select_region)
    
    poly_point,poly_format = region(new_dst,new_left_up,point,left_up,select_region)
    points = poly_point
    plot = np.zeros(new_dst.shape[:2], np.uint8)

    cv2.fillPoly(plot,[np.array(poly_format).astype("int")],255)
    masked = cv2.bitwise_and(new_dst, new_dst, mask=plot)

    poly_format = np.array(poly_format).astype("int")
    poly_format = np.asarray(poly_format)
    poly_format[poly_format<0] = 0
    x1,y1,w1,h1 = min(poly_format[:,0]),min(poly_format[:,1]),max(poly_format[:,0]),max(poly_format[:,1])
    masked = masked[y1:h1,x1:w1]
    
    ##把邊框捕掉
    masked2 = masked.copy()
    for i in range(3):
        masked2[:,:,i] = np.where(masked[:,:,i]==0, np.mean(masked[:,:,i]), masked[:,:,i])
    masked2 = cv2.blur(masked2,(15,15))

    for i in range(3):
        masked[:,:,i] = np.where(masked[:,:,i]==0, masked2[:,:,i], masked[:,:,i])

    masked = np.array(masked, dtype=np.uint8)
    return masked

    #閉運算 先膨脹再侵蝕
def close(img, d, e, k):
    img = cv2.dilate(img, np.ones(k, np.uint8), iterations=d)
    img = cv2.erode(img, np.ones(k), iterations=e)
    return img

def black_circle(filename,select_region):
    masked = crop_pic(filename,select_region)
    imhsv = cv2.cvtColor(masked,cv2.COLOR_RGB2HSV)
    V = imhsv[:,:,2]
    v2 = cv2.applyColorMap(V,cv2.COLORMAP_BONE)
    #找出黑眼圈的threshold
    V2 = norm(V)
    m = (V2-np.mean(V2)) / np.std(V2)
    m = np.where(m<0, 0, m)
    _,m3 = cv2.threshold(m , 0.75, 1, cv2.THRESH_BINARY)
    m4 = copy.deepcopy(m)
    m4[np.where(m3==0)] = 0
    _,m4 = cv2.threshold(m4 , 0.99999, 1, cv2.THRESH_BINARY)
    #抓到的黑眼圈比例
    black_circle = np.sum(m4)/(m3.shape[0]*m3.shape[1])
    return black_circle

def eye_winkle(filename,select_region):
    masked = crop_pic(filename,select_region)
    imhsv = cv2.cvtColor(masked,cv2.COLOR_RGB2HSV)
    S = imhsv[:,:,1]

    S2_x = cv2.Sobel(S,cv2.CV_16S,1,0)
    S2_y = cv2.Sobel(S,cv2.CV_16S,0,1)

    S2_x = cv2.convertScaleAbs(S2_x)
    S2_y = cv2.convertScaleAbs(S2_y)

    S2 = cv2.addWeighted(S2_x,0.5,S2_y,0.5,0)        
    S2 = np.abs(S2 - np.mean(S2))
    S2 = norm(S2)
    m = (S2-np.mean(S2)) / np.std(S2)
    m = np.where(m<0, 0, m)
    m = close(m, 1, 1, 3)
    m = blur(m)
    #找出皺紋的threshold
    _,m3 = cv2.threshold(m , 0.99999, 1, cv2.THRESH_BINARY)
    m4 = copy.deepcopy(m)
    m4[np.where(m3==0)] = 0
    
    #閉運算之後的抓皺紋圖 面積
    eye_winkle = np.sum(m4)/(m3.shape[0]*m3.shape[1])
    return eye_winkle

def fronthead_range(filename):
    masked = crop_pic(filename,'fronthead_range')
    imhsv = cv2.cvtColor(masked,cv2.COLOR_RGB2HSV)
    S = imhsv[:,:,1]
    
    S2 = cv2.Sobel(S,cv2.CV_16S,0,1)
    
    S2 = np.abs(S2 - np.mean(S2))
    S2 = norm(S2)
    m = (S2-np.mean(S2)) / np.std(S2)
    m = np.where(m<0, 0, m)
    m = close(m, 1, 1, 3)
    m = blur(m)
    _,m3 = cv2.threshold(m , 0.99999, 1, cv2.THRESH_BINARY)
    m4 = copy.deepcopy(m)
    m4[np.where(m3==0)] = 0
    m4 = close(m4, 1, 2, 3)

    fronthead_range = np.sum(m4)/(m3.shape[0]*m3.shape[1])
    return fronthead_range

def vertical_wrinkle(filename,select_region):
    masked = crop_pic(filename,select_region)
    imhsv = cv2.cvtColor(masked,cv2.COLOR_RGB2HSV)
    S = imhsv[:,:,1]
    
    S2 = cv2.Sobel(S,cv2.CV_16S,1,0)
    S2 = np.abs(S2 - np.mean(S2))
    S2 = norm(S2)
    m = (S2-np.mean(S2)) / np.std(S2)
    m = np.where(m<0, 0, m)
    m = close(m, 1, 1, 3)
    m = blur(m)

    #找出皺紋的threshold
    _,m3 = cv2.threshold(m , 0.99999, 1, cv2.THRESH_TRUNC)
    m4 = copy.deepcopy(m)
    m4[np.where(m3==0)] = 0
    m4 = close(m4, 1, 3, 3)

    #抓到的皺紋比例
    vertical_wrinkle = np.sum(m4)/(m3.shape[0]*m3.shape[1])
    return vertical_wrinkle
