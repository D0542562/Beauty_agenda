#!/usr/bin/env python3
# coding=utf-8
# -*- coding: UTF-8 -*-
import os
import sys
import datetime
from flask import Flask, jsonify, request, make_response, send_file, abort, render_template
import numpy as np
import cv2

from file_utils import *
from foo import *
from dot_foo import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def main():
    try:
        f1 = request.files['image']
        filestr1 = f1.read()
        npimg = np.fromstring(filestr1, np.uint8)
        # 圖片解碼，內容相當於 cv2.imread
        bgr1 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if not is_allowed_file(f1):
            return {
                'error': 'Not JPG image'
            }
    except Exception as e:
        return {
            'error': str(e)
        }

    # Main section
    # 主要程式放這裡
    is_no_face = check(bgr1)
    if is_no_face:
        forehead = False
        chuan = False
        crow = False
        dark_circle = False
        smile_line = False
        acne = False
        freckle = False
    
    else:
        forehead = fronthead_range(bgr1)

        chuan = vertical_wrinkle(bgr1,'between_eyebrow')

        left_eye_winkle = eye_winkle(bgr1,'left_eye_left')
        right_eye_winkle = eye_winkle(bgr1,'right_eye_right')

        left_black_circle = black_circle(bgr1,'left_eye_down')
        right_black_circle = black_circle(bgr1,'right_eye_down')
        
        have_acne = dot(bgr1,'all')

        left_cheek = vertical_wrinkle(bgr1,'left_cheek')
        right_cheek = vertical_wrinkle(bgr1,'right_cheek')
        nose = vertical_wrinkle(bgr1,'nose')

        freckle = little_dot(bgr1)
        #抬頭紋 forehead
        forehead = True if forehead > 0.2 else False
        #川字紋 chuan
        chuan = True if chuan > 0.225 else False
        #魚尾紋 crow
        left_eye_winkle = True if left_eye_winkle > 0.2262 else False
        right_eye_winkle = True if right_eye_winkle > 0.2262 else False
        crow = True if left_eye_winkle and right_eye_winkle == True else False
        #黑眼圈
        left_black_circle = True if left_black_circle > 0.09 else False
        right_black_circle = True if right_black_circle > 0.09 else False
        dark_circle = True if left_black_circle and right_black_circle == True else False
        #法令紋 smile_line
        left_cheek = True if left_cheek > 0.18 else False
        right_cheek = True if right_cheek > 0.18 else False
        nose = True if nose > 0.19 else False
        smile_line = True if left_cheek and right_cheek and nose else False
        #痘痘
        acne = True if have_acne > 3 else False
        #雀斑
        freckle = True if freckle > 150  else False

    # 整理回傳資料
    data = {
        'is_no_face':is_no_face,
        'forehead':forehead,
        'chuan':chuan,
        'crow':crow,
        'dark_circle':dark_circle,
        'smile_line':smile_line,
        'acne': acne,
        'freckle': freckle
    }

    return data


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
    sys.exit()
