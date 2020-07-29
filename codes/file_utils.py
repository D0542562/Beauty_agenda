#!/usr/bin/env python3
# coding=utf-8
# -*- coding: UTF-8 -*-
import os
import datetime
#import magic
import uuid

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
ALLOWED_MIME_TYPES = {'image/jpeg'}



def is_allowed_file(file):
    if '.' in file.filename:
        ext = file.filename.rsplit('.', 1)[1].lower()
    else:
        return False

    return True

    #mime_type = magic.from_buffer(file.stream.read(), mime=True)
    file.stream.seek(0)
    if (
        mime_type in ALLOWED_MIME_TYPES and
        ext in ALLOWED_EXTENSIONS
    ):
        return True

    return False
