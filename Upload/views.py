from django.shortcuts import render
from django.conf import settings
from PIL import Image
from django.http import HttpResponse
import cv2
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.templatetags.static import static
import os
import glob

# Create your views here.
from Upload import models


def index(request):
    if request.method == 'POST':
        # upload(request)
        files = glob.glob(str(settings.STATICFILES_DIRS[0]) + '/' + 'crop/*')
        for f in files:
            os.remove(f)
        fs = FileSystemStorage()
        extension = os.path.splitext(request.FILES['image'].name)[1]
        file_save = 'static/crop/' + 'raw' + extension
        file_name = fs.save(file_save, request.FILES['image'])
        result = crop_face(file_save)
        if result is None:
            return render(request, 'index.html', {'error': 'No face detected'})
        img_big = result.pop()
        return render(request, 'result.html', {'result': result, 'img_big': img_big})
    name = 'TTCS'
    return render(request, 'index.html', {'name': name})


def crop_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    result = []
    fs = FileSystemStorage()
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.imwrite('face.png', roi_color)
        file_name = fs.save(str(settings.STATICFILES_DIRS[0]) + '/' + 'crop/face.png', open('face.png', 'rb'))
        result.append(fs.url(file_name)[7:])
    cv2.imwrite(str(settings.STATICFILES_DIRS[0]) + '/' + 'crop/image.png', img)
    result.append('crop/image.png')
    return result
