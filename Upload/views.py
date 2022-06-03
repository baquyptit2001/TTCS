from django.shortcuts import render
from django.conf import settings
import cv2
from django.core.files.storage import FileSystemStorage
import os
import glob


age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt.txt')
gender_net = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt.txt')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Nam', 'Nữ']


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
        age_preds = get_age_predictions(roi_color)
        gender_preds = get_gender_predictions(roi_color)
        tmp = []
        file_name = fs.save('static/crop/face.png', open('face.png', 'rb'))
        tmp.append(fs.url(file_name)[7:])
        tmp.append(ageList[age_preds[0].argmax()])
        tmp.append(genderList[gender_preds[0].argmax()])
        result.append(tmp)
    cv2.imwrite('static/crop/image.png', img)
    result.append('crop/image.png')
    return result


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()
