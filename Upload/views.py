from django.shortcuts import render
from django.conf import settings
from PIL import Image

# Create your views here.
from Upload import models


def index(request):
    if request.method == 'POST':
        upload(request)
    else:
        name = 'TTCS'
        print(settings.MEDIA_ROOT)
        return render(request, 'index.html', {'name': name})


def upload(request):
    if request.method == 'POST':
        image = request.FILES['image']
        image_name = image.name
        image_path = settings.MEDIA_ROOT + image_name
        with open(image_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)
        img = Image.open(image_path)
        img.save(image_path, 'JPEG')
        models.Image.objects.create(image=image_name)
        return render(request, 'index.html')
