# _*_coding:utf-8_*_
# __author: guo
import os

from django.core.management.base import BaseCommand
from django.db.models import F

from annotation.models import Annotation
from datasetManage.settings import DATABASES, img_suffix
from image.models import Image, Settings, Category, Project, Merge, ProjectDataBase



class Command(BaseCommand):
    path = "/home/dd/Share/dataset_pool/dd531/train"
    using = "default"

    def handle(self, *args, **options):
        data = {}
        for root, _dir, files in os.walk(self.path):
            for file in files:
                if "." in file and file.rsplit(".", 1)[-1] in img_suffix:
                    data.setdefault(root.rsplit("/", 1)[1], []).append(os.path.join(root, file))
        print(data)
        for class_name, files in data.items():
            print(class_name)
            obj = Category.objects.using(self.using).filter(value=class_name).first()
            if not obj:
                obj = Category.objects.using(self.using).create(value=class_name)
                print(class_name)

            for file in files:
                img = Image.objects.using(self.using).filter(local_path=file).first()
                if not img:
                    img = Image.objects.using(self.using).create(local_path=file, source=1000)
                    print(file)

                num = Annotation.objects.using(self.using).filter(img=img).update(classify=obj, status=1, version=F("version") + 1)
                if not num:
                    Annotation.objects.using(self.using).create(img_id=img.id, classify=obj, status=1, version=1)
