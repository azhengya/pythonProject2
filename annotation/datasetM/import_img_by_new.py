# _*_coding:utf-8_*_
# __author: guo
import os

from django.core.management.base import BaseCommand
from django.db.models import F

from annotation.models import Annotation
from datasetManage.settings import DATABASES, img_suffix
from image.models import Image, Settings, Category, Project, Merge, ProjectDataBase


class Command(BaseCommand):
    databases_dict = {
        v["NAME"]: k
        for k, v in DATABASES.items()
    }
    path = "/home/dd/Share/dataset_pool/other"

    def handle(self, *args, **options):
        data = {}
        for root, _dir, files in os.walk(self.path):
            for file in files:
                if "." in file and file.rsplit(".", 1)[-1] in img_suffix:
                    data.setdefault(root, []).append(file)

        for root, files in data.items():
            class_name = root.rsplit("/", 1)[-1]
            print(class_name)
            obj = Category.objects.using("dataset_true").filter(value=class_name).first()
            if not obj:
                obj = Category.objects.using("dataset_true").create(value=class_name)
                print(class_name)

            for file in files:
                img, _ = Image.objects.using("dataset_true").get_or_create(local_path=os.path.join(root, file),defaults=dict(source=0))
                # img = Image.objects.using("dataset_true").filter(local_path=os.path.join(root, file)).first()

                Annotation.objects.using("dataset_true").update_or_create(img=img, defaults=dict(pred=f"1_{class_name}",status=1, classify_id=obj.id))
                # Annotation.objects.using("dataset_true").filter(img=img).update(img=img, pred=f"1_{class_name}", status=1, classify=obj)
