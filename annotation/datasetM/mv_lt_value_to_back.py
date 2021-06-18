# _*_coding:utf-8_*_
# __author: guo
import os

from django.core.management.base import BaseCommand
from django.db.models import F

from annotation.models import Annotation
from datasetManage.settings import DATABASES, img_suffix
from image.models import Image, Settings, Category, Project, Merge, ProjectDataBase


class Command(BaseCommand):

    def handle(self, *args, **options):
        update_ids = []
        for obj in Annotation.objects.using("dataset_true").filter(is_active=1, status=1):
            preds = [float(obj.pred.split("_", 1)[0]), float(obj.other_pred.split(",", 1)[0])]
            top1_pred = max(preds)
            if top1_pred < 0.02:
                update_ids.append(obj.id)

        print(len(update_ids), update_ids)

