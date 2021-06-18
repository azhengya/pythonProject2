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
        c = dict(Category.objects.using("dataset_true").filter(is_active=1).values_list("value", "id"))

        for obj in Annotation.objects.using("dataset_true").filter(classify_id=1653, pred__gt="0.05", status=0, is_active=1):
            _, class_name = obj.pred.split("_", 1)
            c_id = c[class_name]
            Annotation.objects.using("dataset_true").filter(id=obj.id).update(classify_id=c_id)

