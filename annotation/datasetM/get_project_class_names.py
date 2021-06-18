# _*_coding:utf-8_*_
# __author: guo
from django.core.management.base import BaseCommand

from datasetManage.settings import DATABASES
from image.models import Image, Settings, Category, Project, Merge, ProjectDataBase, Dataset
from annotation.models import Annotation


class Command(BaseCommand):

    def handle(self, *args, **options):
        title = "dddp"
        project_list = Project.objects.exclude(title=title)

        for project in project_list:
            pds = ProjectDataBase.objects.filter(project=project, name="dataset_manage").values_list("id", flat=1)

            result = {}
            for d in Dataset.objects.filter(pb_id__in=pds):
                merge_map = {
                    f"{merge.class_name_id}": set(merge.merge_class_name.split(","))
                    for merge in Merge.objects.filter(pb_id=d.pb_id)
                }

                for main_class_name_id, merge_class_name_ids in merge_map.items():
                    main_class_name = Category.objects.using("dataset_true").filter(id=main_class_name_id).values_list("value", flat=1).first()
                    merge_class_names = Category.objects.using("dataset_true").filter(id__in=merge_class_name_ids).values_list("value", flat=1)
                    if main_class_name is None:
                        print(main_class_name_id, merge_class_names)

                    result[main_class_name] = list(merge_class_names)

            print(project.title, result, len(result))
            print("\n\n\n\n")



