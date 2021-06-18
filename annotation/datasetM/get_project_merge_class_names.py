# _*_coding:utf-8_*_
# __author: guo
from django.core.management.base import BaseCommand

from datasetManage.settings import DATABASES
from image.models import Image, Settings, Category, Project, Merge, ProjectDataBase, Dataset
from annotation.models import Annotation


class Command(BaseCommand):

    def handle(self, *args, **options):
        title = "ddtn_thirty-six"
        project = Project.objects.filter(title=title).first()

        pds = ProjectDataBase.objects.filter(project=project, name="dataset_manage").values_list("id", flat=1)

        class_ids = set()
        for d in Dataset.objects.filter(pb_id__in=pds):
            merge_map = {
                f"{merge.class_name_id}": set(merge.merge_class_name.split(","))
                for merge in Merge.objects.filter(pb_id=d.pb_id)
            }

            cur_class_ids = set(d.required.split(","))
            for cur_class_id in cur_class_ids.copy():
                if cur_class_id in merge_map:
                    cur_class_ids |= merge_map[cur_class_id]

            class_ids |= cur_class_ids

        print(len(class_ids))
        c = dict(Category.objects.using("dataset_true").filter(id__in=class_ids).values_list("id", "value"))
        # for _id, v in list(c.items()):
        #     if v.startswith("垃圾桶-"):
        #         c.pop(_id)

        print(",".join(map(str, c.keys())))
        print(list(c.values()), len(c.values()))



