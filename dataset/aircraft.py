from .base import *


class Aircraft(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)

        self.classes_name = []
        assert classes in [range(0, 50), range(50, 100)]
        start = 0
        if classes == range(50, 100):
            start = 50
        count = 0
        index = 0
        with open(os.path.join(source, "variants.txt")) as f:
            for variant in f.readlines():
                if count in classes:
                    self.classes_name.append(variant.strip())
                count = count + 1
        class_names = []
        with open(os.path.join(source, "images_variant_trainval.txt")) as f:
            for pic_data in f.readlines():
                class_name = str(pic_data).split(" ", 1)[1][:-1]
                class_names.append(class_name)
                if class_name in self.classes_name:
                    pic_name = str(pic_data).split(" ")[0] + ".jpg"
                    self.im_paths.append(os.path.join(root, pic_name))
                    self.ys.append(int(self.classes_name.index(class_name)) + start)
                    self.I += [index]
                    index = index + 1
        with open(os.path.join(source, "images_variant_test.txt")) as f:
            for pic_data in f.readlines():
                class_name = str(pic_data).split(" ", 1)[1][:-1]
                class_names.append(class_name)
                if class_name in self.classes_name:
                    pic_name = str(pic_data).split(" ")[0] + ".jpg"
                    self.im_paths.append(os.path.join(root, pic_name))
                    self.ys.append(int(self.classes_name.index(class_name)) + start)
                    self.I += [index]
                    index = index + 1


