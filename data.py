import os

import torch.utils.data as data
from PIL import Image
from random import randrange

from torchvision.transforms import Compose, ToTensor






class EvalData_haze(data.Dataset):
    def __init__(self, train_data_dir, train_data_gt,downsample_factor):
        super().__init__()
        self.downsample_factor=downsample_factor
        fpaths = [x for x in os.listdir(train_data_dir) if any(x.endswith(extension) for extension in ["jpg","JPG","jpeg","png","bmp"])]
        haze_names = []
        gt_names = []
        for path in fpaths:
            
            haze_names.append(path.replace("\\", "/").split('/')[-1])
            gt_names.append(path.replace("\\", "/").split('/')[-1]) # same name

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.train_data_dir = train_data_dir
        self.train_data_gt = train_data_gt
        self.haze_cache = {}
        self.gt_cache = {}
        for haze_name in haze_names:
           if haze_name in self.haze_cache:
               continue
           haze_img = Image.open(self.train_data_dir + '/' + haze_name).convert('RGB')
           self.haze_cache[haze_name] = haze_img
        for gt_name in gt_names:
           if gt_name in self.gt_cache:
              continue
           gt_img = Image.open(self.train_data_gt +'/'+ gt_name).convert('RGB')
           self.gt_cache[gt_name] = gt_img
        print('use cache')

    def generate_scale_label(self, haze):
        f_scale = self.downsample_factor
        width, height = haze.size
        haze = haze.resize((int(width * f_scale), (int(height * f_scale))), resample=(Image.BICUBIC))
        #gt = gt.resize((int(width * f_scale), (int(height * f_scale))), resample=(Image.BICUBIC))
        return haze

    def get_images(self, index):
        # crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_img = self.haze_cache[haze_name]
        gt_img = self.gt_cache[gt_name]

        haze_img = self.generate_scale_label(haze_img)

        width, height = gt_img.size

        haze_crop_img=haze_img

        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])


        haze = transform_haze(haze_crop_img)
        gt=transform_gt(gt_img)


        return haze ,gt ,gt_name# ,haze1, gt1,haze2, gt2

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)



