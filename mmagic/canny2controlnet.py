import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('/home/xiaoqiang/mlearning/mmlab2/mmagic/configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

canny_path = '/home/xiaoqiang/mlearning/dataset/Demo/house.jpeg'
img = mmcv.imread(canny_path)
img = cv2.Canny(img, 100, 200)
img = img[:, :, None]
img = np.concatenate([img]*3, axis=2)
img = Image.fromarray(img)


prompt = 'Room with blue walls and a yello ceiling'


if __name__ == '__main__':
    output_dict = controlnet.infer(prompt, control=img)
    samples = output_dict['samples']
    for idx, sample in enumerate(samples):
        sample.save(f'sample_{idx}.png')
    controls = output_dict['controls']
    for idx, control in enumerate(controls):
        control.save(f'control_{idx}.png')