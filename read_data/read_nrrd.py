import nrrd
from glob import glob


listt = glob('D:/Pycharm_Projects/ZQ/Datasets/Pancreas-CT/Pancreas-CT_nrrd/Training Set/*/lgemri.nrrd')
print(listt[0])
image, img_header = nrrd.read(listt[0])
label, gt_header = nrrd.read(listt[0].replace('lgemri.nrrd', 'laendo.nrrd'))
print(image.shape)   # (576, 576, 88)
print(label.shape)
