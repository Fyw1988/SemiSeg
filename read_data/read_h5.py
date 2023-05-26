import h5py


# img_path = '/root/data/dataset/Pancreas-CT/Pancreas_h5/image0003_norm.h5'
# img_path = '/root/data/fyw/BraTS2019/data/BraTS19_2013_0_1.h5'
img_path = '/root/data/dataset/Prostate2022/data/CHENG_BEN_YAO_norm2.h5'

h5f = h5py.File(img_path, 'r')
image = h5f['image'][:]
label = h5f['label'][:]
print([key for key in h5f.keys()])
print(image.shape)   # (174, 149, 88)  # (131, 115, 132)
print(label.shape)
