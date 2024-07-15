import SimpleITK as sitk
import numpy as np
import os
from utils import crop_save_head_image

folder_path = '/home/pablo/HeadPlans/HN-CHUM-018/'
file_path = '/home/pablo/HeadPlans/HN-CHUM-018/CT_segmentation.mhd'
itkimage = sitk.ReadImage(file_path)
segmentation = sitk.GetArrayFromImage(itkimage)
print(segmentation.shape)
segmentation = np.transpose(segmentation, (2, 1, 0))
# See how many labels are in the segmentation
print(np.unique(segmentation))

brain_segmentation = np.where(segmentation == 38, 1, 0)

uncropped_shape = [272, 272, 176]
img_name = 'CT_brain_segmentation'

# CT
CT_file_path = os.path.join(folder_path, 'CT.raw')
# Read the img
with open(CT_file_path, 'rb') as f:
    CT = np.frombuffer(f.read(), dtype=np.int16).reshape(uncropped_shape, order='F')

x_min, x_max = 12, -12
y_min, y_max = 27, -105
CT = CT[x_min:x_max, y_min:y_max, :].astype(np.float32)
brain_segmentation = brain_segmentation[x_min:x_max, y_min:y_max, :].astype(np.float32)
print(np.unique(brain_segmentation))

print(CT.shape)

# Masking the CT
CT_masked = CT * brain_segmentation

CT_masked = CT_masked.transpose(2, 1, 0)
CT_masked.tofile(os.path.join(folder_path, f'{img_name}.raw'))


CT_cropped = CT.transpose(2, 1, 0)
CT_cropped.tofile(os.path.join(folder_path, 'CT_cropped.raw'))

    

# superior lobe of left lung
# inferior lobe of left lung
# superior lobe of right lung
# inferior lobe of right lung
# esophagus  # air
# trachea  # air
# thyroid
# T5 vertebra
# T4 vertebra
# T3 vertebra
# T2 vertebra
# T1 vertebra
# C7 vertebra
# C6 vertebra
# C5 vertebra
# C4 vertebra
# C3 vertebra
# C2 vertebra
# C1 vertebra
# aorta
# brachiocephalic trunk
# right subclavian artery
# left subclavian artery
# right common carotid artery
# left common carotid artery
# left brachiocephalic vein
# right brachiocephalic vein
# superior vena cava
# left humerus
# right humerus
# left scapula
# right scapula
# left clavicle
# right clavicle
# spinal cord
# left deep back muscle
# right deep back muscle
# brain
# skull
# right rib 4
# right rib 3
# left rib 1
# left rib 2
# left rib 3
# left rib 4
# left rib 5
# right rib 1
# right rib 2
# right rib 5
# sternum
# costal cartilage