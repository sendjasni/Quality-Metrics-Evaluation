import metrics
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import mean_squared_error

ref_images_list = filter(lambda img: img.split(
    '.')[-1] == "bmp", os.listdir('RefImages/'))
impaired_images_list = filter(lambda img: img.split(
    '.')[-1] == "bmp", os.listdir('ImpairedImages/'))

for ref_image in ref_images_list:
    
    print(ref_image)
    ref = np.asfarray(Image.open('RefImages/' + ref_image).convert('L'))
    
    print (ref.shape)
    print (ref.dtype)
    
    imp =  np.asfarray(Image.open('ImpairedImages/impaired_image.bmp'))
    
    print (imp.dtype)
    
    # cv2.imwrite("impaired_image.bmp", imp)    
    # cv2.imshow('Reference', ref)
    # cv2.imshow('Impaired', imp)
    # cv2.waitKey(0)

    print(metrics.Metrics.peakToSignalNoiseRatio(ref, imp))

    print(metrics.Metrics.weightedSphericalpeakToSignalNoiseRatio(ref, imp))