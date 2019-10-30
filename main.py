from metrics import Metric
import constant
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity
import csv

ref_image_dir = '/home/SIC/asendjas/MEGAsync Downloads/t/'#OIQA/reference_images/'
impaired_image_dir = '/home/SIC/asendjas/MEGAsync Downloads/t_/'#OIQA/distorted_images/'

def sorting_key(word):
    return int(word.replace('img', '').replace('.png', '').replace('.jpg', '').replace('.bmp', ''))

ref_images_list = sorted(filter(lambda img: img.split(
    '.')[-1] == "bmp", os.listdir(ref_image_dir)), key=sorting_key)
impaired_images_list = sorted(filter(lambda img: img.split(
    '.')[-1] == "png" or "jpg", os.listdir(impaired_image_dir)), key=sorting_key)

with open('results.csv', 'w', newline='') as csvfile:
            
    field_names = constant.METRICS

    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()

    start = constant.SUB_LIST_START
    end = constant.SUB_LIST_START
    
    for ref_image in ref_images_list:

        print('The reference image : {}\n'.format(ref_image))
        sub_imp_list = impaired_images_list[start:end]

        for impaired_image in impaired_images_list:

            print('The impaired image : {}\n'.format(impaired_image))

            ref = np.asfarray(Image.open(ref_image_dir + ref_image).convert('L'))

            imp = np.asfarray(Image.open(impaired_image_dir + impaired_image).convert('L'))

            
            writer.writerow(
                {'image': impaired_image,
                'PSNR': Metric.peakToSignalNoiseRatio(ref, imp),
                'WS-PSNR': Metric.weightedSphericalpeakToSignalNoiseRatio(ref, imp),
                'SPSNR': Metric.sphericalPeakToSignalNoiseRatio(ref, imp, True),
                'SPSNRNN': Metric.sphericalPeakToSignalNoiseRatioNN(ref, imp),
                'SSIM': np.mean(Metric.structuralSimilarityIndex(ref, imp, cs_map=False)),
                'MSSSIM': Metric.multiScaleStructuralSimilarityIndex(ref, imp),
                'GMSD': Metric.gradientMagnitudeSimilarityDeviation(ref, imp),
                'VIFp': Metric.visualInformationFidelityP(ref, imp),
                'MAE': Metric.meanAbsoluteError(ref, imp),
                'RMSE': Metric.rootMeanSquaredError(ref, imp),
                'PAMSE': Metric.perceptualFidelityAwareMeanSquaredError(ref, imp)})

     

        start = end
        end = end + constant.SUB_LIST_STEP 
        
        # print('PSNR : {}'.format(Metric.peakToSignalNoiseRatio(ref, imp)))

        # print('WS-PSNR : {}'.format(Metric.weightedSphericalpeakToSignalNoiseRatio(ref, imp)))

        # print('RMSE : {}'.format(Metric.rootMeanSquaredError(ref, imp)))

        # print('SSIM : {}'.format(np.mean(Metric.structuralSimilarityIndex(ref, imp, cs_map=False))))

        # print('MSSSIM : {}'.format(Metric.multiScaleStructuralSimilarityIndex(ref, imp)))

        # print('MAE : {}'.format(Metric.meanAbsoluteError(ref, imp)))

        # print('PAMSE : {}'.format(Metric.perceptualFidelityAwareMeanSquaredError(ref, imp)))

        # print('GMSD : {}'.format(Metric.gradientMagnitudeSimilarityDeviation(ref, imp)))

        # print('VIFP : {}'.format(Metric.visualInformationFidelityP(ref, imp)))

        # print('SSIM___ : {}'.format(structural_similarity(ref, imp, gaussian_weights=False, use_sample_covariance=False)))

        # print('SPSNR : {}'.format(Metric.sphericalPeakToSignalNoiseRatio(ref, imp, True)))

        # print('SPSNRNN : {}'.format(Metric.sphericalPeakToSignalNoiseRatioNN(ref, imp)))

