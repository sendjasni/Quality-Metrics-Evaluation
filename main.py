from metrics import Metric
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity
import csv

ref_images_list = filter(lambda img: img.split(
    '.')[-1] == "bmp", os.listdir('RefImages/'))
impaired_images_list = filter(lambda img: img.split(
    '.')[-1] == "bmp", os.listdir('ImpairedImages/'))

for ref_image in ref_images_list:
    for impaired_image in impaired_images_list:
        print('The reference image : {}'.format(ref_image))
        print('The impaired image : {}\n'.format(impaired_image))

        ref = np.asfarray(Image.open('RefImages/' + ref_image).convert('L'))

        imp = np.asfarray(Image.open('ImpairedImages/' + impaired_image))

        with open('results.csv', 'w', newline='') as csvfile:
            field_names = ['image', 'PSNR', 'WS-PSNR', 'SPSNR', 'SPSNRNN', 'SSIM',
                           'MSSSIM', 'GMSD', 'VIFp', 'MAE', 'RMSE', 'PAMSE']
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
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
