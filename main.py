from metrics import Metric
import constant
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity
import csv

ref_image_dir = 'ref_images/'#'/home/SIC/asendjas/MEGAsync Downloads/OIQA/reference_images/'
impaired_image_dir = 'imp_images/'#'/home/SIC/asendjas/MEGAsync Downloads/OIQA/distorted_images/'


def sorting_key(word):
    return int(word.replace('img', '').replace('.png', '').replace('.jpg', '').replace('.bmp', ''))


ref_images_list = sorted(filter(lambda img: img.split(
    '.')[-1] == "bmp", os.listdir(ref_image_dir)), key=sorting_key)
impaired_images_list = sorted(filter(lambda img: img.split(
    '.')[-1] == "png" or "jpg", os.listdir(impaired_image_dir)), key=sorting_key)

print('Ref image list size : {}\n'.format(len(ref_images_list)))
print('Imp image list size : {}\n'.format(len(impaired_images_list)))

start = constant.SUB_LIST_START
end = constant.SUB_LIST_END
    
sub_imp_list = impaired_images_list[start:end]

with open('results.csv', 'w', newline='') as csv_file:

    field_names = constant.METRICS

    writer = csv.DictWriter(csv_file, fieldnames=field_names)
    writer.writeheader()

    for ref_image in ref_images_list:
        
        print('The reference image : {}\n'.format(ref_image))
        # print('sub list size : {}'.format(len(sub_imp_list)))
        # print('START : {}, END :  {}'.format(start, end))

        for impaired_image in sub_imp_list:

            print('The impaired image : {}\n'.format(impaired_image))

            ref = np.asfarray(Image.open(
                ref_image_dir + ref_image).convert('L'))

            imp = np.asfarray(Image.open(
                impaired_image_dir + impaired_image).convert('L'))

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
                'PAMSE': Metric.perceptualFidelityAwareMeanSquaredError(ref, imp),
                'RECO': Metric.relativeEdgeCoherence(ref, imp)})
        

        start = end
        end = end + constant.SUB_LIST_STEP
        sub_imp_list = impaired_images_list[start:end]
