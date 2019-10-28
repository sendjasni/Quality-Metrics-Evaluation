from __future__ import division
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity

import constant
import numpy as np


class Metrics:

    @staticmethod
    def peakToSignalNoiseRatio(ref_image, impaired_image):
        mse_value = mean_squared_error(ref_image, impaired_image)
        print('The MSE is : {}'.format(mse_value))
        if mse_value == 0:
            return 100
        return 10.0 * np.log10(constant.PIXEL_MAX ** 2 / mse_value)

    @staticmethod
    def rootMeanSquaredError(ref_image, impaired_image):
        return np.sqrt(mean_squared_error(ref_image, impaired_image))

    @staticmethod
    def weightedSphericalpeakToSignalNoiseRatio(ref_image, impaired_image):
        height, width = ref_image.shape

        weight_list = [[math.cos((i + 0.5 - height / 2) * math.pi / height)
                        for j in range(width)] for i in range(height)]

        weighted_mean_square_error = 0.0

        for i in range(height):
            for j in range(width):
                weighted_mean_square_error += (((int(ref_image[i][j]) - int(impaired_image[i][j]))
                                                ** 2) * weight_list[i][j])
            weighted_mean_square_error = (
                weighted_mean_square_error / (width * height))

        if weighted_mean_square_error == 0:
            return 100

        weighted_spherical_psnr = 10 * np.log10((constant.PIXEL_MAX **
                                                 2 / weighted_mean_square_error))

        return weighted_spherical_psnr

    @staticmethod
    def structuralSimilarityIndex(ref_image, impaired_image, multi_chanel):
        if multi_chanel:
            return structural_similarity(ref_image, impaired_image, multichannel=True)
        else:
            return structural_similarity(ref_image, impaired_image, multichannel=False)
