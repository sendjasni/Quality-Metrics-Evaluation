from __future__ import division
import constant
import numpy as np
import math


class Metrics:

    @staticmethod
    def peakToSignalNoiseRatio(ref_image, impaired_image):
        mean_square_error = np.mean((ref_image - ref_image) ** 2)
        if mean_square_error == 0:
            return 100
        return 20 * math.log10(constant.PIXEL_MAX / math.sqrt(mean_square_error))


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
