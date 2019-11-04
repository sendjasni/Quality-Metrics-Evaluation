from __future__ import division
from sklearn.metrics import mean_squared_error
# from sp import ssim
from scipy import ndimage
from scipy import signal

import constant
import math
import numpy as np


class Metric:

    @staticmethod
    def peakToSignalNoiseRatio(ref_image, impaired_image):
        print('Computing PSNR ...')
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
        print('Computing WS-PSNR ...')
        height, width = ref_image.shape

        weight_list = [[np.cos((i + 0.5 - height / 2) * np.pi / height)
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
    def fSpecialGauss(size, sigma):

        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()

    @staticmethod
    def structuralSimilarityIndex(ref_image, impaired_image, cs_map=False):
        print('Computing SSIM ...')        
        window = Metric.fSpecialGauss(constant.SSIM_FILTER_SIZE,
                                      constant.SSIM_FILTER_SIGMA)
        C1 = (constant.SSIM_Constant_1 * constant.PIXEL_MAX) ** 2
        C2 = (constant.SSIM_Constant_2 * constant.PIXEL_MAX) ** 2

        mu1 = signal.fftconvolve(window, ref_image, mode='valid')
        mu2 = signal.fftconvolve(window, impaired_image, mode='valid')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = signal.fftconvolve(
            window, ref_image*ref_image, mode='valid') - mu1_sq
        sigma2_sq = signal.fftconvolve(
            window, impaired_image*impaired_image, mode='valid') - mu2_sq
        sigma12 = signal.fftconvolve(
            window, ref_image*impaired_image, mode='valid') - mu1_mu2

        if cs_map:
            return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)), (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # return structural_similarity(ref_image, impaired_image)
        # return ssim(ref_image, impaired_image, cs_map=False)

    @staticmethod
    def multiScaleStructuralSimilarityIndex(ref_image, impaired_image):
        print('Computing MSSSIM ...')        
        downsample_filter = np.ones((2, 2)) / 4.0
        mssim = np.array([])
        mcs = np.array([])

        tmp_ref_image = ref_image.astype(np.float64)
        tmp_impaired_image = impaired_image.astype(np.float64)
        weights = np.array([0.448, 0.2856, 0.3001, 0.2363, 0.1333])

        levels = weights.size
        for level in range(levels):
            ssim_map, cs_map = Metric.structuralSimilarityIndex(
                ref_image, impaired_image, cs_map=True)

            mssim = np.append(mssim, ssim_map.mean())
            mcs = np.append(mcs, cs_map.mean())

            filtred_ref_image = ndimage.filters.convolve(
                tmp_ref_image, downsample_filter, mode='reflect')

            filtred_impaired_image = ndimage.filters.convolve(
                tmp_impaired_image, downsample_filter, mode='reflect')

            tmp_ref_image = filtred_ref_image[::2, ::2]
            tmp_impaired_image = filtred_impaired_image[::2, ::2]
        print(np.mean(mssim))
        return (np.prod(mcs[0:levels - 1] ** weights[0:levels - 1]) * (mssim[levels - 1] ** weights[levels - 1]))

    @staticmethod
    def meanAbsoluteError(ref_image, impaired_image):
        print('Computing MAE ...')        
        return np.mean(np.abs(ref_image - impaired_image))

    @staticmethod
    def perceptualFidelityAwareMeanSquaredError(ref_image, impaired_image, rescale=True):
        print('Computing PAMSE ...')        
        emap = np.asarray(ref_image, dtype=np.float64) - \
            np.asarray(impaired_image, dtype=np.float64)

        if rescale:
            emap *= (constant.PIXEL_MAX / ref_image.max())
        sigma = 0.8
        herr = ndimage.filters.gaussian_filter(emap, sigma)

        return np.mean(herr ** 2)

    @staticmethod
    def gradientMagnitudeSimilarityDeviation(ref_image, impaired_image, rescale=True):
        print('Computing GMSD ...')        
        if rescale:
            scl = (255.0 / ref_image.max())
        else:
            scl = np.float32(1.0)

        dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
        T = 170.0
        dy = dx.T

        ukrn = np.ones((2, 2)) / 4.0

        aveY1 = signal.convolve2d(
            scl * ref_image, ukrn, mode='same', boundary='symm')
        aveY2 = signal.convolve2d(
            scl * impaired_image, ukrn, mode='same', boundary='symm')

        Y1 = aveY1[0::constant.GMSD_DOWN_STEP, 0::constant.GMSD_DOWN_STEP]
        Y2 = aveY2[0::constant.GMSD_DOWN_STEP, 0::constant.GMSD_DOWN_STEP]

        IxY1 = signal.convolve2d(Y1, dx, mode='same', boundary='symm')
        IyY1 = signal.convolve2d(Y1, dy, mode='same', boundary='symm')
        grdMap1 = np.sqrt(IxY1**2 + IyY1**2)

        IxY2 = signal.convolve2d(Y2, dx, mode='same', boundary='symm')
        IyY2 = signal.convolve2d(Y2, dy, mode='same', boundary='symm')

        grdMap2 = np.sqrt(IxY2**2 + IyY2**2)
        quality_map = (2*grdMap1*grdMap2 + T) / (grdMap1**2 + grdMap2**2 + T)
        score = np.std(quality_map)

        return score

    @staticmethod
    def visualInformationFidelityP(ref_image, impaired_image):
        print('Computing VIFp ...')        
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2**(4 - scale + 1) + 1
            sd = N / 5.0

            if (scale > 1):
                ref_image = ndimage.gaussian_filter(ref_image, sd)
                impaired_image = ndimage.gaussian_filter(impaired_image, sd)

                ref_image = ref_image[::2, ::2]
                impaired_image = impaired_image[::2, ::2]

            mu1 = ndimage.gaussian_filter(ref_image, sd)
            mu2 = ndimage.gaussian_filter(impaired_image, sd)

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = ndimage.gaussian_filter(
                ref_image * ref_image, sd) - mu1_sq
            sigma2_sq = ndimage.gaussian_filter(
                impaired_image * impaired_image, sd) - mu2_sq
            sigma12 = ndimage.gaussian_filter(
                ref_image * impaired_image, sd) - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g *
                                   sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num/den

        return vifp

    @staticmethod
    def coordinateConvertion(image):
        height, width = image.shape

        # Load 655362 sample points
        with open("sphere_655362.txt") as f:
            content = f.readlines()

        sample_points = [x.strip().split() for x in content]
        sample_points.pop(0)
        sample_points = [
            [float(x[0]) * np.pi / 180.0, float(x[1]) * np.pi / 180.0] for x in sample_points]

        # Convert spherical coordinate into Cartesian 3D coordinate
        cartesian_coord = [[math.sin(x[1]) * math.cos(x[0]), math.sin(x[0]), -
                            math.cos(x[1]) * math.cos(x[0])] for x in sample_points]

        # Convert Cartesian 3D coordinate into rectangle coordinate
        rectangle_coord = [[width * (0.5 + math.atan2(x[0], x[2]) / (np.pi*2)),
                            height * (math.acos(x[1]) / np.pi)] for x in cartesian_coord]

        return height, width, rectangle_coord

    @staticmethod
    def sphericalPeakToSignalNoiseRatio(ref_image, impaired_image, color):
        print('Computing SPSNR ...')        
        height, width, rect_coord = Metric.coordinateConvertion(ref_image)

        mse = 0
        if color == True:
            for pt in rect_coord:
                pt[0] = int(np.sinc(pt[0]) *
                            np.sinc(pt[0]/2))  # Lanczos-2
                pt[1] = int(np.sinc(pt[1]) *
                            np.sinc(pt[1]/2))  # Lanczos-2

                pt[0] = width - 1 if pt[0] >= width else pt[0]  # Longtidue
                pt[1] = height - 1 if pt[1] >= height else pt[1]  # Latitude

                mse += (int(ref_image[pt[1]][pt[0]]) -
                        int(impaired_image[pt[1]][pt[0]]))**2
        else:
            for pt in rect_coord:
                pt[0] = int(np.sinc(pt[0]) *
                            np.sinc(pt[0]/3))  # Lanczos-3
                pt[1] = int(np.sinc(pt[1]) *
                            np.sinc(pt[1]/3))  # Lanczos-3

                pt[0] = width-1 if pt[0] >= width else pt[0]  # Longtidue
                pt[1] = height-1 if pt[1] >= height else pt[1]  # Latitude

                mse += (int(ref_image[pt[1]][pt[0]]) -
                        int(impaired_image[pt[1]][pt[0]]))**2

        mse = (mse / (width * height))
        if mse == 0:
            return 100
        return 10 * np.log10((constant.PIXEL_MAX**2 / mse))

    @staticmethod
    def sphericalPeakToSignalNoiseRatioNN(ref_image, impaired_image):
        print('Computing SPSNRNN ...')        
        height, width, rect_coord = Metric.coordinateConvertion(ref_image)

        # Calculate S_PSNR_NN
        mse = 0
        for pt in rect_coord:
            pt[0], pt[1] = int(round(pt[0])), int(
                round(pt[1]))  # Nearest Neighbor
            pt[0] = width-1 if pt[0] >= width else pt[0]  # Longtidue
            pt[1] = height-1 if pt[1] >= height else pt[1]  # Latitude
            mse += (int(ref_image[pt[1]][pt[0]]) -
                    int(impaired_image[pt[1]][pt[0]]))**2

        mse = (mse/(width*height))
        if mse == 0:
            return 100
        return 10 * np.log10((constant.PIXEL_MAX**2 / mse))

    @staticmethod
    def laguerreGaussCircularHarmonic30(size, sigma):
        x = np.linspace(-size/2.0, size/2.0, size)
        y = np.linspace(-size/2.0, size/2.0, size)
        xx, yy = np.meshgrid(x, y)

        r = np.sqrt(xx * xx + yy * yy)
        gamma = np.arctan2(yy, xx)
        l30 = - (1/6.0) * (1 / (sigma * np.sqrt(np.pi))) * np.exp(-r*r / (2*sigma*sigma)
                                                                  ) * (np.sqrt(r*r/(sigma*sigma)) ** 3) * np.exp(-1j * 3 * gamma)

        return l30

    @staticmethod
    def laguerreGaussCircularHarmonic10(size, sigma):
        x = np.linspace(-size/2.0, size/2.0, size)
        y = np.linspace(-size/2.0, size/2.0, size)
        xx, yy = np.meshgrid(x, y)

        r = np.sqrt(xx*xx + yy*yy)
        gamma = np.arctan2(yy, xx)
        l10 = - (1 / (sigma * np.sqrt(np.pi))) * np.exp(-r*r / (2*sigma *
                                                                sigma)) * np.sqrt(r*r/(sigma*sigma)) * np.exp(-1j * gamma)

        return l10

    @staticmethod
    def edgeCoherence(image):
        l10 = Metric.laguerreGaussCircularHarmonic10(17, 2)
        l30 = Metric.laguerreGaussCircularHarmonic30(17, 2)

        y10 = ndimage.filters.convolve(image, np.real(
            l10)) + 1j * ndimage.filters.convolve(image, np.imag(l10))
        y30 = ndimage.filters.convolve(image, np.real(
            l30)) + 1j * ndimage.filters.convolve(image, np.imag(l30))

        return np.sum(- (np.absolute(y30) * np.absolute(y10)) * np.cos(np.angle(y30) - 3 * np.angle(y10)))

    @staticmethod
    def relativeEdgeCoherence(ref_image, impaired_image):
        # return (Metric.edgeCoherence(ref_image) + 1) / (Metric.edgeCoherence(impaired_image) + 1)
        return Metric.edgeCoherence(impaired_image) / Metric.edgeCoherence(ref_image)
