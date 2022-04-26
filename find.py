"""
Statistics of Patch Offsets for Image Completion - Kaiming He and Jian Sun
A Python Implementation - Pranshu Gupta and Shrija Mishra
"""
import os
import random
from collections import defaultdict

import cv2
import sys

from PIL import Image
import matplotlib.pyplot as plt
from skimage import feature

import energy_edge
import plot
import kdtree
import energy
import operator
import numpy as np
import config as cfg
from time import time
from scipy import ndimage
from sklearn.decomposition import PCA


# import matplotlib.pyplot as plot

def GetBoundingBox(mask):
    """
    Get Bounding Box for a Binary Mask
    Arguments: mask - a binary mask
    Returns: col_min, col_max, row_min, row_max
    """
    start = time()
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    if cfg.PRINT_BB_IMAGE:
        cv2.rectangle(mask, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255, 255, 255), 1)
        cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + cfg.BB_IMAGE_SUFFIX, mask)
    end = time()
    print("GetBoundingBox execution time: ", end - start)
    return bbox


def GetSearchDomain(shape, bbox):
    """
    get a rectangle that is 3 times larger (in length) than the bounding box of the hole
    this is the region which will be used for the extracting the patches
    """
    # shape[0] h   shape[1] w
    start = time()
    p = cfg.PATCH_SIZE//2
    col_min, col_max = max(0, 2 * bbox[0] - bbox[1]), min(2 * bbox[1] - bbox[0], shape[1] - 1)
    row_min, row_max = max(0, 2 * bbox[2] - bbox[3]), min(2 * bbox[3] - bbox[2], shape[0] - 1)

    # col_min, col_max = 0, shape[1]-p
    # row_min, row_max = 0, shape[0]-p
    end = time()
    print("GetSearchDomain execution time: ", end - start)
    return col_min, col_max, row_min, row_max


def GetPatches(image,bbox, hole):
    """
    get the patches from the search region in the input image
    """
    start = time()
    print(bbox)
    indices, patches = [],[]
    rows, cols, _ = image.shape
    # for i in range(bbox[0] + cfg.PATCH_SIZE // 2, bbox[1] - cfg.PATCH_SIZE // 2):
    #     for j in range(bbox[2] + cfg.PATCH_SIZE // 2, bbox[3] - cfg.PATCH_SIZE // 2):
    #         if j not in range(hole[2] - cfg.PATCH_SIZE // 2, hole[3] + cfg.PATCH_SIZE // 2) and i not in range(
    #                 hole[0] - cfg.PATCH_SIZE // 2, hole[1] + cfg.PATCH_SIZE // 2):
    #             indices.append([i, j])
    #             patches.append(image[i - cfg.PATCH_SIZE // 2:i + cfg.PATCH_SIZE // 2,
    #                            j - cfg.PATCH_SIZE // 2:j + cfg.PATCH_SIZE // 2].flatten())
    # end = time()
    p = 8
    for i in range(bbox[2] + p // 2, bbox[3] - p // 2):
        for j in range(bbox[0] + p // 2, bbox[1] - p // 2):
            if i not in range(hole[2] - p // 2, hole[3] + p // 2) and j not in range(
                    hole[0] - p // 2, hole[1] + p // 2):
                indices.append([i, j])
                patches.append(image[i - p // 2:i + p // 2,
                               j - p // 2:j + p // 2].flatten())
    end = time()
    print("GetPatches execution time: ", end - start)
    # plot.figure()
    # plot.imshow(patches)
    # plot.show()
    return np.array(indices), np.array(patches, dtype='int64')


def ReduceDimension(patches):  # 通过主成分分析进行降维
    start = time()
    pca = PCA(n_components=cfg.PCA_COMPONENTS)
    reducedPatches = pca.fit_transform(patches)
    end = time()
    print("ReduceDimension execution time: ", end - start)
    return reducedPatches


def GetOffsets(patches, indices):
    start = time()
    kd = kdtree.KDTree(patches, leafsize=cfg.KDT_LEAF_SIZE, tau=cfg.TAU)  # 快速邻近查找
    print(kd)
    dist, offsets = kdtree.get_annf_offsets(patches, indices, kd.tree, cfg.TAU)
    end = time()
    print("GetOffsets execution time: ", end - start)
    return offsets


def GetKDominantOffsets(offsets, K):
    start = time()
    x, y = [offset[0] for offset in offsets if offset != None], [offset[1] for offset in offsets if offset != None]
    bins = [[i for i in range(np.min(x), np.max(x))], [i for i in range(np.min(y), np.max(y))]]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = hist.T
    # plot.PlotHistogram2D(hist, xedges, yedges)
    p, q = np.where(hist == cv2.dilate(hist, np.ones(8)))  # Non Maximal Suppression
    nonMaxSuppressedHist = np.zeros(hist.shape)
    nonMaxSuppressedHist[p, q] = hist[p, q]
    # plot.PlotHistogram2D(nonMaxSuppressedHist, xedges, yedges)
    p, q = np.where(nonMaxSuppressedHist >= np.partition(nonMaxSuppressedHist.flatten(), -K)[-K])
    peakHist = np.zeros(hist.shape)
    peakHist[p, q] = nonMaxSuppressedHist[p, q]
    # plot.PlotHistogram2D(peakHist, xedges, yedges)
    peakOffsets, freq = [[xedges[j], yedges[i]] for (i, j) in zip(p, q)], nonMaxSuppressedHist[p, q].flatten()
    peakOffsets = np.array([x for _, x in sorted(zip(freq, peakOffsets), reverse=True)], dtype="int64")[:2 * K]
    end = time()
    print( "GetKDominantOffsets execution time: ", end - start)
    return peakOffsets


def GetOptimizedLabels(imageR, mask, labels):
    # mask0 = (1 - mask / 255).astype(np.bool)
    # edgemap = feature.canny(image,sigma=2,mask=mask0)
    edgemap = cv2.imread("./gdata/test_set/edge/edge1024.png",cv2.IMREAD_GRAYSCALE)
    start = time()
    # optimizer = energy.Optimizer(image, mask, labels)
    optimizer = energy_edge.Optimizer(imageR,edgemap,mask,offset=labels)

    sites, optimalLabels = optimizer.InitializeLabelling()
    print(len(sites))
    # print(optimalLabels)
    optimalLabels = optimizer.OptimizeLabellingABS(optimalLabels)

    end = time()
    print("GetOptimizedLabels execution time: ", end - start)
    return sites, optimalLabels


def CompleteImage(image, sites, mask, offsets, optimalLabels):
    # failedPoints = False
    h,w = image.shape[0],image.shape[1]
    completedPoints = np.zeros(image.shape)
    finalImg = image
    p = cfg.PATCH_SIZE // 2
    for i in range(len(sites)):
        j = optimalLabels[i]
        m,n = sites[i][0] + offsets[j][0],sites[i][1] + offsets[j][1]
        # print(sites[i][0] + offsets[j][0],sites[i][1] + offsets[j][1])
        # if m in range(p,h-p) and n in range(p,w-p):
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[m -p:m + p,n - p:n + p]
        # elif n+p >w and n -p<w and m+p >h  and m-p<h:
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[h-2*p:, w-2*p:]
        # elif n+p >w and n -p<w and m - p < 0 and m+p>0:
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[:2*p, w - p * 2:]
        # elif m+p >h and m-p<h and n - p < 0 and n+p>0:
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[h-2*p:, :2*p]
        # elif n + p > w and n -p<w and m-p>0 and m+p <h:
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[m - p:m + p, w-p*2:]
        # elif m + p > h and m-p<h and n-p >0 and n+p <w:
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p]= image[h-p*2:, n - p:n + p]
        # elif m-p<0 and m+p>0 and n-p >0 and n+p <w:
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[:2*p, n - p:n + p]
        # elif n-p<0 and n+p>0 and m-p >0 and m+p <h:
        #     finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[m-p:m+p, :2*p]

        finalImg[sites[i][0] - p:sites[i][0] + p, sites[i][1] - p:sites[i][1] + p] = image[m - p:m + p, n - p:n + p]

        completedPoints[sites[i][0], sites[i][1]] = finalImg[sites[i][0], sites[i][1]]
    return finalImg, completedPoints

def main(image, imageR, mask):
    """
    Image Completion Pipeline
        1. Patch Extraction
        2. Patch Offsets
        3. Image Stacking
        4. Blending
    """
    # image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    # imageR = cv2.imread(imageFile)
    # mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
    # mask0 = (1 - mask / 255).astype(np.bool)
    # edgemap = feature.canny(image, sigma=2, mask=mask0)

    bb = GetBoundingBox(mask)
    print(bb)
    bbwidth = bb[3] - bb[2]
    bbheight = bb[1] - bb[0]
    cfg.TAU = max(bbwidth, bbheight) / 15
    # cfg.TAU = 32
    cfg.DEFLAT_FACTOR = image.shape[1]
    sd = GetSearchDomain(image.shape, bb)
    print(sd)
    indices, patches = GetPatches(imageR, sd, bb)  # 在已知区域中寻找patch
    # print(indices.shape)
    print(patches.shape)
    reducedPatches = ReduceDimension(patches)
    offsets = GetOffsets(reducedPatches, indices)
    offsets = GetKDominantOffsets(offsets,60)
    # offsets = findEdgeOffsets(offsets)
    print(len(offsets))

    sites, optimalLabels = GetOptimizedLabels(imageR, mask, offsets)
    print(len(optimalLabels))
    #
    # imageO = cv2.imread("./gdata/test_set/degraded_img/shengshanjuan2_1024_degrade.jpg")
    completedImage,completedPoints = CompleteImage(imageR, sites, mask, offsets, optimalLabels)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "Complete.jpg", completedImage)
    cv2.imwrite(cfg.OUT_FOLDER + cfg.IMAGE + "_CompletePoints.png", completedPoints)

    print("Finished !!!")
    # if (np.sum(failedPoints)):
    #     print("failed !! ")


mask = cv2.imread('./gdata/test_set/mask/shenshanjuan2.png', cv2.IMREAD_GRAYSCALE)
imageR = cv2.imread("./gdata/test_set/degraded_img/shengshanjuan2_1024_degrade.jpg")
image = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
# image = cv2.imread("./gdata/test_set/degraded_img/shengshanjuan2_1024_degrade.jpg",cv2.IMREAD_GRAYSCALE)

main(image, imageR, mask)

# for file in os.listdir("landscape_painting"):
#     print(file)
#     gt_img = cv2.imread(os.path.join("landscape_painting", file))
#     # cv2.imshow("gt",gt_img)
#     # cv2.waitKey(0)
#
#     imageR = cv2.add(gt_img, mask)
#     image = cv2.cvtColor(imageR,cv2.COLOR_BGR2GRAY)
#     main(image,imageR, mask)
