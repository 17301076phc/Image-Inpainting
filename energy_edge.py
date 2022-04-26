# Image Completion using Statistics of Patch Offsets
# Author: Pranshu Gupta and Shrija Mishra

import cv2, numpy as np, sys, math, operator, maxflow, random, config as cfg
from scipy import ndimage
from time import time
from itertools import count, combinations
from skimage.metrics import structural_similarity

from scipy.spatial import minkowski_distance_p


class Optimizer(object):
    def __init__(self, image, edgemap, mask, offset):
        self.image = image/100.0
        self.edgemap = edgemap/100.0
        self.mask_edgemap = None
        self.mask = mask
        self.offsets = offset
        # x, y = np.where(self.mask != 0)
        # sites = [(i, j) for (i, j) in zip(x, y)]
        # self.sites = sites  # 缺失位置坐标（也表示缺失位置的edge图）
        self.sites = []
        self.location = np.where(self.mask != 0)
        self.box = []
        self.neighbors = []
        # self.InitializeD()  # 初始化label 和 mask
        self.InitializeMaskPatch()
        # print(self.sites)
        self.InitializeNeighbors()  # 初始化sites邻居
        print(len(self.neighbors))

    # def get_mask_edgemape(self):
    #     mask_edgemap = cv2.imread("./gdata/test_set/edge/mask_edgemap0.png")
    #     self.mask_edgemap = mask_edgemap

    def InitializeMaskPatch(self):
        box = np.min(self.location[0]), np.max(self.location[0]), np.min(self.location[1]), np.max(self.location[1])
        print(box)
        self.box = box
        p = cfg.PATCH_SIZE // 2
        i_left = (box[1]-box[0])%p
        j_left = (box[3]-box[2])%p
        for i in range(box[0], box[1]-i_left+p, p):
            for j in range(box[2], box[3]-j_left+p,p):
                self.sites.append([i,j])

    def InitializeNeighbors(self):
        start = time()
        # print(self.sites)
        # site_dic = dict(zip(self.sites, list(range(len(self.sites)))))
        for i in range(len(self.sites)):
            ne = []
            neighbors = self.GetNeighbors(self.sites[i])
            for n in neighbors:
                if n in self.sites:
                # if n in site_dic:
                    ne.append(self.sites.index(n))  # 添加sites的索引位置
            self.neighbors.append(ne)
        end = time()
        print("InitializeNeighbors execution time: ", end - start)

    # 初始化距离data term
    def D(self, site, offset):
        i, j = site[0] + offset[0], site[1] + offset[1]
        p = cfg.PATCH_SIZE // 2
        try:
            # if self.mask[i][j] == 0: # 有效,即已知区域
            if j not in range(self.box[2]-p, self.box[3]+p ) and i not in range(self.box[0]-p , self.box[1]+p):
            # if self.mask[i+p][j] == 0 and self.mask[i-p][j] == 0 and self.mask[i][j-p] == 0 and self.mask[i][j+p] == 0: # 有效,即已知区域
                edge_patch0 = self.edgemap[site[0] - p:site[0] + p, site[1] - p:site[1] + p]
                edge_patch1 = self.edgemap[i - p:i + p, j - p:j + p]
                # edge_patch1 = self.boundpatch(i,j,self.edgemap)

                # return np.sum((edge_patch0-edge_patch1)**2)
                return np.mean(np.square(edge_patch0-edge_patch1))
                # if np.mean(np.square(edge_patch0-edge_patch1)) < 0.1:
                #     return 0

            # return float('inf')
            return 1000000.0
        except:
            return 1000000.0



    def V(self, site1, site2, alpha, beta):
        x1a, y1a = site1[0] + alpha[0], site1[1] + alpha[1]
        x2a, y2a = site2[0] + alpha[0], site2[1] + alpha[1]
        x1b, y1b = site1[0] + beta[0], site1[1] + beta[1]
        x2b, y2b = site2[0] + beta[0], site2[1] + beta[1]
        p = cfg.PATCH_SIZE // 2
        try:
            # mask中黑色已知部分
            # if self.mask[x1a, y1a] == 0 and self.mask[x1b, y1b] == 0 and self.mask[x2a, y2a] == 0 \
            #         and self.mask[x2a, y2a] == 0:
                # 计算偏移量
                # return np.sum((self.image[x1a, y1a] - self.image[x1b, y1b]) ** 2) + np.sum(
                #     (self.image[x2a, y2a] - self.image[x2b, y2b]) ** 2) + \
            if x1a not in range(self.box[0] - p, self.box[1] + p) and y1a not in range(self.box[2] - p, self.box[3] + p) and \
                x1b not in range(self.box[0] - p, self.box[1] + p) and y1b not in range(self.box[2] - p, self.box[3] + p) and \
                x2a not in range(self.box[0] - p, self.box[1] + p) and y2a not in range(self.box[2] - p, self.box[3] + p) and \
                x2b not in range(self.box[0] - p, self.box[1] + p) and y2b not in range(self.box[2] - p, self.box[3] + p):

                p0 = self.image[x1a - p:x1a + p, y1a - p:y1a + p]
                p1 = self.image[x1b - p:x1b + p, y1b - p:y1b + p]
                p2 = self.image[x2a - p:x2a + p, y2a - p:y2a + p]
                p3 = self.image[x2b - p:x2b + p, y2b - p:y2b + p]


                return np.sum((p0-p1)**2)+np.sum((p2-p3)**2)
                
            return 1000000.0
        except:
            return 1000000.0

    def AEV(self, site1, site2, alpha, beta):
        x1a, y1a = site1[0] + alpha[0], site1[1] + alpha[1]
        x2a, y2a = site2[0] + alpha[0], site2[1] + alpha[1]
        x1b, y1b = site1[0] + beta[0], site1[1] + beta[1]
        x2b, y2b = site2[0] + beta[0], site2[1] + beta[1]
        p = cfg.PATCH_SIZE // 2
        ml = 0
        try:
            # mask中黑色已知部分
            if x1a not in range(self.box[0] -p, self.box[1]+p ) and y1a not in range(self.box[2] -p, self.box[3]+p ) and \
                x1b not in range(self.box[0]-p, self.box[1]+p ) and y1b not in range(self.box[2]-p, self.box[3]+p ) and \
                x2a not in range(self.box[0] -p, self.box[1]+p ) and y2a not in range(self.box[2]-p , self.box[3]+p ) and \
                x2b not in range(self.box[0] -p, self.box[1]+p ) and y2b not in range(self.box[2]-p, self.box[3]+p):

            # if self.mask[x1a-p][y1a] == 0 and self.mask[x1a+p][y1a] == 0 and self.mask[x1a][y1a-p] == 0 and self.mask[x1a][y1a+p] == 0 \
            #     and self.mask[x2a-p][y2a] == 0 and self.mask[x2a+p][y2a] == 0 and self.mask[x2a][y2a-p] == 0 and self.mask[x2a][y2a+p] == 0 \
            #     and self.mask[x1b-p][y1b] == 0 and self.mask[x1b+p][y1b] == 0 and self.mask[x1b][y1b-p] == 0 and self.mask[x1b][y1b+p] == 0\
            #     and self.mask[x2b-p][y2b] == 0 and self.mask[x2b+p][y2b] == 0 and self.mask[x2b][y2b-p] == 0 and self.mask[x2b][y2b+p] == 0 :
            #
                p0 = self.image[x1a - p:x1a + p, y1a - p:y1a + p]
                p1 = self.image[x1b - p:x1b + p, y1b - p:y1b + p]
                p2 = self.image[x2a - p:x2a + p, y2a - p:y2a + p]
                p3 = self.image[x2b - p:x2b + p, y2b - p:y2b + p]

            
                if site1[0] == site2[0] and site1[1] - site2[1] == p:
                    return np.sum((p0[:, :p+ml] - p3[:, p-ml:]) ** 2) + np.sum((p2[:, p-ml:] - p1[:, :p+ml]) ** 2)
                    # return np.mean(np.square(p0[:, :p] - p3[:, p:])) + np.mean(np.square(p2[:, p:] - p1[:, :p]))
                if site1[0] == site2[0] and site2[1] - site1[1] == p:
                    return np.sum((p0[:, p-ml:] - p3[:, :p+ml]) ** 2) + np.sum((p2[:, :p+ml] - p1[:, p-ml:]) ** 2)
                    # return np.mean(np.square(p0[:, p:] - p3[:, :p])) + np.mean(np.square(p2[:, :p] - p1[:, p:]))
                if site1[1] == site2[1] and site1[0] - site2[0] == p:
                    return np.sum((p0[:p+ml, :] - p3[p-ml:, :]) ** 2) + np.sum((p2[p-ml:, :] - p1[:p+ml, :]) ** 2)
                    # return np.mean(np.square(p0[:p, :] - p3[p:, :])) + np.mean(np.square(p2[p:, :] - p1[:p, :]))
                if site1[1] == site2[1] and site2[0] - site1[0] == p:
                    return np.sum((p0[p-ml:, :] - p3[:p+ml, :]) ** 2) + np.sum((p2[:p+ml, :] - p1[p-ml:, :]) ** 2)
                    # return np.mean(np.square(p0[p:, :] - p3[:p, :])) + np.mean(np.square(p2[:p, :] - p1[p:, :]))


            return 1000000.0
        except:
            return 1000000.0

    def IsLowerEnergy(self, nodes, labelling1, labelling2):
        updatedNodes = np.where(labelling1 != labelling2)[0]
        diff = 0.0
        data_term = 0.0
        v_term = 0.0

        for node in updatedNodes:
            # node_d1 = self.D(self.sites[node], self.offsets[labelling1[node]])
            # node_d2 = self.D(self.sites[node], self.offsets[labelling2[node]])
            #
            # data_term += (node_d2 - node_d1)

            if self.D(self.sites[node], self.offsets[labelling2[node]]) < 1000000.0:
                node_d1 = self.D(self.sites[node], self.offsets[labelling1[node]])
                node_d2 = self.D(self.sites[node], self.offsets[labelling2[node]])
                # print(node_d1,node_d2)
                data_term += (node_d2 - node_d1)*50
            # if tmp<0:
            #     diff += tmp*10
                for n in self.neighbors[node]:# node处的邻居节点
                    if n in updatedNodes: # n为site中索引位置
                        if n > node:
                            v1 = self.AEV(self.sites[node], self.sites[n],self.offsets[labelling1[node]], self.offsets[labelling1[n]])
                            v2 = self.AEV(self.sites[node], self.sites[n], self.offsets[labelling2[node]], self.offsets[labelling2[n]])
                            # print(self.offsets[labelling1[node]] == self.offsets[labelling1[n]])
                            v_term += (v2-v1)
                    else: # 用相邻节点更新
                        v1 = self.AEV(self.sites[node], self.sites[n], self.offsets[labelling1[node]],
                                      self.offsets[labelling1[n]])
                        v2 = self.AEV(self.sites[node], self.sites[n], self.offsets[labelling2[node]],
                                      self.offsets[labelling2[n]])

                        v_term += (v2-v1)
            else:
                return False
        if data_term + v_term < 0:
            return True
        return False

    def GetNeighbors(self, site):
        p = cfg.PATCH_SIZE//2
        return [[site[0]-p, site[1]], [site[0], site[1]-p],
                [site[0]+p, site[1]], [site[0], site[1]+p]]

        # return [(site[0] - 1, site[1]), (site[0], site[1] - 1), (site[0] + 1, site[1]), (site[0], site[1] + 1)]

    def AreNeighbors(self, site1, site2):
        if np.abs(site1[0] - site2[0]) <= cfg.PATCH_SIZE//2 and np.abs(site1[1] - site2[1]) <=cfg.PATCH_SIZE//2:
            return True
        return False

    def InitializeLabelling(self):
        start = time()
        labelling = [None] * len(self.sites)
        l = len(self.offsets)
        for i in range(len(self.sites)):
            perm = np.random.permutation(l)  # 随机排列n个序列
            labelling[i] = 0
            # a = zip(list(range(len(self.offsets))),list(range(len(self.offsets))))
            # perm = dict(a)
            for j in perm:
            # for j in range(l):
                if self.D(self.sites[i], self.offsets[j]) < 1000000.0:
                    labelling[i] = j
                    break
        # 给每个空白区域的site 随机分配offset
        self.sites = [self.sites[i] for i in range(len(self.sites)) if labelling[i] != None]
        labelling = [label for label in labelling if label != None]
        end = time()
        print("InitializeLabelling execution time: ", end - start)
        # print("InitializeLabelling finished! ")
        return self.sites, np.array(labelling)

    def CreateGraphABS(self, alpha, beta, ps, labelling):
        start = time()
        v = len(ps)
        g = maxflow.Graph[float](v, 3*v)
        nodes = g.add_nodes(v)
        for i in range(v):
            # add the data terms here
            ta, tb = self.D(self.sites[ps[i]], self.offsets[alpha]), self.D(self.sites[ps[i]], self.offsets[beta])
            # add the smoothing terms here
            neighbor_list = self.neighbors[ps[i]]
            for ind in neighbor_list:
                try:
                    a, b, j = labelling[ps[i]], labelling[ind], ps.index(ind)
                    if j > i and (b == alpha or b == beta):
                        epq = self.AEV(self.sites[ps[i]], self.sites[ps[j]], self.offsets[alpha], self.offsets[beta])
                        g.add_edge(nodes[i], nodes[j], epq, epq)
                    else:
                        ea = self.AEV(self.sites[ps[i]], self.sites[ps[j]], self.offsets[alpha], self.offsets[b])
                        eb = self.AEV(self.sites[ps[i]], self.sites[ps[j]], self.offsets[beta], self.offsets[b])
                        ta, tb = ta + ea, tb + eb
                except Exception as e:
                    pass
            g.add_tedge(nodes[i], ta, tb)
        end = time()
        #print "CreateGraph execution time: ", end - start
        return g, nodes

    def OptimizeLabellingABS(self, labelling):
        labellings = np.zeros((2, len(self.sites)), dtype=int)
        labellings[0] = labellings[1] = np.copy(labelling)
        iter_count = 0
        while(True):
            start = time()
            success = 0
            for alpha, beta in combinations(range(len(self.offsets)), 2):
                ps = [i for i in range(len(self.sites)) if (labellings[0][i] == alpha or labellings[0][i] == beta)]
                if len(ps) > 0:
                    g, nodes = self.CreateGraphABS(alpha, beta, ps, labellings[0])
                    flow = g.maxflow()
                    for i in range(len(ps)):
                        gamma = g.get_segment(nodes[i])
                        labellings[1, ps[i]] = alpha*(1-gamma) + beta*gamma
                    if self.IsLowerEnergy(ps, labellings[0], labellings[1]):
                        labellings[0, ps] = labellings[1, ps]
                        success = 1
                    else:
                        labellings[1, ps] = labellings[0, ps]
            iter_count += 1
            end = time()
            print( "ABS Iteration " + str(iter_count) + " execution time: ", str(end - start)  )
            if success != 1 or iter_count >= cfg.MAX_ITER:
                break
        return labellings[0]
