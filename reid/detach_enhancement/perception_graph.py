import numpy as np
import torch
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sci
import os.path as osp
import time

from reid.evaluation.re_rank import re_ranking


class HG(object):
    eps=0.1
    def __init__(self, lamda, features, real_labels, cams):
        self.features = features
        self.real_labels = real_labels
        self.lamda = lamda
        self.cams = cams
        self.pd_labels = None
        self.neg_ratio = None
        self.cam_2_imgs = []
        self.check_graph = np.zeros([self.real_labels.size,
                                     self.real_labels.size], dtype=self.features.dtype)
        for j, p in enumerate(self.real_labels):
            index = np.where(self.real_labels == p)
            self.check_graph[j, index] = 1.

        # hyper parameter
        self.general_graph = False
        self.homo_ap = False

    def heter_cam_normalization(self, k1_graph):
        for i in range(len(k1_graph)):
            index = np.where(k1_graph[i] != 0.)
            weights = k1_graph[i][index]
            cd_c = self.cams[index]
            tag_c_set = set(cd_c)
            for c in tag_c_set:
                c_index = np.where(cd_c == c)
                w = weights[c_index]

                w = len(w) / len(cd_c) * w / np.sum(w)  # 1/len(w)
                k1_graph[i][index[0][c_index]] = w
        print(np.sum(k1_graph, axis=1))
        print('heter_cam_normalization')
        return k1_graph

    def row_normalization(self, sim, exp=False):
        if exp:
            return np.exp(sim) / np.sum(np.exp(sim), axis=1)[:, np.newaxis]
        else:
            # todo try np.max
            return sim / np.sum(sim, 1)[:, np.newaxis]


    def old_delta_propagation(self, ks=2, kd=4, k2=11):
        print("gause kernel")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)

        # hyper parameter
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            k1_graph = self.heteo_kneighbors(sim, ks, kd)
        if self.homo_ap:
            print("----homo-ap----")
            k1_graph = self.row_normalization(k1_graph, exp=False)
        else:
            k1_graph = self.heter_cam_normalization(k1_graph)

        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # context information
        sim = sim.numpy()
        sim = self.get_gause_sim(sim, delta)
        k2_graph = self.kneighbors(sim, k2)
        graph_target = self.split_as_camera(k2_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target
    
    def old_tracklet_propagation(self, ks=2, kd=4, k2=11):
        print("gause kernel")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)

        # hyper parameter
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            cams = self.cams[:, np.newaxis]
            flag = (cams.T == cams)
            sim_same_camera = sim * flag
            real_labels = self.real_labels[:, np.newaxis]
            labels_flag = (real_labels.T == real_labels)
            print(labels_flag)
            label_same_sim = sim_same_camera*labels_flag
            flag = 1-flag
            sim_diff_camera = sim * flag
            k_diff_sim = self.kneighbors(sim_diff_camera, kd)
            k1_graph = k_diff_sim + label_same_sim

        if self.homo_ap:
            print("----homo-ap----")
            k1_graph = self.row_normalization(k1_graph, exp=False)
        else:
            k1_graph = self.heter_cam_normalization(k1_graph)
            
        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # context information
        sim = sim.numpy()
        sim = self.get_gause_sim(sim, delta)
        k2_graph = self.kneighbors(sim, k2)
        graph_target = self.split_as_camera(k2_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target


    def only_graph(self, ks=2, kd=4, k2=11):
        print("gause kernel")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            k1_graph = self.heteo_kneighbors(sim, ks, kd)
        graph_target = self.split_as_camera(k1_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target

    def old_propagation(self, ks=2, kd=4, k2=11):
        print("gause kernel")
        delta = 1.
        sim = self.get_gause_sim(self.features, delta)

        # hyper parameter
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            k1_graph = self.heteo_kneighbors(sim, ks, kd)
        if self.homo_ap:
            k1_graph = self.row_normalization(k1_graph, exp=False)
        else:
            k1_graph = self.heter_cam_normalization(k1_graph)

        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # context information
        sim = sim.numpy()
        sim = self.get_gause_sim(sim, delta)
        k2_graph = self.kneighbors(sim, k2)
        graph_target = self.split_as_camera(k2_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target


    def old_cos_propagation(self, ks=2, kd=4, k2=11):
        print("cosine sim")

        sim = self.get_cossim(self.features)


        # hyper parameter
        if self.general_graph:
            print("----general graph----")
            k1_graph = self.kneighbors(sim, ks + kd)

        else:
            k1_graph = self.heteo_kneighbors(sim, ks, kd)
        if self.homo_ap:
            k1_graph = self.row_normalization(k1_graph, exp=False)
        else:
            k1_graph = self.heter_cam_normalization(k1_graph)


        # propagation
        k1_graph = torch.Tensor(k1_graph)
        I = torch.eye(k1_graph.shape[0])
        S = torch.inverse(I - self.lamda * k1_graph)
        del I
        sim = torch.mm(S, k1_graph)
        del S, k1_graph

        # context information
        sim = sim.numpy()
        sim = self.get_cossim(sim)
        k2_graph = self.kneighbors(sim, k2)
        graph_target = self.split_as_camera(k2_graph)
        graph_target[np.diag_indices_from(graph_target)] = 1.
        return graph_target

    def split_as_camera(self, graph_result):

        for c in range(np.max(self.cams)+1):
            if c == 0:
                result_reconstruct = np.squeeze(graph_result
                                                [:, np.where(self.cams == c)])
            else:
                result_reconstruct = np.concatenate(
                    (result_reconstruct,
                     np.squeeze(graph_result[:, np.where(self.cams == c)])
                     ),
                    axis=1
                )
        return result_reconstruct

    def heteo_kneighbors(self, sim_matrix, ks, kd):
        cams = self.cams[:, np.newaxis]
        flag = (cams.T == cams)
        # inside use 0 to pad the empty
        sim_same_camera = sim_matrix * flag
        flag = 1-flag
        sim_diff_camera = sim_matrix * flag
        k_same_sim = self.kneighbors(sim_same_camera, ks)
        k_diff_sim = self.kneighbors(sim_diff_camera, kd)
        return k_diff_sim+k_same_sim

    def kneighbors(self, sim_matrix, knn, unifyLabel=None):
        k_sim = np.zeros_like(sim_matrix)
        argpart = np.argpartition(-sim_matrix, knn)  # big move before knn
        row_index = np.arange(sim_matrix.shape[0])[:, None]
        if unifyLabel:
            k_sim[row_index, argpart[:, 0:knn]] = unifyLabel
        else:
            k_sim[row_index, argpart[:, 0:knn]] = sim_matrix[row_index, argpart[:, 0:knn]]
        return k_sim

    def get_gause_sim(self, features, delta=1.):
        distance = euclidean_distances(features, squared=True)
        distance /= 2 *delta**2
        sim = np.exp(-distance)
        return sim

    def get_cossim(self, features, temp=1.):
        return cosine_similarity(features)

    # def k_means(self, n_clusters):
    #     y_pred = KMeans(n_clusters, n_jobs=8).fit_predict(self.features)
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
    #
    # def ahc(self, n_clusters):
    #     dist = euclidean_distances(self.features)
    #     y_pred = AgglomerativeClustering(n_clusters, affinity="precomputed", linkage="average").fit_predict(dist)
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
    #
    # def sp(self, n_clusters):
    #     y_pred = SpectralClustering(n_clusters).fit_predict(self.features)
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
    #
    # def dbscan(self, epoch):
    #     rerank_dist = re_ranking(self.features)
    #     if epoch ==0:
    #         tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
    #         tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    #         tri_mat = np.sort(tri_mat, axis=None)
    #         top_num = np.round(1.6e-3 * tri_mat.size).astype(int)
    #         self.eps = tri_mat[:top_num].mean()
    #         print('eps in cluster: {:.3f}'.format(self.eps))
    #
    #     eps=self.eps
    #     y_pred = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8).fit_predict(rerank_dist)
    #
    #     max_label = np.max(y_pred)
    #     print(max_label)
    #     # return y_pred
    #     # take care that may has -1
    #     for index in np.where(y_pred == -1)[0]:
    #         y_pred[index] = max_label + 1
    #         max_label += 1
    #     print("new label: ", max_label, np.max(y_pred))
    #
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #
    #     return y_pred
    #
    # def ap(self):
    #     dist = self.kneighbors(euclidean_distances(self.features), 50)
    #     y_pred = AffinityPropagation(preference=np.median(dist)).fit_predict(self.features)
    #
    #     max_label = np.max(y_pred)
    #     for index in np.where(y_pred == -1):
    #         y_pred[index] = max_label + 1
    #         max_label += 1
    #     print(max_label)
    #     y_pred = np.asarray([y_pred])
    #     y_pred = (y_pred == y_pred.T).astype(np.int)
    #     y_pred = self.split_as_camera(y_pred)
    #     print("y_pred", y_pred)
    #     return y_pred
