import torch.nn as nn
import torch.nn.functional as F
import torch
from audtorch.metrics.functional import pearsonr
from scipy.linalg import eigh
from scipy.optimize import linprog
import numpy as np
from scipy.stats import wasserstein_distance


class CategoryAlign_Module(nn.Module):
    def __init__(self, num_classes=7, ignore_bg=False):
        super(CategoryAlign_Module, self).__init__()
        self.num_classes = num_classes
        self.ignore_bg = ignore_bg

    def get_context(self, preds, feats):
        b, c, h, w = feats.size()
        _, num_cls, _, _ = preds.size()

        # softmax preds
        assert preds.max() <= 1 and preds.min() >= 0, print(preds.max(), preds.min())
        preds = preds.view(b, num_cls, 1, h * w)  # (b, num_cls, 1, hw)
        feats = feats.view(b, 1, c, h * w)  # (b, 1, c, hw)

        vectors = (feats * preds).sum(-1) / preds.sum(-1)  # (b, num_cls, C)

        if self.ignore_bg:
            vectors = vectors[:, 1:, :]  # ignore the background
        vectors = F.normalize(vectors, dim=1)
        return vectors

    def get_intra_corcoef_mat(self, preds, feats):
        context = self.get_context(preds, feats).mean(0)

        n, c = context.size()
        mat = torch.zeros([n, n]).to(context.device)
        for i in range(n):
            for j in range(n):
                cor = pearsonr(context[i, :], context[j, :])
                mat[i, j] += cor[0]
        return mat

    def get_cross_corcoef_mat(self, preds1, feats1, preds2, feats2):
        context1 = self.get_context(preds1, feats1).mean(0)
        context2 = self.get_context(preds2, feats2).mean(0)

        n, c = context1.size()
        mat = torch.zeros([n, n]).to(context1.device)
        for i in range(n):
            for j in range(n):
                cor = pearsonr(context1[i, :], context2[j, :])
                mat[i, j] += cor[0]
        return mat

    def regularize(self, cor_mat):
        n = self.num_classes - 1 if self.ignore_bg else self.num_classes
        assert cor_mat.size()[0] == n

        # label = (torch.ones([n, n]) * -1).to(cor_mat.device)
        # diag = torch.diag_embed(torch.Tensor([2]).repeat(1, n)).to(cor_mat.device)
        # label = (label + diag).view(n, n)
        #
        # # loss = - torch.log(torch.clamp(label * cor_mat, min=1e-6))
        # loss = (1 - label*cor_mat).pow(2)
        pos = - torch.log(torch.diag(cor_mat)).mean()
        undiag = cor_mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        # undiag_clone = undiag.clone()
        low = torch.Tensor([1e-6]).to(undiag.device)
        neg = - torch.log(1 - undiag.max(low)).mean()

        loss = pos + neg
        return loss


def emd_intra(inputs, multi_layer=False, ignore_bg=True):
    m = CategoryAlign_Module(ignore_bg=ignore_bg)

    preds1, preds2, feats = inputs

    B = preds1.size()[0]
    preds = ((preds1.softmax(dim=1) + preds2.softmax(dim=1)) / 2).detach()
    feats = F.interpolate(feats, (64, 64), mode='bilinear', align_corners=True)

    preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
    preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
    # cor_mat = m.get_cross_corcoef_mat(preds1, feats1, preds2, feats2)
    context1, context2 = m.get_context(preds1, feats1), m.get_context(preds2, feats2)

    cov_matrices1 = compute_covariance_matrix(context1.cpu().detach())
    cov_matrices2 = compute_covariance_matrix(context2.cpu().detach())

    # 获取主要方向
    directions1 = get_principal_directions(cov_matrices1)
    directions2 = get_principal_directions(cov_matrices2)

    # 计算 Sliced-Wasserstein 距离
    swd = compute_swd(context1.cpu().detach(), context2.cpu().detach(), directions1, directions2)
    return swd


def emd_Cross(source, target, ignore_bg=True, multi_layer=True):
    """
        EMD-Alignment for cross domain
    """
    m = CategoryAlign_Module(ignore_bg=ignore_bg)

    S_preds1, S_preds2, S_feats = source
    T_preds1, T_preds2, T_feats = target
    S_feats = F.interpolate(S_feats, (64, 64), mode='bilinear', align_corners=True)
    T_feats = F.interpolate(T_feats, (64, 64), mode='bilinear', align_corners=True)

    S_preds = ((S_preds1.softmax(dim=1) + S_preds2.softmax(dim=1)) / 2)
    T_preds = ((T_preds1.softmax(dim=1) + T_preds2.softmax(dim=1)) / 2)

    context1, context2 = m.get_context(S_preds.detach(), S_feats.detach()), \
        m.get_context(T_preds.detach(), T_feats)
    # 计算协方差矩阵
    cov_matrices1 = compute_covariance_matrix(context1.cpu().detach())
    cov_matrices2 = compute_covariance_matrix(context2.cpu().detach())

    # 获取主要方向
    directions1 = get_principal_directions(cov_matrices1)
    directions2 = get_principal_directions(cov_matrices2)

    # 计算 Sliced-Wasserstein 距离
    swd = compute_swd(context1.cpu().detach(), context2.cpu().detach(), directions1, directions2)

    return swd


def compute_covariance_matrix(tensor):
    # 计算每个样本的协方差矩阵
    n_samples, n_features, n_instances = tensor.shape
    cov_matrices = []
    for i in range(n_samples):
        data = tensor[i]
        cov_matrix = np.cov(data)
        cov_matrices.append(cov_matrix)
    return np.array(cov_matrices)


def get_principal_directions(cov_matrices, n_directions=7):
    # 计算特征值和特征向量，并选择前n_directions个主要方向
    principal_directions = []
    for cov_matrix in cov_matrices:
        eigvals, eigvecs = eigh(cov_matrix)
        idx = np.argsort(eigvals)[-n_directions:]  # 选择最大的n_directions个特征值对应的特征向量
        principal_directions.append(eigvecs[:, idx])
    return np.array(principal_directions)


def project_onto_directions(tensor, directions):
    # 将数据投影到主要方向上
    n_samples, n_features, n_instances = tensor.shape
    projected_data = np.zeros((n_samples, n_features, n_instances))
    for i in range(n_samples):
        for j in range(n_instances):
            projected_data[i, :, j] = np.dot(directions[i].T, tensor[i, :, j])
    return projected_data


def compute_swd(tensor1, tensor2, directions1, directions2):
    # 投影到主要方向上
    proj_tensor1 = project_onto_directions(tensor1, directions1)
    proj_tensor2 = project_onto_directions(tensor2, directions2)

    # 计算 Wasserstein 距离
    swd = 0
    for i in range(proj_tensor1.shape[1]):
        for j in range(proj_tensor1.shape[0]):
            swd += wasserstein_distance(proj_tensor1[j, i], proj_tensor2[j, i])
    return swd / (proj_tensor1.shape[0] * proj_tensor1.shape[1])
