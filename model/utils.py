import random
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.dirichlet import Dirichlet
# from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


def get_GDD_distribution(evidence,n_classes=1):
    e_s = evidence[:,:n_classes]
    e_c = evidence[:,n_classes:]
    alpha_s = e_s + 1
    concentration = torch.cat([alpha_s,e_c],dim=1)
    beta = torch.sum(concentration,dim=1,keepdim=True)
    belief = evidence / (beta.expand(evidence.shape))
    uncertainty = evidence.size(1) / beta
    return beta,belief

def GDD_fusion_proj(belief,evidence, n_classes=1, composite=[0]):
    b_s = belief[:, :n_classes].clone()
    b_c = belief[:, n_classes:]
    for i, comp in enumerate(composite):
        b_comp = b_c[:, i].unsqueeze(1) / len(comp)
        for c in comp:
            b_s[:, c] += b_comp.squeeze(1)

    e_s = evidence[:, :n_classes].clone()
    e_c = evidence[:, n_classes:]

    for i, comp in enumerate(composite):
        e_comp = e_c[:, i].unsqueeze(1) / len(comp)
        for c in comp:
            e_s[:, c] += e_comp.squeeze(1)

    alpha = e_s + 1
    s = torch.sum(alpha,dim=1,keepdim=True)
    uncertainty = alpha.size(1) / s
    return b_s,e_s,alpha,uncertainty


def edl_singl_comp_loss(
        func,
        targets,
        R,
        alpha,
        evidence_comps,
        num_single,
        kl_reg=True,
        device=None
):
    if targets.dim() == 0:
        targets = targets.unsqueeze(dim=0)
    concentration = torch.cat([alpha, evidence_comps], dim=1)
    beta_sum = torch.sum(concentration, dim=1, keepdim=True)

    multi_hot_embed = multi_hot_embedding_batch(targets, R, num_single, device=device)
    padding = torch.zeros(len(targets), len(R) - num_single, device=device, requires_grad=True)
    multi_hot_pad_zero = torch.cat([multi_hot_embed, padding], dim=1)
    one_hot_embed = one_hot_embedding(targets, num_classes=len(R), device=device)
    scalar_indicator_comp = targets >= num_single
    mixed_hot_embed = multi_hot_pad_zero + one_hot_embed * scalar_indicator_comp[:, None]
    beta_gt_sum = torch.sum(mixed_hot_embed * concentration, dim=1, keepdim=True)
    uce_comp = func(beta_sum) - func(beta_gt_sum)

    alpha_gt_sum = torch.sum(multi_hot_pad_zero * concentration, dim=1, keepdim=True)
    uce_term21 = func(beta_sum) - func(alpha_gt_sum)
    R_comp = complete_comp_labels(targets, R[num_single:])
    targets_pseudo = pseudo_target(R_comp, R, num_single, device=device)
    mul_hot_embed_comp = multi_hot_embedding_batch(targets_pseudo, R, num_single, device=device)
    mul_hot_comp_pad_zero = torch.cat([mul_hot_embed_comp, padding], dim=1)
    one_hot_pseudo = one_hot_embedding(targets_pseudo, num_classes=len(R), device=device)
    scalar_indicator_comp_pseudo = targets_pseudo >= num_single
    mixed_hot_comp_pseudo = mul_hot_comp_pad_zero + one_hot_pseudo * scalar_indicator_comp_pseudo[:, None]
    beta_gt_sum_pseudo = torch.sum(mixed_hot_comp_pseudo * concentration, dim=1, keepdim=True)
    alpha_gt_sum_pseudo = torch.sum(mul_hot_comp_pad_zero * concentration, dim=1, keepdim=True)
    uce_term22 = func(beta_gt_sum_pseudo) - func(alpha_gt_sum_pseudo)
    uce_single = uce_term21 - uce_term22

    scalar_indicator_singl = targets < num_single
    uce_batch = uce_comp * scalar_indicator_comp[:, None] + uce_single * scalar_indicator_singl[:, None]
    uce_mean = torch.mean(uce_batch)

    if not kl_reg:
        return uce_mean, torch.tensor(0.).cuda()
    kl_alpha = alpha
    kl_term = kl_divergence(kl_alpha, num_single, device=device)
    kl_mean = torch.mean(kl_term)
    return uce_mean, kl_mean




def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    label = F.one_hot(p, num_classes=c)

    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))

def unified_UCE_loss(
        evidence,
        targets,
        comps,
        num_single,
        kl_lam_GDD,
        entropy_lam_Dir,
        entropy_lam_GDD,
        anneal=False,
        kl_reg=False,
        device=None
):
    if not anneal:
        assert anneal == False

    if targets.dim() == 0:
        targets = targets.unsqueeze(dim=0)  # compatible with batch_size=1

    if evidence.dim() == 1:
        evidence = evidence.unsqueeze(dim=0)
    R = build_R_from_composites(comps, num_single)
    evidence_single = evidence[:, :num_single]
    alpha = evidence_single + 1
    evidence_comps = evidence[:, num_single:]

    uce_mean, kl_mean = edl_singl_comp_loss(
        torch.digamma,
        targets,
        R,
        alpha,
        evidence_comps,
        num_single,
        kl_reg=kl_reg,
        device=device
    )

    if entropy_lam_Dir:
        entropy = Dirichlet(evidence + 1).entropy().mean()
    else:
        entropy = torch.tensor(0.).cuda()

    # Entropy of GDD
    if entropy_lam_GDD:
        unique_comp_sets = R[num_single:]
        uniques_elements_comp = sum(unique_comp_sets, [])
        comp_rest_labels = list(set(range(num_single)) - set(uniques_elements_comp))
        unique_comp_sets.append(comp_rest_labels)
        pading = torch.zeros(len(targets), 1, device=device, requires_grad=True)
        evidence_comps_custom = torch.cat([evidence_comps, pading], dim=1)
        entropy_GDD = GroupDirichlet(alpha, evidence_comps_custom, unique_comp_sets).entropy().mean()
    else:
        entropy_GDD = torch.tensor(0.).cuda()

    if kl_lam_GDD:
        kl_gdd = kl_GDD(alpha, evidence_comps, num_single, R, targets, device=device)
    else:
        kl_gdd = torch.tensor(0.).cuda()

    loss = uce_mean - entropy_lam_Dir * entropy - entropy_lam_GDD * entropy_GDD + kl_lam_GDD * kl_gdd

    return loss,entropy_GDD

def get_projection_distribution(a1, a2, n_classes):
    s1 = torch.sum(a1, dim=1, keepdim=True)
    s2 = torch.sum(a2, dim=1, keepdim=True)

    e1 = a1 - 1
    e2 = a2 - 1

    b1 = e1 / (s1 + 1e-8)
    b2 = e2 / (s2 + 1e-8)

    u1 = n_classes / (s1 + 1e-8)
    u2 = n_classes / (s2 + 1e-8)

    # 👉 reshape for bmm（保持空间）
    B, C, H, W = b1.shape

    b1_flat = b1.view(B, C, -1).permute(0, 2, 1)   # [B, HW, C]
    b2_flat = b2.view(B, C, -1).permute(0, 2, 1)   # [B, HW, C]

    # element-wise outer product（支持任意维度）
    bb = b1 * b2  # [B, C, H, W]

    # 因为 C=1，这里其实：
    bb_sum = torch.sum(bb, dim=1, keepdim=True)  # [B,1,H,W]
    bb_diag = bb_sum  # 单类情况下对角=全部

    K = bb_sum - bb_diag  # = 0 # [B, HW]

    K = K.view(B, 1, H, W)  # 🔥恢复空间结构

    # broadcast OK
    bu = b1 * u2
    ub = b2 * u1

    denominator = (1 - K + 1e-8)

    b_a = (b1 * b2 + bu + ub) / denominator
    u_a = (u1 * u2) / denominator

    S_a = n_classes / (u_a + 1e-8)
    e_a = b_a * S_a

    alpha_a = e_a + 1

    return alpha_a, b_a, u_a, S_a

def get_prediction_projection(belief):
    return torch.argmax(belief)

def KL(alpha,c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl