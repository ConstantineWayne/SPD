import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from sklearn import metrics
from copy import deepcopy
import numpy as np
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.dirichlet import Dirichlet
from collections import defaultdict


def logBetaFunc(values):
    res = torch.sum(torch.lgamma(values), dim=1) - torch.lgamma(torch.sum(values, dim=1))
    return res


class GroupDirichlet(ExponentialFamily):
    def __init__(self, concentration_a, concentration_b, partition_list, validate_args=False):
        if concentration_a.dim() == 1:
            concentration_a = concentration_a.unsqueeze(0)
        if concentration_b.dim() == 1:
            concentration_b = concentration_b.unsqueeze(0)
        self.concentration_a = concentration_a
        self.concentration_b = concentration_b
        self.partition_list = partition_list
        self.n_partition = len(partition_list)

        assert self.concentration_b.shape[-1] == self.n_partition

        self.partition_cat_list = sum(self.partition_list, [])

        concentration_Y = []
        concentration_R = []
        for i in range(self.n_partition):
            parti_curr_ids = self.partition_list[i]
            concent_parti_cur = torch.index_select(self.concentration_a, 1, torch.tensor(parti_curr_ids).cuda())
            concentration_Y.append(concent_parti_cur)
            concentration_R.append(torch.sum(concent_parti_cur, dim=1) + self.concentration_b[:, i])

        self.concentration_Y = concentration_Y
        self.concentration_R = torch.stack(concentration_R, dim=1)

        super().__init__(validate_args=validate_args)

    def rsample(self, sample_shape=()):
        result = []
        dir_R_tmp = Dirichlet(self.concentration_R)
        R_samples = dir_R_tmp.sample(sample_shape)

        # get samples per partition
        for i in range(self.n_partition):
            dir_Y_curr_part = Dirichlet(self.concentration_Y[i])
            Y_samples_curr = dir_Y_curr_part.sample(sample_shape)
            R_sample_curr = R_samples[:, :, i].unsqueeze(dim=1).expand(Y_samples_curr.shape)
            result.append(R_sample_curr * Y_samples_curr)

        result = torch.cat(result, dim=1)

        _, indx = torch.sort(torch.tensor(self.partition_cat_list))

        return result.index_select(1, indx)

    def log_normalized_constant(self):
        logCg = []
        for i in range(self.n_partition):
            log_beta_value = logBetaFunc(self.concentration_Y[i])
            logCg.append(log_beta_value)
        logCg.append(logBetaFunc(self.concentration_R))
        return torch.stack(logCg, dim=1).sum(-1)

    def log_prob(self, value):
        term_1 = (torch.log(value) * (self.concentration_a - 1.0)).sum(-1)

        partition_prob_sums = []
        for i in range(self.n_partition):
            part_idx_curr = self.partition_list[i]
            prob_sum_part_curr = torch.sum(value[:, part_idx_curr], dim=-1, keepdim=True)
            partition_prob_sums.append(prob_sum_part_curr)
        partition_prob_sums = torch.cat(partition_prob_sums, dim=-1)
        term_2 = (torch.log(partition_prob_sums) * self.concentration_b).sum(-1)
        term_3 = self.log_normalized_constant()
        return term_1 + term_2 - term_3

    def entropy(self):
        log_Cg = self.log_normalized_constant()
        num_singles = self.concentration_a.size(-1)
        a0 = self.concentration_a.sum(-1)
        term2 = ((self.concentration_a - 1.0) * (torch.digamma(self.concentration_a))).sum(-1)

        beta = []
        partition_indicator = []
        for j in range(self.n_partition):
            part_idx_curr = self.partition_list[j]
            alpha_j = torch.sum(self.concentration_a[:, part_idx_curr], dim=1)
            beta_j = alpha_j + self.concentration_b[:, j]
            beta.append(beta_j)
            diff = torch.digamma(beta_j) - torch.digamma(alpha_j)
            partition_indicator.append(diff)
        beta_tensor = torch.stack(beta, dim=1)
        beta0 = beta_tensor.sum(-1)
        term3 = (a0 - num_singles) * torch.digamma(beta0)

        term4 = 0.
        for k in range(num_singles):
            partition_id = find_partition(self.partition_list, k)
            term4 += partition_indicator[partition_id] * (self.concentration_a[:, k] - 1)

        term5 = torch.sum(self.concentration_b * torch.digamma(beta_tensor), dim=1)
        term6 = torch.digamma(beta0) * torch.sum(self.concentration_b, dim=1)

        return log_Cg - term2 + term3 - term4 - term5 + term6


def find_partition(partition_list, class_id):
    n_pars = len(partition_list)
    for i in range(n_pars):
        partition = partition_list[i]
        if class_id in partition:
            return i


def numerator_GDD(probabilities, concentration_a, concentration_b, idx_comp_list):
    log_probs = torch.log(probabilities)
    part_1 = torch.sum(log_probs * (concentration_a - 1), dim=1, keepdim=True)

    n_partition = len(idx_comp_list)

    part_2 = []
    for i in range(n_partition):
        part_idx_curr = idx_comp_list[i]
        tmp = torch.sum(probabilities[:, part_idx_curr], dim=1, keepdim=True)
        log_prob_tmp = torch.log(tmp)
        pow = concentration_b[i]
        part_2.append(log_prob_tmp * pow)
    part_2 = torch.cat(part_2, dim=1)
    part_2 = torch.sum(part_2, dim=1, keepdim=True)

    return part_1 + part_2


def numerator_HDD(probabilities, concentration_a, concentration_b, idx_comp_list):
    log_probs = torch.log(probabilities)
    part_1 = torch.sum(log_probs * (concentration_a - 1), dim=1, keepdim=True)

    n_partition = len(idx_comp_list)

    part_2 = []
    for i in range(n_partition):
        part_idx_curr = idx_comp_list[i]
        tmp = torch.sum(probabilities[:, part_idx_curr], dim=1, keepdim=True)
        log_prob_tmp = torch.log(tmp)
        pow = concentration_b[i]
        part_2.append(log_prob_tmp * pow)
    part_2 = torch.cat(part_2, dim=1)
    part_2 = torch.sum(part_2, dim=1, keepdim=True)

    return part_1 + part_2


class HDD(object):
    def __init__(self, probabilities, concentration_a=None, concentration_b=None, idx_comp_list=None):
        self.probabilities = probabilities
        self.concentration_a = concentration_a
        self.concentration_b = concentration_b
        self.idx_comp_list = idx_comp_list


def weights_assigned(value, probabilities):
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities)
    sum_probs = torch.sum(probabilities)
    return probabilities / sum_probs * value


class GDD_latentZ(object):
    def __init__(self, probabilities, concentr_a, concentr_b_partition, concentr_b_comp, partition_list,
                 idx_comp_list) -> None:
        self.probabilities = probabilities
        self.concentr_a = concentr_a
        self.n_prob = len(concentr_a)

        self.concentr_b_partition = concentr_b_partition
        self.partition_list = partition_list
        self.n_partition = len(partition_list)

        self.concentr_b_comp = concentr_b_comp
        self.idx_comp_list = idx_comp_list
        self.n_comp = len(idx_comp_list)

    def Estep(self):
        latentZ = []
        for i in range(self.n_comp):
            parti_curr_ids = self.idx_comp_list[i]
            probs_curr = self.probabilities[parti_curr_ids]
            Z_curr = weights_assigned(self.concentr_b_comp[i], probs_curr)
            latentZ.append(Z_curr)
        self.latentZ = latentZ

    def Mstep(self):
        indx_z = defaultdict(lambda: 0)
        indx = sum(self.idx_comp_list, [])
        latentZ = torch.cat(self.latentZ, dim=0)
        for idx, z in zip(indx, latentZ):
            if idx not in indx_z:
                indx_z[idx] = z
            else:
                indx_z[idx] += z

        W = torch.sum(self.concentr_a) + torch.sum(self.concentr_b_comp) + torch.sum(self.concentr_b_partition) - len(
            self.probabilities)
        probabs = []
        for i in range(self.n_partition):
            part_curr_idx = self.partition_list[i]

            tmp_curr_part = 0
            for j in part_curr_idx:
                tmp_curr_part += indx_z[j]
                tmp_curr_part += self.concentr_a[j]
            tmp_curr_part -= len(part_curr_idx)

            term_2 = 1 + self.concentr_b_partition[i] / tmp_curr_part

            for j in part_curr_idx:
                term_1 = indx_z[j] + self.concentr_a[j] - 1
                curr_p = term_1 / W * term_2
                probabs.append(curr_p)

        # recover the order
        self.partition_cat_list = sum(self.partition_list, [])
        _, indx = torch.sort(torch.tensor(self.partition_cat_list))
        probabs_tensor = torch.tensor(probabs)
        self.probabilities = probabs_tensor.index_select(0, indx)

    def iterate(self, n_iterations=5, verbose=True):
        N = n_iterations
        for i in range(1, N + 1):
            self.Estep()
            self.Mstep()
            if verbose:
                print(f'\n Iteration: {i}')
                print(f'Latent Z: {self.latentZ}')
                print(f'Theta (probs): {self.probabilities}')

        self.Estep()

        print("EM done")

    def add_Z_to_concentr_a(self):
        indx_z = defaultdict(lambda: 0)
        indx = sum(self.idx_comp_list, [])
        latentZ = torch.cat(self.latentZ, dim=0)
        for idx, z in zip(indx, latentZ):
            if idx not in indx_z:
                indx_z[idx] = z
            else:
                indx_z[idx] += z
        latentZ_list = []
        for i in range(self.n_prob):
            latentZ_list.append(indx_z[i])

        return self.concentr_a + torch.tensor(latentZ_list)


def confidence_interval(accuracies):
    accuracies = np.array(accuracies)
    stds = np.std(accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(len(accuracies))
    return stds, ci95


def one_hot_embedding(labels, num_classes=10, device='cpu'):
    y = torch.eye(num_classes, device=device)
    return y[labels]


def multi_hot_embedding(label, R, num_classes=10, device='cpu'):
    y = torch.eye(num_classes, device=device)
    composite_labels = R[label]
    return y[composite_labels].sum(dim=0)


def multi_hot_embedding_batch(labels, R, num_single, device='cpu'):
    res = []
    if labels.dim() == 0:
        labels = labels.unsqueeze(dim=0) # compatible with batch_size=1
    for label in labels:
        res.append(multi_hot_embedding(label, R, num_single, device))
    return torch.stack(res, dim=0)


def complete_comp_labels(input_list, set_list):
    output = []
    for item in input_list:
        found_set = False
        for set_comp in set_list:
            if item in set_comp:
                output.append(set_comp)
                found_set = True
                break
        if not found_set:
            output.append([item])
    return output


def pseudo_target(pseudo_comp, R, num_single, device='cpu'):
    res = []
    comp_set = R[num_single:]
    for pseudo in pseudo_comp:
        if pseudo in comp_set:
            res.append(comp_set.index(pseudo) + num_single)
        else:
            res.append(pseudo[0])
    return torch.tensor(res, dtype=torch.long, device=device)

def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def kl_GDD(alpha, evidence_comps, num_single, R, targets, device=None):
    one_hot_embed = one_hot_embedding(targets, num_classes=len(R), device=device)
    one_hot_embed_cut = one_hot_embed[:, :num_single]
    alpha = (alpha - 1) * (1 - one_hot_embed_cut) + 1
    one_hot_comp_cut = one_hot_embed[:, num_single:]
    evidence_comps = evidence_comps * (1 - one_hot_comp_cut)
    unique_comp_sets = R[num_single:]
    uniques_elements_comp = sum(unique_comp_sets, [])
    comp_rest_labels = list(set(range(num_single)) - set(uniques_elements_comp))  # [0,2,4,5]
    unique_comp_sets.append(comp_rest_labels)
    pading = torch.zeros(len(targets), 1, device=device, requires_grad=True)
    evidence_comps_custom = torch.cat([evidence_comps, pading], dim=1)
    one_GDD = GroupDirichlet(alpha, evidence_comps_custom, unique_comp_sets)

    alpha_one = torch.ones_like(alpha)
    evidence_comps_one = torch.zeros_like(evidence_comps_custom)
    GDD_tmp = GroupDirichlet(alpha_one, evidence_comps_one, unique_comp_sets)
    log_Cg_tmp = GDD_tmp.log_normalized_constant()

    log_Cg = one_GDD.log_normalized_constant()
    num_singles = one_GDD.concentration_a.size(-1)
    a0 = one_GDD.concentration_a.sum(-1)
    term2 = ((one_GDD.concentration_a - 1.0) * (torch.digamma(one_GDD.concentration_a))).sum(-1)

    beta = []
    partition_indicator = []
    for j in range(one_GDD.n_partition):
        part_idx_curr = one_GDD.partition_list[j]
        alpha_j = torch.sum(one_GDD.concentration_a[:, part_idx_curr], dim=1)
        beta_j = alpha_j + one_GDD.concentration_b[:, j]
        beta.append(beta_j)
        diff = torch.digamma(beta_j) - torch.digamma(alpha_j)
        partition_indicator.append(diff)
    beta_tensor = torch.stack(beta, dim=1)
    beta0 = beta_tensor.sum(-1)
    term3 = (a0 - num_singles) * torch.digamma(beta0)

    term4 = 0.
    for k in range(num_singles):
        partition_id = find_partition(one_GDD.partition_list, k)
        term4 += partition_indicator[partition_id] * (one_GDD.concentration_a[:, k] - 1)

    term5 = torch.sum(one_GDD.concentration_b * torch.digamma(beta_tensor), dim=1)
    term6 = torch.digamma(beta0) * torch.sum(one_GDD.concentration_b, dim=1)

    kl_gdd = log_Cg - log_Cg_tmp - term2 + term3 - term4 - term5 + term6
    return -kl_gdd.mean()