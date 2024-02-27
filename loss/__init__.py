import torch
import torch.nn as nn
import torch.nn.functional as F

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

class Loss(nn.Module):
    #不知道是什么但是work的loss

    def __init__(self, t=0.7):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.t = t

    def forward(self, feature1, feature2):
        """
        feature1: shape(n,c)
        feature2: shape(n,c)
        """
        # n, c = feature1.shape
        # sim_matrix = feature1 @ feature2.transpose(-2, -1) / self.t # n n
        # logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        # sim_matrix = sim_matrix - logits_max
        # exp_matrix=torch.exp(sim_matrix)
        # positive_mask = torch.eye(n).to(feature1.device)
        # negative_mask = torch.tensor(1).to(feature1.device) - positive_mask
        #
        # positive_loss=(exp_matrix * positive_mask).sum(dim=1)
        # negative_loss=(exp_matrix * negative_mask).sum(dim=1)
        # return -torch.log(positive_loss/negative_loss).mean()



        n, c = feature1.shape
        sim_matrix = feature1 @ feature2.transpose(-2, -1)
        target = torch.eye(n, device=sim_matrix.device)
        return self.loss(sim_matrix.flatten(), target.flatten()) / c


class BarlowTwins(nn.Module):
    def __init__(self, t=0.7):
        super().__init__()
        self.loss = nn.MSELoss()
        self.t = t

    def forward(self, feature1, feature2):
        n, c = feature1.shape
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        sim_matrix = feature1 @ feature2.transpose(-2, -1)
        target = torch.eye(n, device=sim_matrix.device)
        sim_matrix=torch.pow(sim_matrix-target,2)
        positive=sim_matrix*target
        negative=(sim_matrix-positive)*self.t

        return (positive+negative).sum()/c





class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.7, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]



class Sup_infonce(nn.Module):

    def __init__(self, temperature=0.7, scale_by_temperature=True):
        super(Sup_infonce, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        else:
            mask = mask.float().to(device)
        # '''
        # 示例:
        # labels:
        #     tensor([[1.],
        #             [2.],
        #             [1.],
        #             [1.]])
        # mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
        #     tensor([[1., 0., 1., 1.],
        #             [0., 1., 0., 0.],
        #             [1., 0., 1., 1.],
        #             [1., 0., 1., 1.]])
        # '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # '''
        # logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        # 示例: logits: torch.size([4,4])
        # logits:
        #     tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
        #             [-1.2576,  0.0000, -0.3367, -0.0725],
        #             [-1.3500, -0.1409, -0.1420,  0.0000],
        #             [-1.4312, -0.0776, -0.2009,  0.0000]])
        # '''
        # 构建mask
        logits_mask = torch.ones_like(mask).to(features.device) - torch.eye(batch_size).to(features.device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        # '''
        # 但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # # 第ind行第ind位置填充为0
        # 得到logits_mask:
        #     tensor([[0., 1., 1., 1.],
        #             [1., 0., 1., 1.],
        #             [1., 1., 0., 1.],
        #             [1., 1., 1., 0.]])
        # positives_mask:
        # tensor([[0., 0., 1., 1.],
        #         [0., 0., 0., 0.],
        #         [1., 0., 0., 1.],
        #         [1., 0., 1., 0.]])
        # negatives_mask:
        # tensor([[0., 1., 0., 0.],
        #         [1., 0., 1., 1.],
        #         [0., 1., 0., 0.],
        #         [0., 1., 0., 0.]])
        # '''
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        # '''
        # 计算正样本平均的log-likelihood
        # 考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        # 所以这里只计算正样本个数>0的
        # '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
