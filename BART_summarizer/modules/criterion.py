import torch.nn as nn


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, label_smoothing, padding_idx=1, reduce=True):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.eps = label_smoothing
        self.padding_idx = padding_idx
        self.reduce = reduce

    def forward(self, lprobs, target):
        """
        Args:
            lprobs (Tensor): [batch_size * tgt_len, vocab_size]
            target (Tensor): [batch_size * tgt_len, 1]

        """
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=self.reduce,
        )
        return loss, nll_loss