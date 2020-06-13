import torch
import torch.nn as nn


class BARTWrapper(nn.Module):
    def __init__(self, bart, criterion):
        super(BARTWrapper, self).__init__()
        self.bart = bart
        self.criterion = criterion

    def forward(self, target, src_tokens, src_lengths, prev_output_tokens):
        net_output = self.bart(src_tokens, src_lengths, prev_output_tokens,
                               features_only=False,
                               classification_head_name=None)

        # [batch_size, max_tgt_len, vocab] => [batch_size * max_tgt_len, vocab]
        lprobs = self.bart.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # calculate loss & gradient
        loss, nll_loss = self.criterion(lprobs, target.view(-1, 1))

        return loss, nll_loss