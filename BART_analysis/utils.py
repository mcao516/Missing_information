import math
import torch
import torch.nn as nn

from fairseq.data.data_utils import collate_tokens


def read_lines(file_path):
    files = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            files.append(line.strip())
    return files


def get_probability(target, tokens, token_probs):
    """Get probability of the given target.

    Args:
        target: Justin Martin
        tokens: ['The', ' Archbishop', ' of', ...]
        token_probs: [0.50, 0.49, 0.88, ...] 
    """
    assert len(tokens) == len(token_probs)
    for i, t in enumerate(tokens):
        if len(t) == 0: continue
        prob = 1.0
        t = t.strip()
        if t in target:
            prob = token_probs[i]
            if t == target: return prob
            for ni, (rt, rp) in enumerate(zip(tokens[i+1:], token_probs[i+1:])):
                if t == target: return prob
                elif len(t) < len(target):
                    t += rt
                    prob *= rp
                else:
                    continue
    print('Target ({}) not found!!!'.format(target))
    return -1.0


def tokenize(src_input, encode_func, verbose=False):
    src_inputs = [src_input]  # list of input string
    src_tokens = collate_tokens([encode_func(i) for i in src_inputs], pad_idx=1, left_pad=True)
    src_tokens = src_tokens.cuda()
    src_lengths = torch.sum(src_tokens != 1, dim=1)
    
    if verbose:
        print('- src tokens: {};\n- src lengths: {}'.format(src_tokens.shape, src_lengths.shape))
    return src_tokens, src_lengths


def decode_sequence(decode_func, decoder, encoder_out, tgt_tokens=None, min_decode_step=10, max_decode_step=60, pad_id=1, eos_id=2, verbose=True):
    """Decode summary given encoder output (and target token).
    
    Args:
        decode_func: bart.decode
        decoder: BART decoder model.
        encoder_out: BART encoder output.
        tgt_tokens (Tensor): torch tensor with size: [1, 23].

    """
    batch_size = encoder_out[0].shape[1]
    init_input = torch.tensor([[2, 0]] * batch_size, dtype=torch.long).cuda()
    softmax = nn.Softmax(dim=1)
    token_probs, tokens, token_logits = [], [], []

    for step in range(max_decode_step):
        decoder_outputs = decoder(init_input, encoder_out, features_only=False)
        logits = decoder_outputs[0][:, -1, :]  # [batch_size, vocab]

        if step + 1 < min_decode_step:
            logits[:, eos_id] = -math.inf
        logits[:, pad_id], logits[:, 0] = -math.inf, -math.inf  # never select pad, start token

        probs = softmax(logits)
        attn = decoder_outputs[1]['attn'][0]  # [batch_size, prev_token_len, src_token_len]
        assert logits.dim() == 2 and attn.dim() == 3

        if tgt_tokens is not None:
            selected_token = tgt_tokens[step].unsqueeze(0)
        else:
            value, indices = torch.topk(probs, 5, dim=1)
            selected_token = indices[:, 0]

        init_input = torch.cat([init_input, selected_token.unsqueeze(1)], dim=-1)
        token, prob = decode_func(selected_token), probs.squeeze()[selected_token.item()].item()
        token_probs.append(prob)
        tokens.append(token)
        token_logits.append(logits)

        if selected_token.item() == eos_id:
            break
        elif verbose:
            print("- {:02d}: {} ({:.2f})".format(step, token, prob), end='\n')

    return init_input, tokens, token_probs, token_logits