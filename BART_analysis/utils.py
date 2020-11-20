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

# def get_probability(target, tokens, token_probs):
#     """Get probability of the given target.

#     Args:
#         target: Justin Martin
#         tokens: ['The', ' Archbishop', ' of', ...]
#         token_probs: [0.50, 0.49, 0.88, ...] 
#     """
#     assert len(tokens) == len(token_probs)
#     for i, t in enumerate(tokens):
#         if len(t) == 0: continue
#         prob = 1.0
#         t = t.strip()
#         if t in target:
#             prob = token_probs[i]
#             if t == target: return prob
#             for ni, (rt, rp) in enumerate(zip(tokens[i+1:], token_probs[i+1:])):
#                 if t == target: return prob
#                 elif len(t) < len(target):
#                     t += rt
#                     prob *= rp
#                 else:
#                     continue
#     print('Target ({}) not found!!!'.format(target))
#     return -1.0


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


def tokenize(src_input, encode_func, verbose=False):
    src_inputs = [src_input]  # list of input string
    src_tokens = collate_tokens([encode_func(i) for i in src_inputs], pad_idx=1, left_pad=True)
    src_tokens = src_tokens.cuda()
    src_lengths = torch.sum(src_tokens != 1, dim=1)
    
    if verbose:
        print('- src tokens: {};\n- src lengths: {}'.format(src_tokens.shape, src_lengths.shape))
    return src_tokens, src_lengths


def tokenize_with_mask(bart, input_sentence):
    bpe_code = bart.bpe.encode(input_sentence)  # <mask>: 1279 27932 29
    input_ids = bart.task.source_dictionary.encode_line('<s> ' + bpe_code.replace('1279 27932 29', '<mask>'), 
                                                        append_eos=True).long()
    input_ids = input_ids.unsqueeze(0).cuda()
    src_lengths = torch.sum(input_ids != 1, dim=1)
    return input_ids, src_lengths


def generate_sequence(bart, encoder_out, batch_size=1, tgt_tokens=None, min_decode_step=1, max_decode_step=100, pad_id=1, eos_id=2, verbose=True):
    init_input = torch.tensor([[2, 0]] * batch_size, dtype=torch.long).cuda()
    softmax = nn.Softmax(dim=1)
    token_probs, tokens = [], []

    for step in range(max_decode_step):
        decoder_outputs = bart.model.decoder(init_input, encoder_out, features_only=False)
        logits = decoder_outputs[0][:, -1, :]  # [batch_size, vocab]
        
        if step + 1 < min_decode_step:
            logits[:, eos_id] = -math.inf
        logits[:, pad_id], logits[:, 0] = -math.inf, -math.inf  # never select pad, start token

        probs = softmax(logits)
        assert logits.shape == probs.shape
        attn = decoder_outputs[1]['attn'][0]  # [batch_size, prev_token_len, src_token_len]
        assert logits.dim() == 2 and attn.dim() == 3

        if tgt_tokens is not None:
            selected_token = tgt_tokens[step].unsqueeze(0)
        else:
            value, indices = torch.topk(probs, 5, dim=1)
            selected_token = indices[:, 0]

        init_input = torch.cat([init_input, selected_token.unsqueeze(1)], dim=-1)
        token, prob = bart.decode(selected_token), probs.squeeze()[selected_token.item()].item()
        
        if selected_token.item() == eos_id:
            break
        elif verbose:
            print("- {:02d}: {} ({:.2f})".format(step, token, prob), end='\n')

        token_probs.append(prob)
        tokens.append(token)

    return init_input, tokens, token_probs


def get_probability(position, tokens, probs, entity):
    """Get probability of the given target.

    Args:
        position: (start, end)
        tokens: ['The', ' Archbishop', ' of', ...]
        probs: [0.50, 0.49, 0.88, ...]
        entity: "Rodgers"
    """
    assert len(tokens) == len(probs)
    
    end_pointer, end_pos = 0, []
    for t in tokens:
        end_pointer += len(t)
        end_pos.append(end_pointer)
    
    assert position[1] in end_pos, "- {}\n- {}\n- {}\n- {}\n- {}\n".format(position, tokens, probs, entity, end_pos)
    last_index = end_pos.index(position[1])
    indexes = [last_index]
    total_length = len(tokens[last_index])
    
    while total_length < (position[1] - position[0]):
        last_index -= 1
        assert last_index >= 0
        indexes.append(last_index)
        total_length += len(tokens[last_index])
    
    indexes.reverse()
    
    generated = ''.join([tokens[i] for i in indexes])
    assert entity in generated, 'entity: {}; prob calculated: {}'.format(entity, generated)
    
    prob = 1.0
    for i in indexes:
        prob *= probs[i]
    return prob


def get_cmlm_probability(bart_model, sentence, masked_sentence, position, entity, verbose=False):
    """Get the posterior probability of entity.
    
    Args:
        bart_model: BART model.
        sentence (str): summary with BOS token (<s>).
        masked_sentence (str): summary with the target entity masked (with ###) + source document.
        position: entity position (start, end). Might need to add 4 for the BOS token.
        entity (str): target entity.
    
    """
    masked_input, masked_lengths = tokenize(masked_sentence, bart_model.encode)
    masked_outputs = generate_sequence(bart_model,
                                       bart_model.model.encoder(masked_input,
                                                                src_lengths=masked_lengths),
                                       tgt_tokens=bart_model.encode(sentence)[1:].cuda(),
                                       verbose=verbose)
    masked_output_ids, masked_tokens, masked_token_probs = masked_outputs
    assert bart_model.decode(masked_output_ids[0]) == sentence
    assert ''.join(masked_tokens) == sentence
    
    return get_probability(position, masked_tokens, masked_token_probs, entity)


def get_prior_probability(bart_model, sentence, masked_sentence, position, entity, verbose=False):
    """Get the prior probability of entity.
    
    Args:
        bart_model: BART model.
        sentence (str): summary.
        masked_sentence (str): summary with the target entity masked.
        position: entity position (start, end).
        entity (str): target entity.
    
    """
    masked_input, masked_lengths = tokenize_with_mask(bart_model, masked_sentence)
    masked_outputs = generate_sequence(bart_model,
                                       bart_model.model.encoder(masked_input,
                                                                src_lengths=masked_lengths),
                                       tgt_tokens=bart_model.encode(sentence)[1:].cuda(),
                                       verbose=verbose)
    masked_output_ids, masked_tokens, masked_token_probs = masked_outputs
    assert bart_model.decode(masked_output_ids[0]) == sentence, '{}; {}'.format(bart_model.decode(masked_output_ids[0]), sentence)

    return get_probability(position, masked_tokens, masked_token_probs, entity)


def cmlm_generate(bart_model, masked_sentence, verbose=False):
    """Conditional masked language generation. Generate output with the masked reference and source document.
    
    Args:
        bart_model: BART model.
        masked_sentence (str): summary with the target entity masked + source document.
        
    """
    masked_input, masked_lengths = tokenize(masked_sentence, bart_model.encode)
    masked_outputs = generate_sequence(bart_model,
                                       bart_model.model.encoder(masked_input, 
                                                                src_lengths=masked_lengths),
                                       tgt_tokens=None,
                                       verbose=verbose)
    masked_output_ids, masked_tokens, masked_token_probs = masked_outputs
    
    return bart_model.decode(masked_output_ids[0])


def prior_generate(bart_model, masked_sentence):
    """Prior generation. Generate output without using the source document.
    
    Args:
        bart_model: BART model.
        masked_sentence (str): summary with the target entity masked.
        
    """
    masked_input, masked_lengths = tokenize_with_mask(bart_model, masked_sentence)
    masked_outputs = generate_sequence(bart_model,
                                       bart_model.model.encoder(masked_input, 
                                                                src_lengths=masked_lengths),
                                       tgt_tokens=None,
                                       verbose=False)
    masked_output_ids, masked_tokens, masked_token_probs = masked_outputs
    
    return bart_model.decode(masked_output_ids[0])