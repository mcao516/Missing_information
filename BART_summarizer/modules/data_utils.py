# Author: Meng Cao

import random

from fairseq.data.language_pair_dataset import collate


class DataLoader(object):
    def __init__(self, fairseq_dict, source_path, target_path, max_positions=1024, no_bos=True):
        """
        Args:
            fairseq_dict (fairseq.data.dictionary.Dictionary): fairseq dictionary.
            source_path (str): path to bpe encoded source.
            target_path (str): path to bpe encoded target.
            max_positions (int): max sentence length.
            no_bos (bool): whether append bos token.

        """
        self.fairseq_dict = fairseq_dict
        self.source_path = source_path
        self.target_path = target_path

        self.max_positions = max_positions
        self.no_bos = no_bos
        self.pad_idx = fairseq_dict.pad()
        self.eos_idx = fairseq_dict.eos()

        source = DataLoader.read_lines(source_path)
        target = DataLoader.read_lines(target_path)
        assert len(source) == len(target), "Source and target size do NOT match!"

        self.data = self.build_sample(source, target)
        del source, target

    def build_sample(self, source, target):
        data = []
        for i, (s, t) in enumerate(zip(source, target)):
            data.append({
                'id': i,
                'source': self.bpe_to_ids(s),
                'target': self.bpe_to_ids(t)
            })
        return data

    @staticmethod
    def read_lines(file_path):
        files = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                files.append(line.strip())
        return files

    def batch_iter(self, batch_size):
        """Create a batch of data.
        """
        samples = []
        for d in self.data:
            if len(samples) == batch_size:
                yield collate(samples, self.pad_idx, self.eos_idx, 
                              left_pad_source=True,
                              left_pad_target=False,
                              input_feeding=True)
                samples = []

            samples += [d]

        if len(samples) != 0:
            yield collate(samples, self.pad_idx, self.eos_idx,
                          left_pad_source=True,
                          left_pad_target=False,
                          input_feeding=True)

    def bpe_to_ids(self, sentence, addl_sentence=None):
        """Convert bpe ids to model input ids.

        Args:
            sentence (str): bpe encoded sentence.
            addl_sentence (str): bpe encoded sentence.

        """
        extra_tokens = 2
        bos, eos = '<s> ', ' </s>'

        if self.no_bos:
            bos = ''
            extra_tokens = 1

        if addl_sentence:
            tokens = sentence + eos + ' ' + addl_sentence
        else:
            tokens = sentence

        if len(tokens.split(' ')) > self.max_positions - extra_tokens:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - extra_tokens])
        bpe_sentence = bos + tokens + eos

        tokens = self.fairseq_dict.encode_line(bpe_sentence, append_eos=False)
        return tokens.long()

    def shuffle(self):
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)