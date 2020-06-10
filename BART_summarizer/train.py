#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging

from fairseq.tasks.translation import TranslationTask

from models.data_utils import DataLoader
from modules.trainer import Trainer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq.train')


def load_dictionary(path):
    """Load source & target fairseq dictionary.
    """
    # path = self.args.data_name_or_path
    src_dict = TranslationTask.load_dictionary(os.path.join(path, 'dict.{}.txt'.format('source')))
    tgt_dict = TranslationTask.load_dictionary(os.path.join(path, 'dict.{}.txt'.format('target')))

    assert src_dict.bos() == tgt_dict.bos() == 0
    assert src_dict.pad() == tgt_dict.pad() == 1
    assert src_dict.eos() == tgt_dict.eos() == 2
    assert src_dict.unk() == tgt_dict.unk() == 3
    logger.info('[{}] dictionary: {} types'.format('source', len(src_dict)))
    logger.info('[{}] dictionary: {} types'.format('target', len(tgt_dict)))

    return src_dict, tgt_dict


def main(args):
    # load dictionary
    src_dict, tgt_dict = load_dictionary(args.data_name_or_path)
    args.src_dict, args.tgt_dict = src_dict, tgt_dict

    # build trainer
    trainer = Trainer(args, logger)

    # create datasets
    train = DataLoader(src_dict, source_path, target_path,
                       max_positions=args.max_positions, no_bos=args.no_bos)
    dev = DataLoader(src_dict, source_path, target_path,
                     max_positions=args.max_positions, no_bos=args.no_bos)

    # train model
    trainer.train(train, dev, None)


if __name__ == "__main__":
    parser = init_arg_parser()
    args = parser.parse_args()

    main(args)