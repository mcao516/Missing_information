import sys
import time
import math
import logging
import argparse
import numpy as np

from datetime import datetime


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


def init_arg_parser():
    parser = argparse.ArgumentParser()

    # save & load path
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())

    parser.add_argument("--bart_path", default='/network/home/caomeng/Downloads/BART/bart.large.xsum', type=str,
                        help="Path to BART model.")
    parser.add_argument("--checkpoint_file", default='model.pt', type=str,
                        help="Model name.")
    parser.add_argument("--data_name_or_path", default='/network/home/caomeng/Downloads/BART/bart.large.xsum', type=str,
                        help="BART data_name_or_path.")
    parser.add_argument('--save-dir', metavar='DIR', default='checkpoints/',
                        help='path to save checkpoints')

    parser.add_argument('--train_source', metavar='DIR', default='/network/home/caomeng/Downloads/summarization/XSum/fairseq_files/train.bpe.source',
                        help='path to training document.')
    parser.add_argument('--train_target', metavar='DIR', default='/network/home/caomeng/Downloads/summarization/XSum/fairseq_files/train.bpe.target',
                        help='path to training summary.')
    parser.add_argument('--dev_source', metavar='DIR', default='/network/home/caomeng/Downloads/summarization/XSum/fairseq_files/val.bpe.source',
                        help='path to devlopment document.')
    parser.add_argument('--dev_target', metavar='DIR', default='/network/home/caomeng/Downloads/summarization/XSum/fairseq_files/val.bpe.target',
                        help='path to devlopment summary.')

    # data preprocessing
    parser.add_argument("--max_positions", default=1024, type=int,
                        help="Max BART input length.") 
    parser.add_argument("--no_bos", action='store_true', help="Append bos to the input sequence.")

    # training
    parser.add_argument("--multi_gpu", action='store_true', default=True, help="Use multi-gpu for training.")
    parser.add_argument("--batch_size", default=24, type=int, help="Training batch-size.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, 
                        help="Label smoothing.")
    parser.add_argument("--reduce", action='store_false', help="Reduce loss.")
    parser.add_argument("--max_epoch", default=100, type=int,
                        help="Max number of epochs for training.")
    parser.add_argument("--total_num_update", default=2e4, type=int,
                        help="Max number of gradient updates for training.") 
    parser.add_argument('--warmup-updates', default=500, type=int, metavar='N',
                        help='warmup the learning rate linearly for the first N updates')
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument('--lr', '--learning-rate', default='3e-5', type=eval_str_list,
                        metavar='LR_1,LR_2,...,LR_N',
                        help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    parser.add_argument('--end-learning-rate', default=0.0, type=float)
    parser.add_argument('--power', default=1.0, type=float)
    parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                        help='force annealing at specified epoch')
    parser.add_argument("--clip_norm", default=0.1, type=float, help="Gradient clip.")
    parser.add_argument("--update-freq", default=2, type=int, help="How often to do gradient update.")
    parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                        help='betas for Adam optimizer')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                        help='epsilon for Adam optimizer')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                        help='weight decay')

    return parser


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.5f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.5f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)