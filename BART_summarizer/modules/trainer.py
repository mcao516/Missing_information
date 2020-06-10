#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import logging

from fairseq.models.bart import BARTModel
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule
from fairseq.utils import move_to_cuda

from models.general_utils import Progbar


class Trainer(object):
    """Class for BART model training, evaluation and test."""

    def __init__(self, args, logger):
        super(Trainer, self).__init__()
        self.args = args
        self.logger = logger

        self.optimizer = None
        self.scheduler = None
        self._num_updates = 0

        # Print args
        self.logger.info(args)

        # set cuda device
        self.cuda = torch.cuda.is_available() and not args.cpu
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Load source & target dictionary
        self.src_dict, self.tgt_dict = args.src_dict, args.tgt_dict

        # build criterion: module for loss computation
        self._build_criterion()
        self.criterion = self.criterion.to(device=self.device)

        # Load BART model
        self._build_model()
        self.model = self.model.to(device=self.device)

        self._build_optimizer()
        self._build_scheduler()

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.scheduler.step_update(self.get_num_updates())
        return new_lr

    def _build_criterion(self):
        """Build criterion.
        """        
        self.criterion = LabelSmoothedCrossEntropyCriterion(self.args.label_smoothing,
                                                            padding_idx=self.src_dict.pad(),
                                                            reduce=self.args.reduce):

    def _build_model(self):
        """Build BART model.
        """
        self.bart = BARTModel.from_pretrained(args.bart_path,
                                              checkpoint_file=args.checkpoint_file,
                                              data_name_or_path=args.data_name_or_path)
        self.model = self.bart.model

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad, self.model.parameters(),
            )
        )
        self.optimizer = FairseqAdam(self.args, params)

    def _build_scheduler(self):
        if self.optimizer is None:
            self._build_optimizer()

        self.scheduler = PolynomialDecaySchedule(self.args, self.optimizer)
        self.scheduler.step_update(0)

    def clip_grad_norm(self, clip_norm):
        return self.optimizer.clip_grad_norm(clip_norm, aggregate_norm_fn=None)

    def save_model(self):
        """Saves session = weights"""
        if not os.path.exists(self.args.dir_model):
            os.makedirs(self.args.dir_model)

        save_path = self.args.dir_model + 'checkpoint.pth.tar'
        torch.save(self.bart.state_dict(), save_path)
        self.logger.info("- model saved at: {}".format(save_path))

    def restore_model(self, model_path):
        """Load pre-trained model.
        """
        self.bart.load_state_dict(torch.load(model_path))
        self.model = self.bart.model
        self.logger.info("- model restored from: {}".format(model_path))

    def run_epoch(self, train, epoch):
        """Performs one complete pass over the train set and evaluate on devlopment set.

        Args:
            train (DataLoader): training set dataloader.
            epoch (int): index of the current epoch.

        """
        self.model.train()
        self.criterion.train()

        # progbar stuff for logging
        batch_size = self.args.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        logging_outputs = []
        for i, batch in enumerate(train.batch_iter(batch_size)):
            batch = move_to_cuda(batch)
            target = batch["target"].view(-1, 1)

            net_output = self.model(**batch['net_input'])

            # [batch_size, max_tgt_len, vocab] => [batch_size * max_tgt_len, vocab]
            lprobs = self.model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))

            # calculate loss & gradient
            loss, nll_loss = self.criterion(lprobs, target)
            self.optimizer.backward(loss)

            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data,
                'ntokens': batch['ntokens'],
                'nsentences': batch['target'].size(0),
            }
            logging_outputs += [logging_output]
            del loss

            # emptying the CUDA cache after the first step can
            # reduce the chance of OOM
            if self.cuda and self.get_num_updates() == 0:
                torch.cuda.empty_cache()

            # clip grads
            grad_norm = self.clip_grad_norm(self.args.clip_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.set_num_updates(self.get_num_updates() + 1)

            prog.update(i + 1,
                        values=[("token_loss", logging_output['loss'] / logging_output['ntokens'])],
                        exact=[("lr", self.get_lr()), 
                               ("num_updates", self.get_num_updates())])

        return logging_outputs

    def train(self, train, dev, samples=None):
        """Train the model and evaluate after each epoch."""
        self.logger.info('- start training...')

        best_score, nepoch_no_imprv = 0, 0  # for early stopping
        for epoch in range(self.args.max_epoch):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.args.nepochs))

            train.shuffle()
            logging_outputs = self.run_epoch(train, epoch)

            # evaluate the model
            self.logger.info('- evaluate on development set...')
            metrics = self.evaluate(dev)
            msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
            self.logger.info(msg)
            score = metrics["acc"]

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_model()
                best_score = score
                self.logger.info("- new best score! ")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break

    def predict_batch(self, context_input, article_input, beam_search=True):
        """Predict referring expression on a batch of data

           Returns:
               preds: list of ids in greedy mode, list of list of ids in beam search mode
        """
        pass

    def evaluate(self, test, pred_file=None):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset
        """
        pass
