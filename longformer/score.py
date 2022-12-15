import os
import argparse
import random
import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import nlp
from rouge_score import rouge_scorer
import sacrebleu

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

import logging
from transformers import MBartTokenizer
from transformers import MBartForConditionalGeneration
from longformer.simplification import SimplificationDataset, Simplifier
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart
from longformer.longformer_encoder_decoder_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig
import datasets
import collections

from . import simplification
from longformer.simplification import prepare_input, get_eval_scores, remove_special_tokens

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# class SimplificationDatasetForInference(Dataset):
#     def __init__(self, inputs, reference, name, tokenizer, max_input_len, max_output_len, src_lang, tgt_lang, tags_included, target_tags):
#         self.inputs = inputs
#         self.reference = reference
#         self.name = name # train, val, test
#         self.tokenizer = tokenizer
#         self.max_input_len = max_input_len
#         self.max_output_len = max_output_len
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang
#         self.tags_included = tags_included
#         self.target_tags = target_tags
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         source = self.inputs[idx]['text']
#         reference = None
#         target_tags = None
#         if self.reference is not None:
#             reference = self.reference[idx]['text']
#         if self.target_tags is not None:
#             target_tags = self.target_tags[idx]['text']
#         if self.tags_included is not None:
#             sample = self.tokenizer.prepare_seq2seq_batch(src_texts=[source], tags_included=True , max_length=self.max_input_len, max_target_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt")
#         else:
#             sample = self.tokenizer.prepare_seq2seq_batch(src_texts=[source], src_lang=self.src_lang, tgt_lang=self.tgt_lang , max_length=self.max_input_len, max_target_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt") # TODO move this to _get_dataloader, preprocess everything at once?
#
#         input_ids = sample['input_ids'].squeeze()
#         input_ids = torch.roll(input_ids, shifts=-1) # move language tag to the end of the sequence in source
#         return input_ids, reference, target_tags
#
#     @staticmethod
#     def collate_fn(batch):
#
#         pad_token_id = 1
#
#         input_ids, ref, target_tags = list(zip(*batch))
#         input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
#         return input_ids, ref, target_tags


class SimplifierScorer(Simplifier):

    def __init__(self, params):
        super().__init__()
        # self.args = params
        # self.hparams = params
        # self.src_lang = self.args.src_lang
        # self.tgt_lang = self.args.tgt_lang
        # self.tags_included = self.args.tags_included
        # if self.args.from_pretrained is not None or args.resume_ckpt is not None:  ## TODO check if this is true with resume_ckpt
        #     self._set_config()
        #     self._load_pretrained()
        #
        # self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        # self.current_checkpoint = 0
        # self.best_checkpoint = None
        # self.best_metric = 10000 if self.args.early_stopping_metric == 'vloss' else 0  ## keep track of best dev value of whatever metric is used in early stopping callback
        # self.num_not_improved = 0
        # self.save_hyperparameters()

    def forward(self, input_ids, decoder_input_ids, labels):
        input_ids, attention_mask = prepare_input(input_ids, self.model, self.config.attention_mode,
                                                  self.tokenizer.pad_token_id, self.args.global_attention_indices)
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False, )
        lm_logits = outputs[0]
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.tokenizer.pad_token_id
            )
        return [loss]

    def test_step(self, batch, batch_nb):
        return self.forward(*batch)


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        parser.add_argument("--save_dir", type=str, default='simplification', help="Directory to save models.")
        parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,
                            help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0,
                            help="Number of evaluation sanity steps to run before starting the training. Default: 0.")

        # data
        parser.add_argument("--train_source", type=str, default=None, help="Path to the source train file.")
        parser.add_argument("--train_target", type=str, default=None, help="Path to the target train file.")
        parser.add_argument("--val_source", type=str, default=None, help="Path to the source validation file.")
        parser.add_argument("--val_target", type=str, default=None, help="Path to the target validation file.")
        parser.add_argument("--test_source", type=str, default=None,
                            help="Path to the source test file (to evaluate after training is finished).")
        parser.add_argument("--test_target", type=str, default=None,
                            help="Path to the target test file (to evaluate after training is finished).")
        parser.add_argument("--src_lang", type=str, default=None,
                            help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tgt_lang", type=str, default=None,
                            help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tags_included", action='store_true',
                            help="Text files already contain special tokens (language tags and </s>. Source:  src_tag seq, Target:  tgt_tag seq. Note: actual source sequence is seq src_tag </s>, will be changed internally after possibly clipping sequence to given max_length.")
        parser.add_argument("--max_output_len", type=int, default=256,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=512,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--wandb", type=str, default=None,
                            help="WandB project name to use if logging fine-tuning with WandB.")
        parser.add_argument("--remove_special_tokens_containing", type=str, nargs="+",
                            help="Remove tokens from the special_tokens_map that contain this string")

        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulation steps.")
        parser.add_argument("--gpus", type=int, default=-1, help="Number of gpus. 0 for CPU")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")

        ## model params:
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        parser.add_argument("--activation_dropout", type=float, default=0.0, help="activation_dropout")
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--global_attention_indices", type=int, nargs='+', default=[-1], required=False,
                            help="List of indices of positions with global attention for longformer attention. Supports negative indices (-1 == last non-padding token). Default: [-1] == last source token (==lang_id) .")

        # Optimization params:
        # parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Initial learning rate")
        parser.add_argument("--val_every", type=float, default=1.0,
                            help="Number of training steps between validations in percent of an epoch.")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--max_epochs", type=int, default=100000,
                            help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached).")
        parser.add_argument("--early_stopping_metric", type=str, default='rougeL',
                            help="Metric to be used for early stopping: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu")
        parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
        parser.add_argument("--min_delta", type=float, default=0.0,
                            help="Minimum change in the monitored quantity to qualify as an improvement.")
        parser.add_argument("--lr_reduce_patience", type=int, default=8,
                            help="Patience for LR reduction in Plateau scheduler.")
        parser.add_argument("--lr_reduce_factor", type=float, default=0.5,
                            help="Learning rate reduce factor for Plateau scheduler.")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--save_top_k", type=int, default=5,
                            help="Number of best checkpoints to keep. Others will be removed.")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')

        ## inference params
        parser.add_argument("--decoded", type=str, default='decoded.out',
                            help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=4,
                            help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')

        # logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0,
                            help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = Simplifier(args)

    if args.print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    model.datasets = datasets.load_dataset('text', data_files={'train_source': args.train_source,
                                                               'train_target': args.train_target,
                                                               'val_source': args.val_source,
                                                               'val_target': args.val_target,
                                                               'test_source': args.test_source,
                                                               'test_target': args.test_target})

    if args.wandb:
        logger = WandbLogger(project=args.wandb)
    else:
        logger = TestTubeLogger(
            save_dir=args.save_dir,
            name=args.save_prefix,
            version=0  # always use version=0
        )

    print(args)

    model.lr_mode = 'max'
    # if args.early_stopping_metric == 'val_loss':
    if args.early_stopping_metric == 'vloss':
        model.lr_mode = 'min'
    early_stop_callback = EarlyStopping(monitor=args.early_stopping_metric, min_delta=args.min_delta,
                                        patience=args.patience, verbose=True,
                                        mode=model.lr_mode)  # metrics: val_loss, bleu, rougeL

    custom_checkpoint_path = "checkpoint{{epoch:02d}}_{{{}".format(args.early_stopping_metric)
    custom_checkpoint_path += ':.5f}'

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, custom_checkpoint_path),
        save_top_k=args.save_top_k,
        verbose=True,
        monitor=args.early_stopping_metric,
        mode=model.lr_mode,
        prefix='')

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.max_epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.grad_accum,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=args.num_sanity_val_steps,
                         check_val_every_n_epoch=1 if not (args.debug) else 1,
                         limit_val_batches=args.val_percent_check,
                         limit_test_batches=args.test_percent_check,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=32 if args.fp32 else 16, amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         callbacks=[early_stop_callback]
                         )
    ## write config + tokenizer to save_dir
    # model.model.save_pretrained(args.save_dir + "/" + args.save_prefix)
    # if args.remove_special_tokens_containing:
    #     print("special tokens before:", model.tokenizer.special_tokens_map)
    #     model.tokenizer = remove_special_tokens(model.tokenizer, args.remove_special_tokens_containing)
    #     print("special tokens after:", model.tokenizer.special_tokens_map)
    # model.tokenizer.save_pretrained(args.save_dir + "/" + args.save_prefix)
    # trainer.fit(model)
    # print("Training ended. Best checkpoint {} with {} {}.".format(model.best_checkpoint, model.best_metric,
    # args.early_stopping_metric))
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="simplification")
    parser = SimplifierScorer.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
