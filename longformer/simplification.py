import os
import argparse
import random
import numpy as np
from typing import Optional, Tuple, List
import torch
from torch import Tensor

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import nlp
from rouge_score import rouge_scorer
import sacrebleu

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

import logging
from transformers import MBartTokenizer
from transformers import MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart
from longformer.longformer_encoder_decoder_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig
import datasets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def nansum(x):
    # Define nansum, as pytorch 1.6.x doesn't offer it inbuilt.
    return x[~torch.isnan(x)].sum()

def alignment_attention_loss(
    cross_attentions: Tuple[Tensor], 
    input_features: Tensor, 
    input_lens: Tensor, 
    output_lens: Tensor,
    attention_mass: float = 0.1, 
    layer: int = 2,
    head: int = -1,
    ):
    """
    Custom loss function to encourage model to pay attention
    to aligned segments where provided.

    Intuition: 
        - computes the cumulative attention on the annotated
        alignment spans
        - loss is defined as the difference between the
        reference value of expected attention mass 
        (assumed to be a uniform distribution of the
        total attention mass over the source sequence + the
        portion of attention mass provided as a hyperparameter) 
        and the average attention mass on all aligned tokens

    params:
        attention_mass: hyperparameter for deciding how much
        of the total attention should be directed towards
        aligned segments in addition to an underlying
        uniformly distributed attention
        layer: cross attention encoder layer for which to
        compute loss
        head: attention head for which to compute loss
    """

    # cross_attentions is a tuple of float tensors (one for each layer)
    # select cross_attention for specified layer
    cross_attentions = cross_attentions[layer]
    
    bsz, heads, tgt_length, src_length = cross_attentions.shape
    
    # select cross_attention for specified head 
    cross_attentions = cross_attentions[:,head,:,:] # => cross_attentions.shape = (batch_size, target_length, source_length)

    # input_features.shape = (batch_size, source_length)
    align_span_indices = [(i != 0).nonzero(as_tuple=False).squeeze(-1) for i in input_features] # => list of tensors (one for each item in batch) containing indices corresponding to aligned segments
    
    # compute attention mass of the aligned source positions across all target positions    
    total_attention_mass_on_spans = torch.stack(
        [torch.index_select(cross_attentions[i], dim=-1, index=align_span_indices[i]).sum(dim=-1).sum(dim=-1) for i in range(bsz)]
        )
    # compute average by dividing by the true tgt lengths (no padding)
    avg_attention_mass_on_spans = total_attention_mass_on_spans / output_lens # => shape = (batch_size)
    num_in_span_tokens = torch.LongTensor([i.size() for i in align_span_indices]).view(-1).to(cross_attentions.device) # => shape = (batch_size)
    
    # reference values are the expected attention mass 
    # ocross all source tokens given a uniform distribution
    # + an addition percentage of attention mass (hyperparam)
    uniform_att_mass_on_src = input_lens.float().reciprocal() # => shape = (batch_size)
    uniform_att_mass_on_span = num_in_span_tokens * uniform_att_mass_on_src # => shape = (batch_size)
    expected_att_mass_on_span = uniform_att_mass_on_span * (1.0 + attention_mass)

    # compute the difference between the reference value 
    # and the average attention mass on tokens in alignment spans
    loss_per_item = expected_att_mass_on_span - avg_attention_mass_on_spans
    
    # zero-out negative loss values (i.e. where avg
    # attention is already > expected uniform values)
    loss_per_item[loss_per_item < 0] = 0.0

    # compute the ratio of aligned span tokens to
    # non-aligned span tokens in source text
    # scaling_factor = torch.Tensor([len(align_span_indices[i]) / input_lens[i].item() for i in range(bsz)]).to(cross_attentions.device)
    # scaling_factor = num_in_span_tokens.true_divide(input_lens)
    # multiply by each item's individual loss by its
    # relevant scaling factor - items with no alignment
    # spans will be zero.
    # loss_per_item_scaled = loss_per_item * scaling_factor
    
    total_loss = nansum(loss_per_item)
    return total_loss.item()

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def prepare_input(input_ids, model, attention_mode, pad_token_id, global_attention_indices):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # attention longformer: 1 local, 2 global, 0 none
        attention_mask[input_ids == pad_token_id] = 0
        index_of_last_nonpad = (attention_mask.ne(0).sum(dim=1) - 1).squeeze(-1)
        if isinstance(model, MLongformerEncoderDecoderForConditionalGeneration):
            for glob_i in global_attention_indices:
                ## negative indices: discount from index_of_last_nonpad (only need to do this if batch_size > 1, otherwise there is no padding at this point and we can just use the negative indices directly
                if glob_i < 0 and input_ids.shape[0] > 1:
                    for i, last_nonpad in enumerate(index_of_last_nonpad): # i: iterator over samples in batch
                        glob = int(last_nonpad) + glob_i +1
                        attention_mask[i][int(glob)] = 2
                # indices > 0
                else:
                    attention_mask[:, glob_i] = 2
            if attention_mode == 'sliding_chunks':
                half_padding_mod = model.config.attention_window[0]
            elif attention_mode == 'sliding_chunks_no_overlap':
                half_padding_mod = model.config.attention_window[0] / 2
            else:
                raise NotImplementedError
            input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
                input_ids, attention_mask, half_padding_mod, pad_token_id)
        return input_ids, attention_mask

def get_eval_scores(ref, generated_strs, tags_included=False, vloss=None):
        if vloss is None:
            vloss = torch.zeros(len(ref))
        if tags_included:
            # remove tags from target text
            # print(gold_strs)
            gold_strs = [' '.join(r.split(' ')[1:]) for r in ref]
            # print(gold_strs)  ## TODO fix
        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0
        for ref, pred in zip(gold_strs, generated_strs):
            score = scorer.score(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure
        rouge1 /= len(generated_strs)
        rouge2 /= len(generated_strs)
        rougel /= len(generated_strs)
        rougelsum /= len(generated_strs)
        bleu = sacrebleu.corpus_bleu(generated_strs, [gold_strs])

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + rouge1,
                'rouge2': vloss.new_zeros(1) + rouge2,
                'rougeL': vloss.new_zeros(1) + rougel,
                'rougeLsum': vloss.new_zeros(1) + rougelsum,
                'bleu' : vloss.new_zeros(1) + bleu.score,
                'decoded' : generated_strs}

def flatten(lst):
    return [item for sublist in lst for item in sublist]

class SimplificationDataset(Dataset):
    def __init__(self, inputs, labels, name, tokenizer, max_input_len, max_output_len, src_lang, tgt_lang, tags_included, src_features):
        self.inputs = inputs
        self.labels = labels
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tags_included = tags_included
        self.src_features = src_features

    def __len__(self):
        return len(self.inputs)

    def convert_word_alignmnet_features_to_sub_word(self, word_sequence: str, feature_sequence: str) -> torch.Tensor:
        """
        words: source or target string
        """
        # account for the eos_token which gets
        # appended in the model's tokenization step
        words = word_sequence.split() + [self.tokenizer.eos_token]
        # eos_token token gets 0 as feature
        features = feature_sequence.split() + ['0']            
        # ensure that lengths of features correspond to
        # the length on the input text (separated by whtespace)
        assert len(words) == len(features), f"[!] Input token sequence has different length to token feature sequence\n{words}\n{features}"
        
        # get the sub-word representation of each word
        # e.g. [['de_DE', '], ['▁Gleich', 'stellung'], ['▁von'], ['▁Menschen'], ...]
        sub_words = [self.tokenizer.sp_model.EncodeAsPieces(word) if word not in self.tokenizer.all_special_tokens_extended else [word] for word in words]
        # expand each word-level feature to its
        # corresponding sub-words (note: features are
        # expected to be either 0 or 1)
        sub_word_features = [[int(feature)] * len(sp_tok) for feature, sp_tok in zip(features, sub_words)]

        # flatten 2D lists to 1D
        sub_words = flatten(sub_words)
        sub_word_features = flatten(sub_word_features)    

        # ensure that lengths still match
        assert len(sub_words) == len(sub_word_features), f"[!] Output token sequence has different length to sentiment sequence\n{sub_words}\n{sub_word_features}"
        
        features = torch.LongTensor(sub_word_features)
        return features

    def __getitem__(self, idx):
        source = self.inputs[idx]['text']
        target = self.labels[idx]['text']
        
        # alignment annotations
        src_features = None
        if self.src_features is not None:
            src_features = self.src_features[idx]['text']
            src_features = self.convert_word_alignmnet_features_to_sub_word(source, src_features)
        
        if self.tags_included:
            sample = self.tokenizer.prepare_seq2seq_batch(src_texts=[source], tgt_texts=[target], tags_included=True, max_length=self.max_input_len, max_target_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt")
        else:
            sample = self.tokenizer.prepare_seq2seq_batch(src_texts=[source], src_lang=self.src_lang, tgt_texts=[target], tgt_lang=self.tgt_lang , max_length=self.max_input_len, max_target_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt") # TODO move this to _get_dataloader, preprocess everything at once?

        input_ids = sample['input_ids'].squeeze()
        output_ids = sample['labels'].squeeze()
        if self.tags_included: # move language tag to the end of the sequence in source, also in target (so we can use mbarts shift_tokens_right that takes padding into account)
            input_ids = torch.cat([input_ids[1:], input_ids[:1]])
            output_ids = torch.cat([output_ids[1:], output_ids[:1]])
            if src_features is not None:
                # in case source inputs are truncated,
                # trim the corresponding feature sequence to
                # the same length of the final input sequence
                src_features = src_features[:len(input_ids)]
                # move the feature corresponding to the
                # language tag (originally in position 0)
                # to the end of the sequence (as above)
                src_features = torch.cat([src_features[1:], src_features[:1]])
        
        return input_ids, output_ids, src_features

    @staticmethod
    def collate_fn(batch):
        pad_token_id = 1
        input_ids, output_ids, src_features = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        if all(elem is not None for elem in src_features):
            src_features = torch.nn.utils.rnn.pad_sequence(src_features, batch_first=True, padding_value=0)
        
        return input_ids, output_ids, src_features


class Simplifier(pl.LightningModule):

    def __init__(self, params):
        super().__init__()
        self.args = params
        self.hparams = params
        self.src_lang = self.args.src_lang
        self.tgt_lang = self.args.tgt_lang
        self.tags_included = self.args.tags_included
        if self.args.from_pretrained is not None or args.resume_ckpt is not None: ## TODO check if this is true with resume_ckpt
            self._set_config()
            self._load_pretrained()

        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None
        self.current_checkpoint =0
        self.best_checkpoint = None
        self.best_metric = 10000 if self.args.early_stopping_metric == 'vloss' else 0 ## keep track of best dev value of whatever metric is used in early stopping callback
        self.num_not_improved = 0
        self.save_hyperparameters()
        
    def _load_pretrained(self):
        self.model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(self.args.from_pretrained, config=self.config)
        self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)
        if self.tags_included:
            self.model.config.decoder_start_token_id = -1
        else:
            self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[self.tgt_lang]
    
    def _set_config(self):
        self.config = MLongformerEncoderDecoderConfig.from_pretrained(self.args.from_pretrained)
        self.config.attention_dropout = self.args.attention_dropout
        self.config.dropout = self.args.dropout
        self.config.activation_dropout = self.args.activation_dropout
        self.config.gradient_checkpointing = self.args.grad_ckpt
        self.config.attention_mode = self.args.attention_mode
        self.config.attention_window = [self.args.attention_window] * self.config.encoder_layers

    def forward(self, input_ids, output_ids, input_features=None,):
        input_ids, attention_mask = prepare_input(input_ids, self.model, self.config.attention_mode, self.tokenizer.pad_token_id, self.args.global_attention_indices)
        decoder_input_ids = shift_tokens_right(output_ids, self.config.pad_token_id) # (in: output_ids, eos_token_id, tgt_lang_id out: tgt_lang_id, output_ids, eos_token_id)
        labels = decoder_input_ids[:, 1:].clone()
        decoder_input_ids = decoder_input_ids[:, :-1] # without eos/last pad
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        # breakpoint()
        # NOTE: --grad_ckpt must be False to get output_attentions
        
        padded_input_features = None
        if args.factor_features:
            padded_input_features = torch.nn.functional.pad(input_features, (self.args.max_input_len-input_features.size()[-1], 0), mode='constant')
    
        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                output_attentions=True,
                input_features=padded_input_features)

        lm_logits = outputs[0]
        
        # attentions are tuples of float tensors (one for each layer)
        # breakpoint()
        att_loss = 0
        if (outputs.cross_attentions is not None) and (input_features[0] is not None):
            input_lens = (input_ids != self.tokenizer.pad_token_id).sum(dim=-1)
            output_lens = (output_ids != self.tokenizer.pad_token_id).sum(dim=-1)
            att_loss = alignment_attention_loss(outputs.cross_attentions, input_features, input_lens, output_lens, attention_mass=self.args.att_loss_mass, layer=self.args.att_loss_layer, head=self.args.att_loss_head)

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
        
        combined_loss = loss + (self.args.att_loss_lambda * att_loss)
    
        return {
            'loss': loss,
            'losses': {
                'attention_alignment_loss': att_loss,
                'combined_loss': combined_loss
                }
            }

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output['losses']['combined_loss']
        att_loss = output['losses']['attention_alignment_loss']
        ce_loss = output['loss']

        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        self.log('train-loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('align_att_loss', att_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs['losses']['combined_loss']
        att_loss = outputs['losses']['attention_alignment_loss']
        ce_loss = outputs['loss']

        input_ids, output_ids, input_features = batch
        input_ids, attention_mask = prepare_input(input_ids, self.model, self.config.attention_mode, self.tokenizer.pad_token_id, self.args.global_attention_indices)
        if self.tags_included:
            # get list of target language tags
            # output_ids (batch_size, seq_len), with padding
            shifted = shift_tokens_right(output_ids, self.config.pad_token_id) # (in: output_ids, eos_token_id, tgt_lang_id out: tgt_lang_id, output_ids, eos_token_id)
            decoder_start_token_ids = shifted.narrow(dim=1, start=0, length=1)
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_ids=decoder_start_token_ids)
        else:
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang])

        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # get scores as dict
        scores = get_eval_scores(gold_str, generated_str, self.tags_included, vloss)
        
        outfile = self.args.save_dir + "/" + args.save_prefix + "/_val_out_checkpoint_" + str(self.current_checkpoint)

        with open(outfile, 'a') as f:
            for sample in generated_str:
                f.write(sample + "\n")
        self.log('vloss', vloss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('ce_loss', ce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('att_loss', att_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('bleu', scores['bleu'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rouge1', scores['rouge1'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rouge2', scores['rouge2'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rougeL', scores['rougeL'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('rougeLsum', scores['rougeLsum'], on_step=False, on_epoch=True, prog_bar=True)
        
        return scores

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        print("Evaluation on checkpoint [{}] ".format(self.current_checkpoint))
        print(logs)
        
        ## save metric value + number of checkpoint if best
        if self.args.early_stopping_metric == 'vloss' and logs['vloss'] < self.best_metric:
            self.best_metric = logs['vloss']
            self.best_checkpoint = self.current_checkpoint
            print("New best checkpoint {}, with {} {}.".format(self.best_checkpoint, self.best_metric, self.args.early_stopping_metric))
        elif logs[self.args.early_stopping_metric] > self.best_metric:
            self.best_metric = logs[self.args.early_stopping_metric]
            self.best_checkpoint = self.current_checkpoint
            print("New best checkpoint {}, with {} {}.".format(self.best_checkpoint, self.best_metric, self.args.early_stopping_metric))
        self.current_checkpoint +=1
        

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        print(result)
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode=self.lr_mode, factor=self.args.lr_reduce_factor, patience=self.args.lr_reduce_patience)
        return {
       'optimizer': self.optimizer,
       'lr_scheduler': self.scheduler,
       'monitor': self.args.early_stopping_metric
        }

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        
        dataset = SimplificationDataset(
            inputs=self.datasets[split_name + "_source"],
            labels=self.datasets[split_name + "_target"] ,
            name=split_name, tokenizer=self.tokenizer,
            max_input_len=self.args.max_input_len,
            max_output_len=self.args.max_output_len,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang, 
            tags_included=args.tags_included,
            src_features=self.datasets.get(split_name + "_features", None)
            )
      
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SimplificationDataset.collate_fn)

    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'val', is_train=False)
        return self.val_dataloader_object

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        parser.add_argument("--save_dir", type=str, default='simplification', help="Directory to save models.")
        parser.add_argument("--save_prefix", type=str, default='test', help="subfolder in save_dir for this model")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument("--from_pretrained", type=str, default=None,  help="Path to a checkpoint to load model weights but not training state")
        parser.add_argument("--num_sanity_val_steps", type=int, default=0,  help="Number of evaluation sanity steps to run before starting the training. Default: 0.")
        
        #data
        parser.add_argument("--train_source", type=str, default=None,  help="Path to the source train file.")
        parser.add_argument("--train_target", type=str, default=None, help="Path to the target train file.")
        parser.add_argument("--val_source", type=str, default=None, help="Path to the source validation file.")
        parser.add_argument("--val_target", type=str, default=None, help="Path to the target validation file.")
        parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file (to evaluate after training is finished).")
        parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (to evaluate after training is finished).")
        parser.add_argument("--src_lang", type=str, default=None, help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tgt_lang", type=str, default=None, help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tags_included", action='store_true', help="Text files already contain special tokens (language tags and </s>. Source:  src_tag seq, Target:  tgt_tag seq. Note: actual source sequence is seq src_tag </s>, will be changed internally after possibly clipping sequence to given max_length.")
        
        parser.add_argument("--train_features", type=str, default=None,  help="Path to alignment feature train file.")
        parser.add_argument("--val_features", type=str, default=None, help="Path to alignment feature validation file.")
        parser.add_argument("--test_features", type=str, default=None, help="Path to alignment feature test file (to evaluate after training is finished).")
        
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=512, help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--wandb", type=str, default=None, help="WandB project name to use if logging fine-tuning with WandB.")
        
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
        parser.add_argument("--global_attention_indices", type=int, nargs='+', default=[-1], required=False, help="List of indices of positions with global attention for longformer attention. Supports negative indices (-1 == last non-padding token). Default: [-1] == last source token (==lang_id) .")
        
        # Optimization params:
        #parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.00003, help="Initial learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations in percent of an epoch.")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--max_epochs", type=int, default=100000, help="Maximum number of epochs (will stop training even if patience for early stopping has not been reached).")
        parser.add_argument("--early_stopping_metric", type=str, default='rougeL', help="Metric to be used for early stopping: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu")
        parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
        parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum change in the monitored quantity to qualify as an improvement.")
        parser.add_argument("--lr_reduce_patience", type=int, default=8, help="Patience for LR reduction in Plateau scheduler.")
        parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Learning rate reduce factor for Plateau scheduler.")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--save_top_k", type=int, default=5, help="Number of best checkpoints to keep. Others will be removed.")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        
        ## inference params
        parser.add_argument("--decoded", type=str, default='decoded.out', help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=4, help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')
        
        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")
        
        # custom loss for attention on alignment spans
        parser.add_argument("--att_loss_mass", type=float, default=0.1, help="Percentage of attention mass that should be applied to aligned segments in source text")
        parser.add_argument("--att_loss_lambda", type=float, default=1.0, help="Lambda value for weighting cutsom attention loss")
        parser.add_argument("--att_loss_head", type=int, default=2, help="Attention head for wich attention on aligned spans is computed and incorporated into custom loss function. Note, parameter is zero-indexed (i.e. possible values [0-15]")
        parser.add_argument("--att_loss_layer", type=int, default=-1, help="Layer for wich attention on aligned spans is computed and incorporated into custom loss function (default = last layer (i.e. -1))")
        
        parser.add_argument("--factor_features", action='store_true', help="If specified, src features provided are combined with token embeddings (currently binary features are simply added to input embeddings, the features themselved are not embedded (cf. Sennrich and Haddow 2016))")
        

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
    
    # if alignment features provided load these as part of
    # the dataset
    if args.train_features and args.val_features and args.test_features:
        model.datasets = datasets.load_dataset('text', data_files={
            'train_source': args.train_source,
            'train_features': args.train_features,
            'train_target': args.train_target,
            'val_source': args.val_source,
            'val_features': args.val_features,
            'val_target': args.val_target,
            'test_source': args.test_source,
            'test_features': args.test_features,
            'test_target': args.test_target,
            })
    else:
        model.datasets = datasets.load_dataset('text', data_files={
            'train_source': args.train_source,
            'train_target': args.train_target,
            'val_source': args.val_source,
            'val_target': args.val_target,
            'test_source': args.test_source,
            'test_target': args.test_target,
            })
    
    if args.wandb:
        logger = WandbLogger(project=args.wandb)
    else:
        logger = TestTubeLogger(
            save_dir=args.save_dir,
            name=args.save_prefix,
            version=0  # always use version=0
        )

    print(args)

    model.lr_mode='max'
    # if args.early_stopping_metric == 'val_loss':
    if args.early_stopping_metric == 'vloss':
        model.lr_mode='min'
    early_stop_callback = EarlyStopping(monitor=args.early_stopping_metric, min_delta=args.min_delta, patience=args.patience, verbose=True, mode=model.lr_mode) # metrics: val_loss, bleu, rougeL
    
    custom_checkpoint_path = "checkpoint{{epoch:02d}}_{{{}".format(args.early_stopping_metric )
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
    model.model.save_pretrained(args.save_dir + "/" + args.save_prefix)
    model.tokenizer.save_pretrained(args.save_dir + "/" + args.save_prefix)
    trainer.fit(model)
    print("Training ended. Best checkpoint {} with {} {}.".format(model.best_checkpoint, model.best_metric, args.early_stopping_metric))
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="simplification")
    parser = Simplifier.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

