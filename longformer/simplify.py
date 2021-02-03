import os
import argparse
import random
import numpy as np

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
from transformers.models.bart.modeling_bart import shift_tokens_right
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart
from longformer.longformer_encoder_decoder_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig
import datasets
import collections

from . import simplification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)





class SimplificationDatasetForInference(Dataset):
    def __init__(self, inputs, reference, name, tokenizer, max_input_len, max_output_len, src_lang, tgt_lang):
        self.inputs = inputs
        self.reference = reference
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]['text']
        reference = None
        if self.reference is not None:
            reference = self.reference[idx]['text']
        sample = self.tokenizer.prepare_seq2seq_batch(src_texts=[source], src_lang=self.src_lang, tgt_lang=self.tgt_lang , max_length=self.max_input_len, max_target_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt") # TODO move this to _get_dataloader, preprocess everything at once? 
        ## changed set_tgt_lang_special_tokens in transformers.tokenization_mbart to prefix the target id instead of adding it at the end. See https://arxiv.org/pdf/2001.08210.pdf, "A language id symbol <LID> is usedas the initial token to predict the sentence." 

        input_ids = sample['input_ids'].squeeze()
      
        return input_ids, reference

    @staticmethod
    def collate_fn(batch):
        
        pad_token_id = 1
    
        input_ids, ref = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, ref


class InferenceSimplifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
       
        self.src_lang = "de_DE"
        self.tgt_lang = "de_DE" ## TODO make this more generic
        
        self.config = MLongformerEncoderDecoderConfig.from_pretrained(self.args.model_path)
        self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)
        
        self.max_input_len = self.args.max_input_len if self.args.max_input_len is not None else self.config.max_encoder_position_embeddings
        self.max_output_len = self.args.max_output_len if self.args.max_output_len is not None else self.config.max_decoder_position_embeddings 
        self.test_dataloader_object = None

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        if isinstance(self.model, MLongformerEncoderDecoderForConditionalGeneration):
            attention_mask[:, -1] = 2  # put global attention on last token (target language tag)
            if self.config.attention_mode == 'sliding_chunks':
                half_padding_mod = self.model.config.attention_window[0]
            elif self.config.attention_mode == 'sliding_chunks_no_overlap':
                half_padding_mod = self.model.config.attention_window[0] / 2
            else:
                raise NotImplementedError
            input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
                input_ids, attention_mask, half_padding_mod, self.tokenizer.pad_token_id)
        return input_ids, attention_mask


    
    def test_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        input_ids, ref  = batch
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.max_input_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang]) # TODO: mBART source: X eso src_tag -> target: trg_tag X eos (no bos). Original fairseq: eos = language tag. Use </s> for fine-tuning or set eos_token_id to self.tokenizer.lang_code_to_id[self.tgt_lang]?

        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        with open(self.args.translation, 'a') as f:
            for sample in generated_str:
                f.write(sample + "\n")
        
        if self.args.test_target is not None:
            gold_str = [ref[0]]
            scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
            rouge1 = rouge2 = rougel = rougelsum = 0.0
            for ref, pred in zip(gold_str, generated_str):
                score = scorer.score(ref, pred)
                rouge1 += score['rouge1'].fmeasure
                rouge2 += score['rouge2'].fmeasure
                rougel += score['rougeL'].fmeasure
                rougelsum += score['rougeLsum'].fmeasure
            rouge1 /= len(generated_str)
            rouge2 /= len(generated_str)
            rougel /= len(generated_str)
            rougelsum /= len(generated_str)
            bleu = sacrebleu.corpus_bleu(generated_str, [gold_str])
        
            return {'rouge1': rouge1,
                    'rouge2': rouge2,
                    'rougeL': rougel,
                    'rougeLsum':  rougelsum, 
                    'bleu' :  bleu.score,
                    'decoded' : generated_str}
        else:
            return {'decoded' : generated_str}

    def test_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True
    
        if self.args.test_target is not None:
            names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
            metrics = []
            for name in names:
                scores = [x[name] for x in outputs]
                metric = sum(scores)/len(scores)
                metrics.append(metric)
            logs = dict(zip(*[names, metrics]))
            print("Evaluation on provided reference [{}] ".format(self.args.test_target))
            print(logs)


    def forward(self):
        pass
    
    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        reference = None
        if self.args.test_target is not None:
            reference = self.datasets[split_name + "_target"]
        dataset = SimplificationDatasetForInference(inputs=self.datasets[split_name + "_source"], reference=reference , name=split_name, tokenizer=self.tokenizer,
                                       max_input_len=self.max_input_len, max_output_len=self.max_output_len, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
      
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SimplificationDatasetForInference.collate_fn)

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
        parser.add_argument("--model_path", type=str, help="Path to the checkpoint directory or model name")
        parser.add_argument("--checkpoint_name", type=str, help="Checkpoint in model_path to use.")
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        
        #data
        parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file.")
        parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (optional, if given, will output rouge and bleu).")
        parser.add_argument("--max_input_len", type=int, default=256, help="maximum num of wordpieces, if unspecified, will use number of encoder positions from model config.")
        parser.add_argument("--max_output_len", type=int, default=512, help="maximum num of wordpieces, if unspecified, will use number of decoder positions from model config.")
        
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--gpus", type=int, default=-1, help="Number of gpus. 0 for CPU")
        
        
        ## inference params
        parser.add_argument("--translation", type=str, default='decoded.out', help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=4, help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')
        
        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")
        
        return parser


def main(args):
    checkpoint_path=os.path.join(args.model_path, args.checkpoint_name)
    simplifier = InferenceSimplifier(args)
    
    cp = torch.load(checkpoint_path)
    simplifier.model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.model_path)
   
    simplifier.load_state_dict(cp["state_dict"])
    #simplifier.load_from_checkpoint(checkpoint_path, args) ## does not work ("unexpected keys")
     
    if args.print_params:
        for name, param in simplifier.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)
    
    if args.test_target is not None:
        simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source, 'test_target': args.test_target })
    else:
        simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source })
    
    logger = TestTubeLogger(
        save_dir=".",
        name="decode.log",
        version=0  # always use version=0
    )

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if torch.cuda.is_available() else None,
                         replace_sampler_ddp=False,
                         limit_test_batches=args.test_percent_check,
                         logger=logger,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=32 if args.fp32 else 16, amp_level='O2'
                         )
    trainer.test(simplifier)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="simplification")
    parser = InferenceSimplifier.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

