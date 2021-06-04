# MBart for hospitality review response generation

Pretrained mBART model from huggingface with a trimmed
embedding matrix to allow for efficient fine-tuning on a
regular GPU.

The scripts in this branch are adapted from
https://github.com/a-rios/longformer-private, which extend
the model further by replacing regular full self-attention
in the encoder with Longformer windowed attention. This
additional change to the underlying model is only
required for very long input documents.

### Installation

```
    conda create --name hospo_respo_mbart python=3.8.5
    conda activate hospo_respo_mbart
    git clone https://github.com/ZurichNLP/longformer.git mbart
    cd mbart
    git checkout readvisor
    conda install cudatoolkit=your-cuda-version # if GPU instance
    pip install .
    pip install -r requirements.txt
```

### Bash script wrappers

The bash scripts 

- `run_mbart_conversion.sh`
- `run_finetuning.sh`
- `run_inference.sh`

provide simple wrappers to tie the distinct
steps together. To use these, simply adjust the absolute
directory paths at the top of each script and run using the
example call provided in each script.


### Model setup

To trim the embedding matrix of the huggingface mBART model, use `trim_mbart.py`, for example:
   
   ``` 
   python trim_mbart.py \
    --base_model facebook/mbart-large-cc25 \
    --save_model_to path-to-save-new-model \
    --reduce-to-vocab list-of-spm-pieces \
    --cache_dir path-to-huggingface-mbart \
    --add_special_tokens list-of-special-tokens
  ```

   `--reduce-vocab-to-list` will resize the orginal pretrained model's vocabulary to the the pieces given in the list (text file, one piece per line). Pieces must be part of the pretrained sentencepiece model. 
   `--add_special_tokens` will add extend the model's
   vocabulary with new tokens (by default, these are added with `special_tokens=False`).

### Fine-tuning

   To fine-tune the trimmed model, use `train.py`. If training on multilingual data, preprocess your data to contain the language tags and </s> like this:
   * source text: `src_lang source_sequene` (actual sequence in the model will be `source_sequence </s> src_lang`, reordering happens internally)
   * target text: `trg_lang target_sequence` 
   
 Example for fine-tuning (see `train.py` for all options):
   
```

pretrained=path-to-trimmed-model
data=path-to-raw-data-dir
SRC=source-suffix
TGT=target-suffix

mkdir $pretrained/ft/name-of-model/

python train.py \
--from_pretrained $pretrained \
--tokenizer $pretrained \
--save_dir $pretrained/ft \
--save_prefix name-of-model \
--train_source $data/train.$SRC --train_target $data/train.$TGT \
--val_source $data/valid.$SRC --val_target $data/valid.$TGT \
--test_source $data/test.$SRC --test_target $data/test.$TGT \
--tags_included \
--max_input_len 512 --max_output_len 400 \
--batch_size 8 \
--grad_accum 5 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric vloss \
--patience 5 --max_epochs 20 \
--lr_reduce_patience 8 --lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 1 \
--save_top_k 5
```

Early stopping on one of these metrics: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu (requires rouge_score and sacrebleu to be installed).

In a setting where translating from A to B, set `--src_lang A` and `--tgt_lang B` (input has no language tags), in a multilingual setting where source and target text already have language tags, use `--tags_included`. 

### Inference

To run inference with a fine-tuned model, use `inference.py`, for example like this:
```
python inference.py \
--model_path path-to-fine-tuned-model \
--checkpoint "checkpointepoch=name-of-checkpoint" \
--tokenizer path-to-fine-tuned-model \
--translation output-file \
--test_source path-to-source \
--test_target path-to-reference \
--max_output_len max_target_length \
--max_input_len max_source_length \
--batch_size 2 \
--num_workers 5 \
--gpus 1 \
--beam_size 6 \
--progress_bar_refresh_rate 1 \
--tags_included
```

Reference file is optional, if given, will print evaluation metrics (rouge1, rouge2, rougeL, rougeLsum, bleu). 
If only one target language, use `--tgt_lang` to set, if
multiple languages, either give a reference file with tags
(`tgt_lang target_sequence`) with `--tags_included` or just
a list of target tags with `--target_tags` (one tag per line
for each sample in `--test_source`).

Alternatively, use --infer_tags if the target language tag
matches the source language tag and the source language tag
is included in the input file.

# Acknowledgements

This implemenation is based on the scripts from AllenAI's
Longformer implementation and Annette Rios (UZH CL) who
adapted mBART with Longformer attention.