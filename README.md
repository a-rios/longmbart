# <p align=center>`MBart with Longformer windowed attention`</p>

Pretrained mBART model from huggingface with `Longformer` windowed attention in encoder (decoder has standard attention).

NOTE: [ats_models](https://github.com/a-rios/ats-models) is an updated version of this repository that does not rely on modifications of the transformer library itself, i.e. it runs with newer versions of their code. 

### Installation

```
    bash
    conda create --name longmbart python=3.8.5
    conda activate longmbart
    git clone https://github.com/ZurichNLP/longformer.git longmbart
    cd longmbart
    git checkout longmbart_hf4
    conda install cudatoolkit=your-cuda-version
    pip install .
    pip install -r requirements.txt
  ```
    
   To convert the huggingface mBART model, use scripts/convert_mbart_to_longformerencoderdecoder.py, for example:
   
   ``` 
   python $longformer_dir/scripts/convert_mbart_to_longformerencoderdecoder.py \
   --save_model_to path-to-save-new-model \
   --attention_window 512 \
   --reduce-to-vocab list-of-spm-pieces \
   --cache_dir path-to-huggingface-mbart \
   --add_language_tags de_A1 de_A2 de_B1 \
   --initialize_tags de_DE de_DE de_DE
   ```
    
   `--reduce-vocab-to-list` will resize the orginal pretrained model's vocabulary to the the pieces given in the list (text file, one piece per line). Pieces must be part of the pretrained sentencepiece model. 
   `--add_language_tags` will add new language tags, use `--initialize_tags` to specify which embeddings they should be initialized with, e.g. for German language levels, start with the German embeddings.
   
   To fine-tune the converted model, use `longformer/simplification.py`. If training on multilingual data, preprocess your data to contain the language tags and </s> like this:
   * source text: `src_lang source_sequene` (actual sequence in the model will be `source_sequence </s> src_lang`, reordering happens internally)
   * target text: `trg_lang target_sequence` 
   
 Example for fine-tuning (see `longformer/simplification.py` for all options):
   
```
python -m longformer.simplification \
--from_pretrained path-to-converted-model \
--tokenizer path-to-converted-model \
--save_dir path-to-save-fine-tuned-model \
--save_prefix "w512" \
--train_source path-to-source-train \
--train_target path-to-target-train \
--val_source path-to-source-dev \
--val_target path-to-target-dev \
--test_source path-to-source-test \
--test_target path-to-target-test \
--max_output_len max_target_length \
--max_input_len max_source_length \
--batch_size 1 \
--grad_accum 60 \
--num_workers 5 \
--gpus 1 \
--seed 222 \
--attention_dropout 0.1 \
--dropout 0.3 \
--attention_mode sliding_chunks \
--attention_window 512 \
--label_smoothing 0.2 \
--lr 0.00003 \
--val_every 1.0 \
--val_percent_check 1.0 \
--test_percent_check 1.0 \
--early_stopping_metric 'rougeL' \
--patience 10 \
--lr_reduce_patience 8 \
--lr_reduce_factor 0.5 \
--grad_ckpt \
--progress_bar_refresh_rate 10
```

Early stopping on one of these metrics: vloss, rouge1, rouge2, rougeL, rougeLsum, bleu (requires rouge_score and sacrebleu to be installed).
In a setting where translating from A to B, set `--src_lang A` and `--tgt_lang B` (input has no language tags), in a multilingual setting where source and target text already have language tags, use `--tags_included`. 

To translate with a fine-tuned model, use `longformer/simplify.py`, for example like this:
```
python -m longformer.simplify \
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
If only one target language, use `--tgt_lang` to set, if multiple languages, either give a reference file with tags (`tgt_lang target_sequence`) with `--tags_included` or just a list of target tags with `--target_tags` (one tag per line for each sample in `--test_source`).

### Citation
If you use code in this repository, please cite the following publication:

Annette Rios, Nicolas Spring, Tannon Kew, Marek Kostrzewa, Andreas Säuberli, Mathias Müller, and Sarah Ebling. 2021. [A New Dataset and Efficient Baselines for Document-level Text Simplification in German.](https://aclanthology.org/2021.newsum-1.16/) In Proceedings of the Third Workshop on New Frontiers in Summarization, pages 152–161, Online and in Dominican Republic. Association for Computational Linguistics.

Bibtex:
```
@inproceedings{rios-etal-2021-new,
    title = "A New Dataset and Efficient Baselines for Document-level Text Simplification in {G}erman",
    author = {Rios, Annette  and
      Spring, Nicolas  and
      Kew, Tannon  and
      Kostrzewa, Marek  and
      S{\"a}uberli, Andreas  and
      M{\"u}ller, Mathias  and
      Ebling, Sarah},
    booktitle = "Proceedings of the Third Workshop on New Frontiers in Summarization",
    month = nov,
    year = "2021",
    address = "Online and in Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.newsum-1.16",
    doi = "10.18653/v1/2021.newsum-1.16",
    pages = "152--161",
}
```
