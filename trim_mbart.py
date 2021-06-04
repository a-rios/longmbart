#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This code is adapted from AllenAI's Longformer:
    https://github.com/allenai/longformer/

Note: 
    Annette Rios (arios@cl.uzh.ch) initially adapted it for long-document simplication.
    Tannon Kew (kew@cl.uzh.ch) made minor changes for its
    application in the ReAdvisor project for short-document response
    generation.
    
Date: 04/06/2021

"""

import argparse
import logging
import os
from tqdm import tqdm
from collections import defaultdict
import sentencepiece.sentencepiece_model_pb2 as pb2

from transformers import MBartTokenizer, MBartConfig, MBartForConditionalGeneration

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def trim_embedding_matrix_of_pretrained_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    cache_dir,
    reduce_to_vocab,
    print_params,
):
    """
    trims embedding matrix based on vocab in `reduce_to_vocab` (optional) 
    """
    logger.info("loading pretrained models and config...")
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, cache_dir=cache_dir)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=1024, cache_dir=cache_dir)
    config = MBartConfig.from_pretrained(base_model, cache_dir=cache_dir)
    model.config = config

    if print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    ## reduce vocabulary of >250k to vocab given in reduce_to_vocab
    ## embedding matrix is model.shared.weight:torch.Size([250027, 1024])
    original_embed_weight = model.model.shared.weight
    original_vocab_size, model_size = original_embed_weight.shape
    
    # trim embed matrix
    if reduce_to_vocab is not None:
        with open(reduce_to_vocab, 'r') as f:
            keep_pieces = defaultdict()
            for piece in f.readlines():
                # check if this piece is actually in the spm vocab (some junk might not be)
                if tokenizer.sp_model.piece_to_id(piece.rstrip()) > 0:
                    keep_pieces[piece.rstrip()] = 1
                    #print(piece)
                #print(keep_pieces)

            num_special_tokens = 4 # <unk>, <s>, </s> <pad>
            new_vocab_size = len(keep_pieces) + num_special_tokens + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset + 1 # for mask token
            new_embed_weight = model.model.shared.weight.new_empty(new_vocab_size, model_size)
            ## need to reduce final_logits_bias too
            final_logits_bias_original = model.final_logits_bias.transpose(0,1) # (1, vocab_size)
            final_logits_bias_new = final_logits_bias_original.new_empty(new_vocab_size,1) # TODO: this seems to be just all zeros?

            ## keep order same as in original vocab.. iterate over 250k entries
            # `added_vocab_length` = length of special
            # mabrt's specual tokens used (27)
            added_vocab_length = len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset + 1
            base_vocab_length_original = original_vocab_size - added_vocab_length
            base_vocab_length_new = len(keep_pieces) + num_special_tokens

            ## delete ununsed entries from sentencepiece model of the tokenizer and save the new ModelProto
            pb2_model = pb2.ModelProto()
            pb2_model.ParseFromString(open(os.path.join(cache_dir, "sentencepiece.bpe.model"), 'rb').read())
            indices_to_remove = []
            count=0

            ## from transformers.tokenization_xlm_roberta.py -> self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
            ## sentencepiece model: 0 = <unk>, 1 = <s>, 2 = </s> -> need to copy first 4 rows in embedding matrix and then shift spm ids by +1
            for i in range(0,4):
                piece_embed = original_embed_weight[i]
                piece_final_logits_bias = final_logits_bias_original[i]
                new_embed_weight[i] = piece_embed
                final_logits_bias_new[i] = piece_final_logits_bias

            new_embed_iter = 4
            for embed_iter, spm_iter in zip(range(4,base_vocab_length_original), range(3,base_vocab_length_original-1)): # full vocab size with (!) the added tokens, 250027 | 

                if new_embed_iter > base_vocab_length_new:
                    print("ran out of space at position {} in new matrix with vocab size {}".format(j, base_vocab_length_new))
                    exit(0)

                piece = pb2_model.pieces[spm_iter].piece
                #print("embed iter: {}, spm iter {}, piece {}".format(embed_iter, spm_iter, piece))
                if piece in keep_pieces.keys():
                    count +=1
                    ### get embedding
                    piece_embed = original_embed_weight[embed_iter]
                    piece_final_logits_bias = final_logits_bias_original[embed_iter]
                    new_embed_weight[new_embed_iter] = piece_embed
                    final_logits_bias_new[new_embed_iter] = piece_final_logits_bias
                    #print("id : {}, piece {} ".format(new_embed_iter, piece))
                    new_embed_iter +=1
                else:
                    indices_to_remove.append(spm_iter)
                    #print(piece)

            ##total count matched  59586
            ##len vocabs to keep  59586 + special tokens 4
            ##new vocab size  59617
            print("total count matched ", count) #
            print("len vocabs to keep {} + special tokens {}".format(len(keep_pieces.keys()), num_special_tokens))
            print("new vocab size ", new_vocab_size)

            # breakpoint()
            # check ids in reduced spm model
            removed =0
            for i in tqdm(indices_to_remove):
                position = i-removed
                # print("deleting ", pb2_model.pieces[position].piece)
                del pb2_model.pieces[position]
                removed +=1

            ## fill in additional vocab positions (language ids etc)
            # breakpoint()
            for i in range(added_vocab_length):
                new_embed_weight[base_vocab_length_new+i] = original_embed_weight[base_vocab_length_original+i]
                
                final_logits_bias_new[base_vocab_length_new+i] = final_logits_bias_original[base_vocab_length_original+i]
                #print("position in new tensor ", base_vocab_length_new+i)
                #print("position in old tensor ", base_vocab_length_original+i)
                #print("embed ", new_embed_weight[base_vocab_length_new+i])

            assert len(torch.nonzero(final_logits_bias_new, as_tuple=False)) == 0, "final logits bias must be all zeros for fine-tuning but found non zero values. Hint: check update to new_embed_weights and final_logits_bias_new."

            

            model.model.shared.weight.data = new_embed_weight
            model.final_logits_bias.data = final_logits_bias_new.transpose(0,1) # swap dimensions back to (1, vocab_size

            with open(os.path.join(save_model_to, 'reduced.spm.model'), 'wb') as f:
                f.write(pb2_model.SerializeToString())

            tokenizer.init_kwargs['vocab_file'] = os.path.join(save_model_to, "reduced.spm.model")
            tokenizer.vocab_file = os.path.join(save_model_to, "reduced.spm.model")
            logger.info(f"saving reduced tokenizer vocabulary with size {new_vocab_size}")
            tokenizer.save_vocabulary(save_model_to)
            config.vocab_size = new_vocab_size
    
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    logger.info('saving tokenizer')
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer

def read_in_symbols(infile):
    symbols = set()
    with open(infile, 'r', encoding='utf8') as inf:
        for line in inf:
            line = line.strip().split()
            symbols.update(line)
    return sorted(list(symbols))

def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the tokenizer'
    )
    parser.add_argument(
        '--save_model_to',
        type=str,
        required=True,
        help='The path to save the converted model'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        help='where to save original model'
    )
    parser.add_argument(
        '--reduce-to-vocab',
        type=str,
        help='List of subword entries to keep in new model (one token per line).'
    )
    parser.add_argument(
        '--add_special_tokens',
        type=str, help="path to text file containing special tokens to add to the tokenizer so that they won't be split during tokenization. Note, embeddings are initialised randomly."
    )
    parser.add_argument("--print-params",
                        action='store_true',
                        help="Print parameter names and shapes.")               
    parser.add_argument("--verbose",
                        type=int, default=1, help="Levels of verbosity affect what is tested/shown after converting model")
                        
    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    if args.add_special_tokens is not None:
        user_special_tokens = read_in_symbols(args.add_special_tokens)
    else:
        user_special_tokens = None

    trim_embedding_matrix_of_pretrained_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        cache_dir=args.cache_dir,
        reduce_to_vocab=args.reduce_to_vocab,
        print_params=args.print_params,
    )
        
    tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)
    model = MBartForConditionalGeneration.from_pretrained(args.save_model_to)
    print("loaded tokenizer with len ", len(tokenizer.sp_model))
    print(model.config)

    if user_special_tokens:
        # https://huggingface.co/transformers/internal/tokenization_utils.html?highlight=add_tokens
        num_added_toks = tokenizer.add_tokens(user_special_tokens, special_tokens=False)
        print('added', num_added_toks, 'special tokens to tokenizer')
        model.resize_token_embeddings(len(tokenizer))

        print("saving tokenizer with new tags")
        tokenizer.save_pretrained(args.save_model_to)
        print("saving model with new tags")
        model.save_pretrained(args.save_model_to)

    print(tokenizer.special_tokens_map)
    print(tokenizer.id_to_lang_code)
    print(tokenizer.lang_code_to_id)

    if args.verbose > 0:
        # TXT1 = "en_XX this is a test."
        # TXT2 = "de_DE Noch ein Test."

        # TXT3 = "en_XX <5> <restaurant> this is a test <endtitle> still a test."
        # TXT4 = "de_DE <1> <hotel> Noch ein Test <endtitle>."
        
        TXT1 = "this is a test. en_XX"
        TXT2 = "Noch ein Test. de_DE"

        TXT3 = "<5> <restaurant> this is a test <endtitle> still a test. en_XX"
        TXT4 = "<1> <hotel> Noch ein Test <endtitle>. de_DE"

        batch: dict = tokenizer.prepare_seq2seq_batch(src_texts=[TXT1, TXT2], max_length=512, truncation=False, padding="max_length", return_tensors="pt", tags_included=True)
        print(batch)
        decoder_start_token_ids = [tokenizer.lang_code_to_id["en_XX"], tokenizer.lang_code_to_id["de_DE"]]
        decoder_start_token_ids = torch.tensor(decoder_start_token_ids)
        print("decoder start ids ", decoder_start_token_ids)
        translated_tokens = model.generate(**batch, decoder_start_token_ids=decoder_start_token_ids, use_cache=True, num_beams=2)
        breakpoint()
        translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        for translation in translations:
            if not len(translation.strip()):
                print('[!] Looks like something went wrong - translation is empty.')
            else:
                print(translation)

        batch: dict = tokenizer.prepare_seq2seq_batch(src_texts=[TXT3, TXT4], max_length=512, truncation=False, padding="max_length", return_tensors="pt", tags_included=True)
        print(batch)
        decoder_start_token_ids = [tokenizer.lang_code_to_id["en_XX"], tokenizer.lang_code_to_id["de_DE"]]
        decoder_start_token_ids = torch.tensor(decoder_start_token_ids)
        print("decoder start ids ", decoder_start_token_ids)
        translated_tokens = model.generate(**batch, decoder_start_token_ids=decoder_start_token_ids, use_cache=True, num_beams=2)
        translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        for translation in translations:
            if not len(translation.strip()):
                print('[!] Looks like something went wrong - translation is empty.')
            else:
                print(translation)


if __name__ == "__main__":
    main()
    

