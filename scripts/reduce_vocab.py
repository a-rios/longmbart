import argparse
import logging
import os
import copy
from collections import defaultdict
import sentencepiece.sentencepiece_model_pb2 as pb2
import sentencepiece as spm

from transformers import MBartTokenizer

from transformers import MBartForConditionalGeneration, MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right
import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    cache_dir,
    reduce_to_vocab,
    print_params
):
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, cache_dir=cache_dir)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
    tokenizer.save_vocabulary(cache_dir)
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
    
    if reduce_to_vocab is not None:
        with open(reduce_to_vocab, 'r') as f:
            keep_pieces = defaultdict()
            for piece in f.readlines():
                # check if this piece is actually in the spm vocab (some junk might not be)
                if tokenizer.sp_model.piece_to_id(piece.rstrip()) > 0:
                    keep_pieces[piece.rstrip()] = 1
                    #print(piece)
                #print(keep_pieces)

            ##TODO clean up pieces vocabulary more
            num_special_tokens = 4 # <unk>, <s>, </s> <pad>
            new_vocab_size = len(keep_pieces) +num_special_tokens + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset + 1 # for mask token
            new_embed_weight = model.model.shared.weight.new_empty(new_vocab_size, model_size)
            ## need to reduce final_logits_bias too
            final_logits_bias_original = model.final_logits_bias.transpose(0,1) # (1, vocab_size)
            final_logits_bias_new = final_logits_bias_original.new_empty(new_vocab_size,1) # TODO: this seems to be just all zeros?

            ## keep order same as in original vocab.. iterate over 250k entries
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

            # check ids in reduced spm model
            removed =0
            for i in tqdm(indices_to_remove):
                position = i-removed
                #print("deleting ", pb2_model.pieces[position].piece)
                del pb2_model.pieces[position]
                removed +=1

            ## fill in additional vocab positions (language ids etc)
            for i in range(added_vocab_length):
                new_embed_weight[base_vocab_length_new+i] = original_embed_weight[base_vocab_length_original+i]
                final_logits_bias_new[base_vocab_length_new+i] = final_logits_bias_original[base_vocab_length_original+i]
                #print("position in new tensor ", base_vocab_length_new+i)
                #print("position in old tensor ", base_vocab_length_original+i)
                #print("embed ", new_embed_weight[base_vocab_length_new+i])

            model.model.shared.weight.data = new_embed_weight
            model.final_logits_bias.data = final_logits_bias_new.transpose(0,1) # swap dimensions back to (1, vocab_size

            with open(os.path.join(save_model_to, 'reduced.spm.model'), 'wb') as f:
                f.write(pb2_model.SerializeToString())

            tokenizer.init_kwargs['vocab_file'] = os.path.join(save_model_to, "reduced.spm.model")
            tokenizer.vocab_file = os.path.join(save_model_to, "reduced.spm.model")
            tokenizer.save_vocabulary(save_model_to)
            #print("saving tokenizer with len ", len(tokenizer.sp_model))
            #tokenizer.save_pretrained(save_model_to)
            config.vocab_size = new_vocab_size
    
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    print("saving tokenizer")
    tokenizer.save_pretrained(save_model_to)
    #print(model)
    return model, tokenizer


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
    parser.add_argument("--print-params",
                        action='store_true',
                        help="Print parameter names and shapes.")

    parser.add_argument(
        '--add_language_tags',
        type=str, nargs='+',
        help='List of additional language tags (will replace tags given with --replace_tags and initialize with embeddings given with --initialize_tags).'
    )
    parser.add_argument(
        '--initialize_tags',
        type=str, nargs='+',
        help='Initialize new language tags with embeddings of these tags.'
    )
                    
    parser.add_argument("--verbose",
                        type=int, default=1, help="Levels of verbosity affect what is tested/shown after converting model")
                        
    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    if args.add_language_tags is not None:
        assert args.initialize_tags is not None, "Need --initialize_tags to add new language tags"
        assert len(args.add_language_tags) == len(args.initialize_tags), "Need same number of values for --add_language_tags and --initialize_tags but got %i and %i" %(len(args.add_language_tags), len(args.initialize_tags))


    create_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        cache_dir=args.cache_dir,
        reduce_to_vocab=args.reduce_to_vocab,
        print_params=args.print_params
    )

    tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)
    model = MBartForConditionalGeneration.from_pretrained(args.save_model_to)
    print("loaded tokenizer with len ", len(tokenizer.sp_model))

    if args.add_language_tags is not None:
        embed_weight = model.model.shared.weight # (vocab, dim)
        print(embed_weight.shape)
        ## need to reduce final_logits_bias too
        final_logits_bias = model.final_logits_bias.transpose(0,1) # (1, vocab_size)
        #print("new model, logits bias ", final_logits_bias)
        #print("new model, logits bias non zero", final_logits_bias.nonzero())

        print(tokenizer._additional_special_tokens)
        print("tokenizer orig len ", tokenizer.vocab_size)
        tokenizer.add_tokens(args.add_language_tags)
        print("tokenizer len ", tokenizer.vocab_size)

        for (new_tag, init_tag) in zip(args.add_language_tags, args.initialize_tags):
            init_tag_id = tokenizer.lang_code_to_id[init_tag]
            print("init_tag_id ", init_tag_id)
            init_embed = model.model.shared.weight[init_tag_id].unsqueeze(0)
            embed_weight = torch.cat((embed_weight, init_embed), dim=0)
            init_bias = final_logits_bias[init_tag_id].unsqueeze(dim=0)
            final_logits_bias = torch.cat((final_logits_bias, init_bias), dim=0)
            print("added ", new_tag)
            print("tag embedding shape ", init_embed.shape)
            print("embedding matrix shape ", embed_weight.shape)

        model.final_logits_bias.data = final_logits_bias.transpose(0,1)
        model.model.shared.weight.data = embed_weight
        model.config.vocab_size = embed_weight.shape[0]

        print("saving tokenizer with new tags")
        tokenizer.save_pretrained(args.save_model_to)
        print("saving model with new tags")
        model.save_pretrained(args.save_model_to)

    #reload tokenizer with extended tags
    tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)
    print("special tokens map ", tokenizer.special_tokens_map)
    print("id-to-lang-code ",tokenizer.id_to_lang_code)
    print("lang-code-to-id", tokenizer.lang_code_to_id)

    ## check embeddings
    if args.add_language_tags is not None and args.initialize_tags is not None:
        for new_tag, init_tag in zip(args.add_language_tags, args.initialize_tags):
            print("original language embedding for {}: {}".format(init_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(init_tag)]))
            print("initialized {} with embedding: {}".format(new_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(new_tag)]))


    TXT = "de_DE Das ist ein Test."
    TXT2 = "de_DE Noch ein Test."

    batch: dict = tokenizer.prepare_seq2seq_batch(src_texts=[TXT, TXT2], max_length=1024, truncation=False, padding="max_length", return_tensors="pt", tags_included=True)
    print(batch)
    translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["de_A1"], use_cache=True, num_beams=2)
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(translation)


if __name__ == "__main__":
    main()
    

