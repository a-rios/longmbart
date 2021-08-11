import argparse
import logging
import os
import copy
from collections import defaultdict
import sentencepiece.sentencepiece_model_pb2 as pb2
import sentencepiece as spm

from transformers import MBartTokenizer

from transformers import MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart
from longformer.longformer_encoder_decoder_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    attention_window,
    max_pos,
    cache_dir,
    reduce_to_vocab,
    print_params
):
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, cache_dir=cache_dir)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos, cache_dir=cache_dir)
    tokenizer.save_vocabulary(cache_dir)
    config = MLongformerEncoderDecoderConfig.from_pretrained(base_model, cache_dir=cache_dir)
    model.config = config

    if print_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['MLongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_pos == config.max_position_embeddings + 2

    config.max_encoder_position_embeddings = max_pos
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings ## will be filled in from_pretrained with default value 1024, will initialize model.encoder.embed_positions.weight as (1026, 1024) instead of max_encoder_position_embeddings --> changed in line 630 of transformers.models.mbart.modeling_mbart for encoder, line 778 for decoder, also changed init in lines 640 (encoder) and 787 (decoder), as max length is again read from config instead of using defined values :/^
    max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_pos >= current_max_pos

    # allocate a larger position embedding matrix for the encoder
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

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
            for i in indices_to_remove:
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

    # allocate a larger position embedding matrix for the decoder
    # new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_pos - 2
    # while k < max_pos - 1:
    #     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
    #     k += step
    # model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForBart(config, layer_id=i)

        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
        longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

        longformer_self_attn_for_bart.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_bart
    
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
        '--attention_window',
        type=int,
        default=512,
        help='attention window size for longformer self attention (one sided)'
    )
    parser.add_argument(
        '--max_pos',
        type=int,
        default=4096 * 4,
        help='maximum encoder positions'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        required=True,
        help='where to save original model'
    )
    parser.add_argument(
        '--reduce-to-vocab',
        type=str,
        help='List of subword entries to keep in new model (one token per line).'
    )
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
    parser.add_argument("--print-params",
                        action='store_true',
                        help="Print parameter names and shapes.")
                    
    parser.add_argument("--verbose",
                        type=int, default=1, help="Levels of verbosity affect what is tested/shown after converting model")
                        
    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    if args.add_language_tags is not None:
        assert args.initialize_tags is not None, "Need --initialize_tags to add new language tags"
        assert len(args.add_language_tags) == len(args.initialize_tags), "Need same number of values for --add_language_tags and --initialize_tags but got %i and %i" %(len(args.add_language_tags), len(args.initialize_tags))

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        attention_window=args.attention_window,
        max_pos=args.max_pos,
        cache_dir=args.cache_dir,
        reduce_to_vocab=args.reduce_to_vocab,
        print_params=args.print_params
    )
    tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)

    model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.save_model_to)
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

    print("special tokens map ", tokenizer.special_tokens_map)
    print("id-to-lang-code ",tokenizer.id_to_lang_code)
    print("lang-code-to-id", tokenizer.lang_code_to_id)

    ## check embeddings
    if args.add_language_tags is not None and args.initialize_tags is not None:
        for new_tag, init_tag in zip(args.add_language_tags, args.initialize_tags):
            print("original language embedding for {}: {}".format(init_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(init_tag)]))
            print("initialized {} with embedding: {}".format(new_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(new_tag)]))

    if args.verbose > 0:
        tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)
        TXT = "de_DE Das ist ein Test."
        TXT2 = "de_DE Noch ein Test."
        #print("string in pieces ", tokenizer.sp_model.encode(TXT, out_type=str))
        #print("string in ids ", tokenizer.sp_model.encode(TXT, out_type=int))
        TXT3 = "en_XX this is a test."
        TXT4 = "es_XX otro ejemplo."

        if args.verbose > 1:
            ## input = src_lang sequence, target = tgt_lang sequence
            tgt_texts = [TXT3, TXT4]
            batch: dict = tokenizer.prepare_seq2seq_batch(src_texts=[TXT, TXT2], max_length=2048, truncation=False, padding="max_length", return_tensors="pt", tags_included=True)
            print(batch)

            #decoder_start_token_ids = [ tokenizer.lang_code_to_id[sample.split(' ')[-1]] for sample in tgt_texts]
            decoder_start_token_ids = [tokenizer.lang_code_to_id["de_A1"], tokenizer.lang_code_to_id["de_B1"]]
            decoder_start_token_ids = torch.tensor(decoder_start_token_ids)
            print("decoder start ids ", decoder_start_token_ids)
            translated_tokens = model.generate(**batch, decoder_start_token_ids=decoder_start_token_ids, use_cache=True, num_beams=2)
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(translation)

            batch: dict = tokenizer.prepare_seq2seq_batch(src_texts=[TXT, TXT2], src_lang="de_DE", max_length=2048, truncation=False, padding="max_length", return_tensors="pt",  tags_included=True)
            translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["es_XX"], use_cache=True, num_beams=2, max_length=20)
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(translation)

if __name__ == "__main__":
    main()
    

