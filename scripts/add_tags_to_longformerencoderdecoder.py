import argparse
import logging
import os

from transformers import MBartTokenizer

from longformer.longformer_encoder_decoder_mbart import MLongformerEncoderDecoderForConditionalGeneration
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Add language tags to existing longmbart model")
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='The path to save the converted model'
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
    parser.add_argument(
        '--fix_added_token_ids',
        action='store_true',
        help='EXPERIMENTAL: set this flag to avoid non-consecutive added tokens'
    )
    parser.add_argument("--overwrite", action="store_true", help="EXPERIMENTAL: set this flag to overwrite the embeddings of existing tags instead of adding new ones.")
    parser.add_argument("--checkpoint", nargs="+", default=[], type=str, help="EXPERIMENTAL: checkpoints to add tags to")
    parser.add_argument("--verbose",
                        type=int, default=1, help="Levels of verbosity affect what is tested/shown after converting model")

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if args.add_language_tags is not None:
        assert args.initialize_tags is not None, "Need --initialize_tags to add new language tags"
        assert len(args.add_language_tags) == len(args.initialize_tags), "Need same number of values for --add_language_tags and --initialize_tags but got %i and %i" %(len(args.add_language_tags), len(args.initialize_tags))

    tokenizer = MBartTokenizer.from_pretrained(args.model_dir)
    model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.model_dir)
    print("loaded tokenizer with len ", len(tokenizer.sp_model))
    checkpoints = [(path, torch.load(path, map_location=torch.device('cpu'))) for path in args.checkpoint]
    print("adding tag to checkpoints: ", [name for name, _ in checkpoints])

    if args.add_language_tags is not None:
        embed_weight = model.model.shared.weight # (vocab, dim)
        print(embed_weight.shape)
        # need to reduce final_logits_bias too
        final_logits_bias = model.final_logits_bias.transpose(0,1) # (1, vocab_size)
        #print("new model, logits bias ", final_logits_bias)
        #print("new model, logits bias non zero", final_logits_bias.nonzero())
        checkpoint_embed_weights = [
            checkpoint["state_dict"]["model.model.shared.weight"]
            for _, checkpoint in checkpoints
        ]
        checkpoint_final_logits_biases = [
            checkpoint["state_dict"]["model.final_logits_bias"].transpose(0,1)
            for _, checkpoint in checkpoints
        ]

        print(tokenizer._additional_special_tokens)
        print("tokenizer orig len ", tokenizer.vocab_size)
        tokenizer.add_tokens(args.add_language_tags)
        print("tokenizer len ", tokenizer.vocab_size)

        # check that tags are already present when overwriting embeddings
        # raises KeyError if this is not the case
        if args.overwrite:
            for new_tag in args.add_language_tags:
                _ = tokenizer.lang_code_to_id[new_tag]

        for (new_tag, init_tag) in zip(args.add_language_tags, args.initialize_tags):
            init_tag_id = tokenizer.lang_code_to_id[init_tag]
            new_tag_id = tokenizer.lang_code_to_id[new_tag] if args.overwrite else None
            print(init_tag)
            print("init_tag_id ", init_tag_id)
            print(new_tag)
            print("new_tag_id ", new_tag_id)
            init_embed = model.model.shared.weight[init_tag_id].unsqueeze(0)
            checkpoint_init_embeds = [
                checkpoint["state_dict"]["model.model.shared.weight"][init_tag_id].unsqueeze(0)
                for _, checkpoint in checkpoints
            ]
            if not args.overwrite:
                embed_weight = torch.cat((embed_weight, init_embed), dim=0)
                for i, c_embed_weight in enumerate(checkpoint_embed_weights):
                    checkpoint_embed_weights[i] = torch.cat((c_embed_weight, checkpoint_init_embeds[i]), dim=0)
            else:
                with torch.no_grad():
                    print(embed_weight[new_tag_id])
                    embed_weight[new_tag_id] = init_embed
                    for i in range(len(checkpoint_embed_weights)):
                        checkpoint_embed_weights[i][new_tag_id] = checkpoint_init_embeds[i]
                    print(embed_weight[new_tag_id])
            init_bias = final_logits_bias[init_tag_id].unsqueeze(dim=0)
            checkpoint_init_biases = [
                checkpoint["state_dict"]["model.final_logits_bias"].transpose(0,1)[init_tag_id].unsqueeze(dim=0)
                for _, checkpoint in checkpoints
            ]
            if not args.overwrite:
                final_logits_bias = torch.cat((final_logits_bias, init_bias), dim=0)
                for i, c_final_logits_bias in enumerate(checkpoint_final_logits_biases):
                    checkpoint_final_logits_biases[i] = torch.cat((c_final_logits_bias, checkpoint_init_biases[i]), dim=0)
            else:
                with torch.no_grad():
                    final_logits_bias[new_tag_id] = init_bias
                    for i in range(len(checkpoint_final_logits_biases)):
                        checkpoint_final_logits_biases[i][new_tag_id] = checkpoint_init_biases[i]
            print("added ", new_tag)
            print("tag embedding shape ", init_embed.shape)
            print("embedding matrix shape ", embed_weight.shape)

        model.final_logits_bias.data = final_logits_bias.transpose(0,1)
        model.model.shared.weight.data = embed_weight
        model.config.vocab_size = embed_weight.shape[0]

        for (path, checkpoint), checkpoint_embed_weight, checkpoint_final_logits_bias in zip(
            checkpoints, checkpoint_embed_weights, checkpoint_final_logits_biases
        ):
            checkpoint["state_dict"]["model.model.shared.weight"] = checkpoint_embed_weight
            checkpoint["state_dict"]["model.lm_head.weight"] = checkpoint_embed_weight
            checkpoint["state_dict"]["model.model.encoder.embed_tokens.weight"] = checkpoint_embed_weight
            checkpoint["state_dict"]["model.model.decoder.embed_tokens.weight"] = checkpoint_embed_weight
            checkpoint["state_dict"]["model.final_logits_bias"] = checkpoint_final_logits_bias.transpose(0,1)

        if args.fix_added_token_ids:
            # Avoid AssertionError: Non-consecutive added token 'TOKEN'
            # found. Should have index 20031 but has index 20032 in saved
            # vocabulary.
            print("fixing added token ids:")
            added_tokens = tokenizer.added_tokens_encoder
            print("before ", added_tokens)
            sorted_tokens = sorted(
                [(token, old_id) for token, old_id in added_tokens.items()], key=lambda x: x[1]
            )
            tokenizer.added_tokens_encoder = {
                token: new_id for new_id, (token, _) in zip(
                    range(sorted_tokens[0][1], sorted_tokens[0][1]+len(sorted_tokens)),
                    sorted_tokens,
                )
            }
            print("after ", tokenizer.added_tokens_encoder)

        print("saving tokenizer with new tags")
        tokenizer.save_pretrained(args.model_dir)
        print("saving model with new tags")
        model.save_pretrained(args.model_dir)
        print("saving checkpoints")
        for path, checkpoint in checkpoints:
            torch.save(checkpoint, path)

    print("special tokens map ", tokenizer.special_tokens_map)
    print("id-to-lang-code ",tokenizer.id_to_lang_code)
    print("lang-code-to-id", tokenizer.lang_code_to_id)

    tokenizer = MBartTokenizer.from_pretrained(args.model_dir)
    # check embeddings
    if args.add_language_tags is not None and args.initialize_tags is not None:
        for new_tag, init_tag in zip(args.add_language_tags, args.initialize_tags):
            print("original language embedding for {}: {}".format(init_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(init_tag)]))
            print("initialized {} with embedding: {}".format(new_tag, model.model.shared.weight[tokenizer.convert_tokens_to_ids(new_tag)]))

    if args.verbose > 0:
        tokenizer = MBartTokenizer.from_pretrained(args.model_dir)
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
    
