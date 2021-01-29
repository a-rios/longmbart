from typing import List, Optional, Tuple, Dict
from torch import nn, Tensor, narrow
from longformer.longformer import LongformerSelfAttention
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration


class LongformerEncoderDecoderForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.model.encoder.layers):
                layer.self_attn = LongformerSelfAttentionForBart(config, layer_id=i)


class LongformerEncoderDecoderConfig(BartConfig):
    def __init__(self, attention_window: List[int] = None, attention_dilation: List[int] = None,
                 autoregressive: bool = False, attention_mode: str = 'sliding_chunks',
                 gradient_checkpointing: bool = False, **kwargs):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Longformer
                selfattention, 'sliding_chunks' for another implementation of Longformer selfattention
        """
        super().__init__(**kwargs)
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.gradient_checkpointing = gradient_checkpointing
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2']

class LongformerSelfAttentionForBart(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.longformer_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        self.output = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor, # shape (batch_size, q_len, model_size)
        key_value_states: Optional[Tensor] = None, # cross-attention in transformers.models.mbart.modeling_mbart
        past_key_value: Optional[Tuple[Tensor]] = None, # only for decoder
        attention_mask: Optional[Tensor] = None, # shape (batch_size, 1, q_len, k_len) -> used to be key_padding_mask, attn_mask is now decoder_attention_mask (= autoregressive mask). 
        layer_head_mask: Optional[Tensor] = None, # head dropout?
        output_attentions: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:

        bsz, tgt_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        assert list(hidden_states.size()) == [bsz, tgt_len, embed_dim]
        ## new attention mask is (batch_size, 1, q_len, k_len), last dim (k) is the same for all q's, need to remove q (torch.narrow) -> (batch_size, 1, k_len) for LongformerSelfAttention
        attention_mask = narrow(input=attention_mask, dim=2, start=0, length=1) # shape (batch_size, 1, 1, key_len
        outputs = self.longformer_self_attn(
            hidden_states,
            attention_mask=attention_mask * -1, # shape (batch_size, 1, 1, key_len)
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
        )

        ## new: MBart encoder expects shape (seq_len, bsz, embed_dim), no transpose needed
        attn_output = self.output(outputs[0])
        # new return in MBartAttention has attn_output, attn_weights_reshaped, past_key_value (only for decoder), need to return 3 values (None for past_key_value)
        return (attn_output, outputs[1:] ,None) if len(outputs) == 2 else (attn_output, None, None)
