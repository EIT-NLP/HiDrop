""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from llava.constants import IGNORE_INDEX
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaSdpaAttention,
    LlamaRMSNorm,
    LlamaConfig,
    LLAMA_START_DOCSTRING,
    LlamaPreTrainedModel,
    LLAMA_INPUTS_DOCSTRING,
    LlamaDecoderLayer, _get_unpad_data, apply_rotary_pos_emb, LlamaMLP
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        total_seq_token_len: Optional[int] = None,  ###
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        ### 实现
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if total_seq_token_len is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=total_seq_token_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        total_seq_token_len: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            total_seq_token_len=total_seq_token_len,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaExitModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # 新增参数
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """" """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        original_position_ids = position_ids
        original_attention_mask = attention_mask
        original_hidden_states = hidden_states
        original_labels = labels

        # print("\n---------------------------------------------")
        # if position_ids is None:
        #     print(f"original position_ids: {position_ids}")
        # if attention_mask is None:
        #     print(f"original attention_mask: {attention_mask}") # 可能是None
        # if hidden_states is None:
        #     print(f"original hidden_states: {hidden_states}")
        # if labels is None:
        #     print(f"original label: {labels}")
        # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
        #     print(f"original: {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
        # else:
        #     print(f"original: there is NoneType")
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:   # Training
                if layer_idx == 0:    # Before Late Entry
                    (
                        position_ids,
                        attention_mask,
                        hidden_states,
                        labels,  # update labels and return
                        total_seq_token_len
                    )  = self.pure_text_before_late_entry(
                        features=hidden_states,         # (bs, ???, dim=4096)  ???可能是seq_len
                        position_ids=position_ids,      # (1, ???)
                        attention_mask=attention_mask,  # (bs, ???)
                        labels=labels,                  # (bs, ???)
                    )
                    noim_position_ids = position_ids
                    noim_attention_mask = attention_mask
                    noim_labels = labels
                    # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                    #     print(f"Before Late Entry ---> Layer {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                    # print(f"Before Late Entry total_seq_token_len = {total_seq_token_len}")
                elif layer_idx == self.late_entry_layer-1:    # Late Entry
                    (
                        position_ids,
                        attention_mask,
                        hidden_states,
                        labels,  # update labels and return
                    )  = self.late_entry(
                        features=hidden_states,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        original_features=original_hidden_states,
                        original_attention_mask=original_attention_mask,
                        original_labels=original_labels,
                    )
                    # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                    #     print(f"Late Entry ---> Layer {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                elif layer_idx == self.early_exit_layer-1:  # Early Exit
                    (
                        position_ids,
                        attention_mask,
                        hidden_states,
                        labels,  # update labels and return
                    )  = self.early_exit(
                        features=hidden_states,
                        attention_mask=attention_mask,
                        noim_position_ids=noim_position_ids,
                        noim_attention_mask=noim_attention_mask,
                        noim_labels=noim_labels,
                    )
                    # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                    #     print(f"Early Exit ---> Layer {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                    # print(f"Early Exit total_seq_token_len = {total_seq_token_len}")
                # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                #     print(f"Current {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    total_seq_token_len
                )
            else:   # Inference
                if hidden_states.shape[1] == 1: # Decode Stage
                    # text len + decode len
                    cur_seq_token_len = past_key_values.get_usable_length(seq_length, layer_idx=layer_idx) + seq_length
                    if layer_idx == 0 or layer_idx == self.early_exit_layer-1:
                        total_seq_token_len = cur_seq_token_len + 576

                    if original_attention_mask is not None: ### 待移除
                        print("!!!!!!!!!!!!!!!!!!!!!!!! Decode Stage !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                else:   # Prefill Stage
                    if layer_idx == 0 and hidden_states.shape[1] > 1:    # Before Late Entry
                        (
                            position_ids,
                            attention_mask,
                            hidden_states,
                            labels,  # update labels and return
                            total_seq_token_len
                        )  = self.pure_text_before_late_entry(
                            features=hidden_states,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        noim_position_ids = position_ids
                        noim_attention_mask = attention_mask
                        noim_labels = labels
                        # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                        #     print(f"Before Late Entry ---> Layer {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                        # else:
                        #     print(f"Before Late Entry ---> Layer {layer_idx} : {hidden_states.shape}")
                        # print(f"Before Late Entry total_seq_token_len = {total_seq_token_len}")
                    elif layer_idx == self.late_entry_layer-1 and hidden_states.shape[1] > 1:    # Late Entry
                        (
                            position_ids,
                            attention_mask,
                            hidden_states,
                            labels,  # update labels and return
                        )  = self.late_entry(
                            features=hidden_states,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            original_features=original_hidden_states,
                            original_attention_mask=original_attention_mask,
                            original_labels=original_labels,
                        )
                        # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                        #     print(f"Late Entry ---> Layer {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                        # else:
                        #     print(f"Late Entry ---> Layer {layer_idx} : {hidden_states.shape}")
                    elif layer_idx == self.early_exit_layer-1 and hidden_states.shape[1] > 1:  # Early Exit
                        (
                            position_ids,
                            attention_mask,
                            hidden_states,
                            labels,  # update labels and return
                        )  = self.early_exit(
                            features=hidden_states,
                            attention_mask=attention_mask,
                            noim_position_ids=noim_position_ids,
                            noim_attention_mask=noim_attention_mask,
                            noim_labels=noim_labels,
                        )
                        # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                        #     print(f"Early Exit ---> Layer {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                        # else:
                        #     print(f"Early Exit ---> Layer {layer_idx} : {hidden_states.shape}")
                        # print(f"Early Exit total_seq_token_len = {total_seq_token_len}")
                # if position_ids is not None and attention_mask is not None and hidden_states is not None and labels is not None:
                #     print(f"Current {layer_idx} : {position_ids.shape}, {attention_mask.shape}, {hidden_states.shape}, {labels.shape}")
                # else:
                #     print(f"Current {layer_idx} : {hidden_states.shape}")

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    total_seq_token_len=total_seq_token_len,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            outputs = (hidden_states, next_cache, all_hidden_states, all_self_attns)
            return outputs + (labels,)  # 不直接返回，添加一个元素来传递修改后的标签
        else:   # 使用标准的 BaseModelOutputWithPast，不包含 labels
            base_output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
            base_output.updated_labels = labels # 添加一个属性来传递修改后的标签
            return base_output

    def pure_text_before_late_entry(self, features, position_ids, attention_mask, labels):
        """ 当前只在Flash-Attn中写过 """
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if getattr(self.config, 'tokenizer_padding_side', 'right') == "right":  # 处理右侧填充得情况，这是常见的填充方式
            bsz = features.shape[0]
            total_seq_token_len = features.shape[1]

            features_list = []
            attention_mask_list = []
            labels_list = []

            if attention_mask is None:
                attention_mask = torch.ones((bsz, features.shape[1]), dtype=torch.bool, device=features.device)
            else:
                attention_mask = attention_mask.bool()
            if labels is None:
                labels = torch.full((bsz, features.shape[1]), IGNORE_INDEX, device=features.device)

            # 提取有效数据，padding的部分去掉
            features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features, attention_mask)]
            labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
            attention_mask = [cur_attention_mask[cur_attention_mask] for cur_attention_mask, cur_attention_mask in zip(attention_mask, attention_mask)]

            for i in range(bsz):
                image_index = self.image_token_posi[i]  # 获取图像token的起始位置
                if image_index == -1:  # 如果样本中没有图像token
                    # 保持原始特征不变
                    cur_input_embeds = features[i]
                    features_list.append(cur_input_embeds)
                    attention_mask_list.append(attention_mask[i])
                    labels_list.append(labels[i])
                    continue

                """
                    features: [(seq_len, dim)]
                    labels: [(seq_len)]
                    attention_mask: [(seq_len)]
                """
                image_end_index = image_index + 576
                new_input_embeds = torch.cat([features[i][:image_index, :], features[i][image_end_index:, :]], dim=0)
                new_labels = torch.cat([labels[i][:image_index], labels[i][image_end_index:]], dim=0)
                new_attention_mask = torch.cat([attention_mask[i][:image_index], attention_mask[i][image_end_index:]], dim=0)

                features_list.append(new_input_embeds)
                labels_list.append(new_labels)
                attention_mask_list.append(new_attention_mask)

            # 处理序列长度和填充
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]
                new_attention_mask = [x[:tokenizer_model_max_length] for x in attention_mask_list]
                new_labels = [x[:tokenizer_model_max_length] for x in labels_list]

            # 计算最大长度并填充所有序列到相同长度
            """
                如果序列超过最大长度，则截断
                找到批次中最长序列的长度
                填充所有序列到相同长度
                更新位置编码以匹配新的序列结构
            """
            max_len = max(x.shape[0] for x in new_input_embeds)

            # padding the sequences to form batch
            embeds_padded = []
            labels_paded = []
            attention_mask_padded = []
            position_ids = torch.zeros((bsz, max_len), dtype=position_ids.dtype, device=position_ids.device)
            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len_emb = cur_new_embed.shape[0]
                dif = max_len - cur_len_emb  # padding to longest seq

                cur_new_embed = torch.cat([cur_new_embed, torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)], dim=0)
                cur_new_labels = torch.cat([cur_new_labels, torch.full((dif,), IGNORE_INDEX, dtype=cur_new_labels.dtype, device=cur_new_labels.device)], dim=0)
                cur_attention_mask = new_attention_mask[i]
                cur_attention_mask = torch.cat([cur_attention_mask, torch.full((dif,), False, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)], dim=0)

                embeds_padded.append(cur_new_embed)
                labels_paded.append(cur_new_labels)
                attention_mask_padded.append(cur_attention_mask)

                cur_len = new_attention_mask[i].sum().item()
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                if self.image_token_posi[i] != -1:
                    position_ids[i, self.image_token_posi[i]:cur_len] += 576    # 注意，调整位置编码，相当于把<IMG>的位置空出来了
                #     print(f"\nbsz{i}: w/  image")
                # else:
                #     print(f"bsz{i} w/o image")

            new_input_embeds = torch.stack(embeds_padded, dim=0)
            new_input_embeds = new_input_embeds.to(features[0].dtype)

            new_attention_mask = torch.stack(attention_mask_padded, dim=0)
            new_labels = torch.stack(labels_paded, dim=0)

            if _position_ids is None:
                position_ids = None
            if _labels is None:
                new_labels = None
            if _attention_mask is None:
                new_attention_mask = None
            else:
                new_attention_mask = new_attention_mask.to(dtype=_attention_mask.dtype)
            return position_ids, new_attention_mask, new_input_embeds, new_labels, total_seq_token_len
        else:
            raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")

    def late_entry(self, features, position_ids, attention_mask, original_features, original_attention_mask, original_labels):
        _position_ids = position_ids
        _attention_mask = attention_mask
        _original_attention_mask= original_attention_mask

        if getattr(self.config, 'tokenizer_padding_side', 'right') == "right":  # 处理右侧填充得情况，这是常见的填充方式
            bsz = features.shape[0]

            features_list = []

            if attention_mask is None:
                attention_mask = torch.ones((bsz, features.shape[1]), dtype=torch.bool, device=features.device)
            else:
                attention_mask = attention_mask.bool()

            if original_attention_mask is None:
                original_attention_mask = torch.ones((bsz, original_features.shape[1]), dtype=torch.bool, device=original_features.device)
            else:
                original_attention_mask = original_attention_mask.bool()

            # 提取有效数据，padding的部分去掉
            features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features, attention_mask)]
            temp_original_features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(original_features, original_attention_mask)]

            for i in range(bsz):
                image_index = self.image_token_posi[i]  # 获取图像token的起始位置
                if image_index == -1:  # 如果样本中没有图像token
                    # 保持原始特征不变
                    cur_input_embeds = features[i]
                    features_list.append(cur_input_embeds)
                    continue

                image_end_index = image_index + 576
                new_input_embeds = torch.cat([features[i][:image_index, :],  temp_original_features[i][image_index:image_end_index, :], features[i][image_index:, :]], dim=0)
                features_list.append(new_input_embeds)

            # 处理序列长度和填充
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]

            # 计算最大长度并填充所有序列到相同长度
            max_len = max(x.shape[0] for x in new_input_embeds)

            embeds_padded = []
            position_ids = torch.zeros((bsz, max_len), dtype=position_ids.dtype, device=position_ids.device)
            for i, cur_new_embed in enumerate(new_input_embeds):
                cur_len_emb = cur_new_embed.shape[0]
                dif = max_len - cur_len_emb  # padding to longest seq

                cur_new_embed = torch.cat([cur_new_embed, torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)], dim=0)
                embeds_padded.append(cur_new_embed)

                # print(f"\ncur_len: {cur_len == cur_len_emb}")
                position_ids[i, :cur_len_emb] = torch.arange(0, cur_len_emb, dtype=position_ids.dtype, device=position_ids.device)

            new_input_embeds = torch.stack(embeds_padded, dim=0)
            new_input_embeds = new_input_embeds.to(features[0].dtype)

            if _position_ids is None:
                position_ids = None
            if _attention_mask is None:
                # print(f"attention_mask: {_attention_mask}")
                # print(f"original_attention_mask: {_original_attention_mask}")
                original_attention_mask = None
            # print(f"\nposition_ids: {torch.equal(original_position_ids, position_ids)}")  # False
            # print(f"\nattn-mask: {torch.equal(original_attention_mask, new_attention_mask)}") # True
            # print(f"\nlabels: {torch.equal(original_labels, new_labels)}")    # True
            return position_ids, original_attention_mask, new_input_embeds, original_labels
        else:
            raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")

    def early_exit(self, features, attention_mask, noim_position_ids, noim_attention_mask, noim_labels):
        _attention_mask = attention_mask

        if getattr(self.config, 'tokenizer_padding_side', 'right') == "right":  # step2: 处理右侧填充得情况，这是常见的填充方式
            bsz = features.shape[0]

            features_list = []

            if attention_mask is None:
                attention_mask = torch.ones((bsz, features.shape[1]), dtype=torch.bool, device=features.device)
            else:
                attention_mask = attention_mask.bool()

            # 提取有效数据，padding的部分去掉
            features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features, attention_mask)]

            for i in range(bsz):
                image_index = self.image_token_posi[i]  # 获取图像token的起始位置
                if image_index == -1:  # 如果样本中没有图像token
                    # 保持原始特征不变
                    cur_input_embeds = features[i]
                    features_list.append(cur_input_embeds)
                    continue

                image_end_index = image_index + 576
                new_input_embeds = torch.cat([features[i][:image_index, :], features[i][image_end_index:, :]], dim=0)
                features_list.append(new_input_embeds)

            # 处理序列长度和填充
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]

            # 计算最大长度并填充所有序列到相同长度
            max_len = max(x.shape[0] for x in new_input_embeds)

            embeds_padded = []
            for i, cur_new_embed in enumerate(new_input_embeds):
                cur_len_emb = cur_new_embed.shape[0]
                dif = max_len - cur_len_emb  # padding to longest seq

                cur_new_embed = torch.cat([cur_new_embed, torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)], dim=0)
                embeds_padded.append(cur_new_embed)

                # if self.image_token_posi[i] != -1:
                #     print(f"\nbsz{i}: w/  image")
                # else:
                #     print(f"bsz{i} w/o image")

            new_input_embeds = torch.stack(embeds_padded, dim=0)
            new_input_embeds = new_input_embeds.to(features[0].dtype)

            # print(f"\nposition_ids: {torch.equal(noim_position_ids, position_ids)}")  # Ture
            # print(f"\nattn-mask: {torch.equal(noim_attention_mask, new_attention_mask)}") # True
            # print(f"\nlabels: {torch.equal(noim_labels, new_labels)}")    # True
            return noim_position_ids, noim_attention_mask, new_input_embeds, noim_labels
        else:
            raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")




class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaExitModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,  ####
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # >>> 获取更新后的标签
        if return_dict and hasattr(outputs, 'updated_labels'):
            labels = outputs.updated_labels
        elif not return_dict and len(outputs) > 4:  # 如果是元组，最后一个元素是修改后的标签
            labels = outputs[4]
        else:
            labels = labels

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
                if past_key_values is not None: ### 新增
                    """ 当存在`past_key_values`（已缓存的键值对）时，这个条件将输入标记进一步裁剪为仅保留最后一个标记
                    在自回归生成过程中，使用过去的缓存时，模型只需要预测下一个单一标记
                    """
                    input_ids = input_ids[:, -1:]   # 只保留最后一个标记
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past