"""Spatially gate RDA reference tokens with CARF region predictions."""

import math

import torch
import torch.nn.functional as F


class RDACARFSpatialAttnProcessor2_0:
    """PyTorch 2 attention with a bias only on appended reference tokens."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        carf_attention_gate=None,
        rda_token_count=0,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)
        query = attn.to_q(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query_length = query.shape[-2]
        key_length = key.shape[-2]
        prepared_mask = None
        if attention_mask is not None:
            prepared_mask = attn.prepare_attention_mask(
                attention_mask, key_length, batch_size
            ).view(batch_size, attn.heads, -1, key_length)

        token_count = min(max(int(rda_token_count), 0), key_length)
        if (
            is_cross_attention
            and carf_attention_gate is not None
            and token_count > 0
        ):
            gate = carf_attention_gate.float()
            if gate.shape[0] != batch_size:
                if batch_size % gate.shape[0] != 0:
                    raise ValueError(
                        "CARF gate batch does not match attention batch"
                    )
                gate = gate.repeat_interleave(batch_size // gate.shape[0], dim=0)
            side = int(round(math.sqrt(query_length)))
            if side * side == query_length:
                gate = F.interpolate(
                    gate, (side, side), mode="bilinear", align_corners=False
                ).flatten(2).unsqueeze(-1)
            else:
                gate = F.interpolate(
                    gate.flatten(2), size=query_length, mode="linear",
                    align_corners=False,
                ).unsqueeze(-1)
            reference_bias = gate.clamp_min(1e-4).log().to(query.dtype)
            spatial_bias = torch.cat(
                [
                    query.new_zeros(
                        batch_size, 1, query_length,
                        key_length - token_count,
                    ),
                    reference_bias.expand(-1, -1, -1, token_count),
                ],
                dim=-1,
            )
            prepared_mask = (
                spatial_bias
                if prepared_mask is None
                else prepared_mask + spatial_bias
            )

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=prepared_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        ).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_rda_carf_attention(unet):
    """Install compatible processors; gating activates on cross-attention only.

    Diffusers forwards cross_attention_kwargs through both transformer attention
    calls. Installing the compatible processor on self-attention as well avoids
    repeated "unexpected kwargs" warnings; its is_cross_attention guard leaves
    self-attention numerically unchanged.
    """
    processors = {}
    installed = 0
    for name, processor in unet.attn_processors.items():
        processors[name] = RDACARFSpatialAttnProcessor2_0()
        if name.endswith("attn2.processor"):
            installed += 1
    if installed == 0:
        raise RuntimeError("No UNet cross-attention processors were found")
    unet.set_attn_processor(processors)
    return installed
