from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from timm.models.layers import trunc_normal_
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import ModelOutput

from VisCPM.models.cpmbee import CPMBeeTorch
import os


def construct_query_parameter(query_k, h_size, init_weights):
    query_data = torch.zeros(query_k, h_size)
    trunc_normal_(query_data, std=.02)
    for idx in range(query_k):
        if init_weights[idx] is not None:
            query_data[idx] = init_weights[idx]
    query = torch.nn.Parameter(query_data)
    return query


@dataclass
class CausalVLLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class VLU_CPMBee(torch.nn.Module):
    def __init__(self, llm: CPMBeeTorch, vpm, vision_dim, query_num, device=None) -> None:
        super().__init__()
        self.device = device
        self.vpm = vpm
        self.llm = llm

        self.vision_dim = vision_dim
        self.query_num = query_num
        self.query = None

        if query_num is not None:
            bos_weight = self.vpm.beit3.text_embed.weight.data[0]
            eos_weight = self.vpm.beit3.text_embed.weight.data[2]
            query_init_weight = [bos_weight] + [None] * (self.query_num - 2) + [eos_weight]
            self.query = construct_query_parameter(
                self.query_num, self.vision_dim, query_init_weight)

        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(self.vpm.hidden_size, self.llm.config.dim_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.llm.config.dim_model, self.llm.config.dim_model)
        )

    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            pixel_values = data['pixel_values']
            vision_hidden_states = self.vpm(pixel_values=pixel_values, query_embed=self.query)

            vision_hidden_states = self.mapping(vision_hidden_states)  # (query_num, llm_dim)
        else:
            vision_hidden_states = data['vision_hidden_states']


        vllm_embedding = self.llm.input_embedding(data['input_ids'], data['input_id_subs'])
        vision_hidden_states = vision_hidden_states.type(vllm_embedding.dtype)

        image_bound = data['image_bound']
        image_bound = image_bound.squeeze(1)
        image_indices = torch.stack(
            [torch.arange(r[0], r[1], dtype=torch.long) for r in image_bound]
        ).to(self.device)

        vllm_embedding.scatter_(1, image_indices.unsqueeze(-1).repeat(1, 1, vllm_embedding.shape[-1]),
                                vision_hidden_states)

        return vllm_embedding, vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        logits, hidden_states = self.llm(
            input=data['input_ids'],
            input_sub=data['input_id_subs'],
            length=data['length'],
            context=data['context'],
            sample_ids=data['sample_ids'],
            num_segments=data['num_segments'],
            segment=data['segment_ids'],
            segment_rel_offset=data['segment_rel_offset'],
            segment_rel=data['segment_rel'],
            span=data['span'],
            ext_table_ids=data['ext_table_ids'],
            ext_table_sub=data['ext_table_sub'],
            hidden_states=vllm_embedding
        )

        return CausalVLLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            vision_hidden_states=vision_hidden_states
        )
