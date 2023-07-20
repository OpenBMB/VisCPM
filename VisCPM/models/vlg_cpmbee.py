import os
import torch


class VLG_CPMBee(torch.nn.Module):
    def __init__(self, llm, sd) -> None:
        super().__init__()
        self.sd = sd
        self.llm = llm

    def forward(self, data):
        device = data['input_ids'].device
        bs = data['input_ids'].size(0)

        llm_hidden_state = self.llm.input_embedding(data['input_ids'], data['input_id_subs'])

        _, hidden_states = self.llm(
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
            hidden_states=llm_hidden_state
        )
        loss, model_pred = self.sd(data['pixel_values'], hidden_states)
        return loss, model_pred

    @torch.no_grad()
    def generate(
        self,
        data,
        uncond_data,
        **generate_kwargs,
    ):
        device = data['input_ids'].device
        bs = data['input_ids'].size(0)
        with torch.no_grad():
            llm_hidden_state = self.llm.input_embedding(data['input_ids'], data['input_id_subs'])
            _, hidden_states = self.llm(
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
                hidden_states=llm_hidden_state
            )

        with torch.no_grad():
            uncond_llm_hidden_state = self.llm.input_embedding(uncond_data['input_ids'], uncond_data['input_id_subs'])

            _, uncond_hidden_states = self.llm(
                input=uncond_data['input_ids'],
                input_sub=uncond_data['input_id_subs'],
                length=uncond_data['length'],
                context=uncond_data['context'],
                sample_ids=uncond_data['sample_ids'],
                num_segments=uncond_data['num_segments'],
                segment=uncond_data['segment_ids'],
                segment_rel_offset=uncond_data['segment_rel_offset'],
                segment_rel=uncond_data['segment_rel'],
                span=uncond_data['span'],
                ext_table_ids=uncond_data['ext_table_ids'],
                ext_table_sub=uncond_data['ext_table_sub'],
                hidden_states=uncond_llm_hidden_state
            )
        image = self.sd.generate(
            hidden_states,
            uncond_hidden_states,
            **generate_kwargs
        )
        return image
