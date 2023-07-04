from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from VisCPM.generation.generation_utils import BeamHypotheses, apply_repetition_penalty
from VisCPM.cpm_tokenizers.bee import CPMBeeTokenizer
from VisCPM.models import VLU_CPMBee
from VisCPM.models.cpmbee import CPMBeeTorch
from VisCPM.utils.utils import convert_data_to_id, pad


class VLLMCPMBeeGeneration:
    def __init__(self, model: VLU_CPMBee, tokenizer: CPMBeeTokenizer, transform):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform

    def _convert_to_tensors(self, data: Any, in_context_samples: List[Any] = [], max_inp_length: Optional[int] = None):
        answer_placeholders = []

        def _put_placeholder(data: Any, path: List[str] = []):
            if isinstance(data, dict):
                ret = {}
                for k, v in data.items():
                    ret[k] = _put_placeholder(v, path + [k])
                return ret
            else:
                answer_placeholders.append(path)
                return "<ans_{}>".format(len(answer_placeholders))

        data["<ans>"] = _put_placeholder(data["<ans>"])
        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,
            image_bound,
        ) = convert_data_to_id(self.tokenizer, data, shuffle_answer=False, max_depth=8)

        if max_inp_length is not None:
            input_ids = input_ids[: max_inp_length]
            context = context[: max_inp_length]
            segment_ids = segment_ids[: max_inp_length]

        sub_ans_map: Dict[int, int] = {}
        for fake_id, token_sub in table_states["token_id_table"]["<ans>"].items():
            token = table_states["ext_table"][fake_id]
            if token.startswith("<ans_") and token.endswith(">"):
                ans_id = int(token[5:-1])
                sub_ans_map[token_sub] = ans_id

        tmp_input_ids = []
        tmp_input_sub = []
        tmp_input_seg = []

        predict_segments: List[Tuple[int, int]] = []
        for i in range(input_ids.shape[0]):
            if context[i] == 0:
                if input_ids[i] == self.tokenizer.encoder["<ans>"]:
                    # is ans
                    # (segment_id, ans_id)
                    predict_segments.append((segment_ids[i], sub_ans_map[input_id_subs[i]]))
            else:
                tmp_input_ids.append(input_ids[i])
                tmp_input_sub.append(input_id_subs[i])
                tmp_input_seg.append(segment_ids[i])

        if len(predict_segments) == 0:
            raise ValueError("No answer to predict")

        input_ids = np.array(tmp_input_ids, dtype=np.int32)
        input_id_subs = np.array(tmp_input_sub, dtype=np.int32)
        context = np.full_like(tmp_input_ids, 1, dtype=np.int8)
        segment_ids = np.array(tmp_input_seg, dtype=np.int32)
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)
        image_bound = np.array(image_bound, dtype=np.int32)

        for i, sample in enumerate(in_context_samples):
            (
                sample_input_ids,
                sample_id_subs,
                _,
                sample_segments,
                sample_rel,
                n_segments,
                table_states,
                image_bound,
            ) = convert_data_to_id(self.tokenizer, sample, table_states, max_depth=8)
            input_ids = np.concatenate([input_ids, sample_input_ids], axis=0)
            input_id_subs = np.concatenate([input_id_subs, sample_id_subs], axis=0)
            context = np.concatenate(
                [context, np.ones(sample_input_ids.shape, dtype=np.int8)], axis=0
            )
            segment_ids = np.concatenate([segment_ids, sample_segments], axis=0)
            segment_rel_offset = np.concatenate(
                [
                    segment_rel_offset,
                    np.full(sample_input_ids.shape, segment_rel.shape[0], dtype=np.int32),
                ],
                axis=0,
            )
            segment_rel = np.concatenate([segment_rel, sample_rel], axis=0)
            sample_ids = np.concatenate(
                [sample_ids, np.full(sample_input_ids.shape, i + 1, dtype=np.int32)], axis=0
            )
            num_segments = np.concatenate(
                [num_segments, np.full(sample_input_ids.shape, n_segments, dtype=np.int32)], axis=0
            )
        input_pos = np.arange(input_ids.shape[0], dtype=np.int32)

        return (
            input_ids,
            input_id_subs,
            input_pos,
            context,
            segment_ids,
            segment_rel_offset,
            segment_rel,
            sample_ids,
            num_segments,
            predict_segments,
            answer_placeholders,
            table_states["ext_table"],
            table_states["token_id_table"],
            image_bound
        )

    def _process_list(self, data_list: List[Any], max_inp_length: Optional[int] = None):
        pack_tensor = []
        other_info = []
        segment_rel_pack = []

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []

        for data in data_list:
            (
                input_ids,
                input_id_subs,
                input_pos,
                context,
                segment_ids,
                segment_rel_offset,
                segment_rel,
                sample_ids,
                num_segments,
                predict_segments,
                answer_placeholders,
                ext_table,
                token_id_table,
                image_bound
            ) = self._convert_to_tensors(data, [], max_inp_length)
            rev_ext_table: Dict[int, str] = {}
            for token, mp in token_id_table.items():
                if token == "<ans>":
                    continue
                token_id = self.tokenizer.encoder[token]
                for fake_id, token_sub in mp.items():
                    if token_sub > 0:
                        if (token_id, token_sub) not in batch_ext_table_map:
                            batch_ext_table_map[(token_id, token_sub)] = (
                                len(batch_ext_table_ids) + self.tokenizer.vocab_size
                            )
                            batch_ext_table_ids.append(token_id)
                            batch_ext_table_sub.append(token_sub)
                        rev_ext_table[batch_ext_table_map[(token_id, token_sub)]] = ext_table[
                            fake_id
                        ]
                    else:
                        rev_ext_table[token_id] = ext_table[fake_id]
            pack_tensor.append(
                {
                    "input_ids": torch.from_numpy(input_ids).unsqueeze(0),
                    "input_id_subs": torch.from_numpy(input_id_subs).unsqueeze(0),
                    "input_pos": torch.from_numpy(input_pos).unsqueeze(0),
                    "context": torch.from_numpy(context).unsqueeze(0),
                    "sample_idx": torch.from_numpy(sample_ids).unsqueeze(0),
                    "num_segments": torch.from_numpy(num_segments).unsqueeze(0),
                    "segment_ids": torch.from_numpy(segment_ids).unsqueeze(0),
                    "segment_rel_offset": torch.from_numpy(segment_rel_offset).unsqueeze(0),
                }
            )
            segment_rel_pack.append(torch.from_numpy(segment_rel))
            other_info.append(
                {
                    "predict_segments": predict_segments,
                    "answer_placeholders": answer_placeholders,
                    "ext_table": rev_ext_table,
                }
            )

        keys = set(pack_tensor[0].keys())
        padded = {}
        for key in keys:
            padded[key] = pad(pack_tensor, key).cuda()

        max_num_rels = 0
        for rel in segment_rel_pack:
            max_num_rels = max(max_num_rels, rel.size(0))
        padded_rels = torch.zeros(len(segment_rel_pack), max_num_rels, dtype=torch.int32)
        for i, rel in enumerate(segment_rel_pack):
            padded_rels[i, : rel.size(0)] = rel
        padded["segment_rel"] = padded_rels.cuda()
        padded["batch_ext_table_ids"] = torch.tensor(
            batch_ext_table_ids, dtype=torch.int32, device="cuda"
        )
        padded["batch_ext_table_sub"] = torch.tensor(
            batch_ext_table_sub, dtype=torch.int32, device="cuda"
        )
        padded['image_bound'] = torch.from_numpy(image_bound).unsqueeze(0).cuda()
        return padded, other_info

    def generate(self, img_list, max_inp_length: Optional[int] = None, extra_inp_dict: dict = None, vision_hidden_states=None, return_vision_hidden_states=False, **kwargs):
        data_list = []

        pixel_values = []
        for img in img_list:
            if vision_hidden_states is None:
                pixel_values.append(self.transform(img))
            inp_dict = {'image': self.tokenizer.unk_token * self.model.query_num}
            if extra_inp_dict:
                inp_dict.update(extra_inp_dict)
            inp_dict['<ans>'] = ''
            data_list.append(inp_dict)

        model_inputs, other_info = self._process_list(data_list, max_inp_length)

        with torch.inference_mode():
            if vision_hidden_states is None:
                pixel_values = torch.stack(pixel_values).cuda()
                model_inputs['pixel_values'] = pixel_values
            else:
                model_inputs['vision_hidden_states'] = vision_hidden_states

            model_inputs['hidden_states'], vision_hidden_states = self.model.get_vllm_embedding(model_inputs)
            result_ids = self._decode(model_inputs, other_info, **kwargs)
        for sent_id, result in enumerate(result_ids):
            ans_result_map: Dict[int, List[int]] = {}
            for raw_word_id, ans_id in result:
                if ans_id not in ans_result_map:
                    ans_result_map[ans_id] = []
                ans_result_map[ans_id].append(raw_word_id)

            answer_placeholders = other_info[sent_id]["answer_placeholders"]
            ext_table = other_info[sent_id]["ext_table"]
            data = data_list[sent_id]
            for ans_id, token_ids in ans_result_map.items():
                if token_ids[-1] == self.tokenizer.eos_id:
                    token_ids = token_ids[:-1]
                text = self.tokenizer.decode(token_ids, ext_table)
                path = answer_placeholders[ans_id - 1]

                if len(path) > 0:
                    p = data["<ans>"]
                    for part in path[:-1]:
                        p = p[part]
                    p[path[-1]] = text
                else:
                    data["<ans>"] = text
            for ans_id in range(len(answer_placeholders)):
                if (ans_id + 1) not in ans_result_map:
                    path = answer_placeholders[ans_id]
                    p = data["<ans>"]
                    for part in path[:-1]:
                        p = p[part]
                    p[path[-1]] = None

        if return_vision_hidden_states:
            return data_list, vision_hidden_states

        return data_list

    def _decode(self, model_inputs, other_info, **kwargs):
        raise NotImplementedError("_decode is not implemented.")


class VLLMCPMBeeBeamSearch(VLLMCPMBeeGeneration):
    """ use case
        beam_search = VLLMCPMBeeGeneration(vlu_cpmbee, tokenizer, transform)
        img = Image.open('xxx.jpg').convert('RGB')
        print(beam_search.generate([img], max_inp_length=128)[0]['<ans>'])
    """

    def _decode(
        self,
        model_inputs,
        other_info,
        beam_size=3,
        max_length=100,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        repetition_window=None,
    ):
        """
        Beam search
        Args:
            model_inputs (dict): input ids.
            beam_size (int, optional, defaults to 3): beam size of beam search.
            generate_length (int, optional, defaults to 100): maximum generation length.
            repetition_penalty (float, optional, defaults to 1.0): repetition penalty coefficient, 1.0 means no penalty.
            repetition_window (int, optional, defaults to None): window size of repetition penalty, None means that all output tokens are penalized.
        """  # noqa: E501
        # generate_length + 1 for EOS token
        max_length += 1

        # expand dimmension
        batch_size = model_inputs["input_ids"].size(0)
        input: torch.Tensor = (
            model_inputs["input_ids"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        input_sub: torch.Tensor = (
            model_inputs["input_id_subs"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        input_pos: torch.Tensor = (
            model_inputs["input_pos"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        context: torch.Tensor = (
            model_inputs["context"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        sample_ids: torch.Tensor = (
            model_inputs["sample_idx"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        num_segments: torch.Tensor = (
            model_inputs["num_segments"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment: torch.Tensor = (
            model_inputs["segment_ids"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment_rel_offset: torch.Tensor = (
            model_inputs["segment_rel_offset"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment_rel: torch.Tensor = (
            model_inputs["segment_rel"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        hidden_states: torch.Tensor = (
            model_inputs["hidden_states"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, *model_inputs["hidden_states"].shape[1:])
            .contiguous()
            .view(batch_size * beam_size, *model_inputs["hidden_states"].shape[1:])
        )
        ext_table_ids: torch.Tensor = model_inputs["batch_ext_table_ids"]
        ext_table_sub: torch.Tensor = model_inputs["batch_ext_table_sub"]
        ext_table_ids_cpu = ext_table_ids.cpu()
        ext_table_sub_cpu = ext_table_sub.cpu()

        done = [False for _ in range(batch_size)]

        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, max_length, length_penalty=length_penalty, early_stopping=False)
            for _ in range(batch_size)
        ]

        pred_start_index = input.size(-1)
        _, _, past_key_values = self.model.llm.inference(
            input=input,
            input_sub=input_sub,
            position=input_pos,
            context=context,
            sample_ids=sample_ids,
            num_segments=num_segments,
            segment=segment,
            segment_rel_offset=segment_rel_offset,
            segment_rel=segment_rel,
            ext_table_ids=ext_table_ids,
            ext_table_sub=ext_table_sub,
            past_key_values=None,
            hidden_states=hidden_states
        )

        beam_states = []
        for sent_id in range(batch_size):
            instance_beam_states = []

            for beam_id in range(beam_size):
                instance_beam_states.append(
                    {
                        "idx": 0,
                        "ans": [],
                        "nx_token_id": self.tokenizer.bos_id,
                        "nx_token_sub": 0,
                        "nx_segment_id": other_info[sent_id]["predict_segments"][0][0],
                        "nx_position": 0,
                    }
                )
            beam_states.append(instance_beam_states)
        for i in range(max_length + 1):
            tmp_input = []
            tmp_input_sub = []
            tmp_position = []
            tmp_segment = []
            for sent_id in range(batch_size):
                for beam_id in range(beam_size):
                    tmp_input.append(beam_states[sent_id][beam_id]["nx_token_id"])
                    tmp_input_sub.append(beam_states[sent_id][beam_id]["nx_token_sub"])
                    tmp_position.append(beam_states[sent_id][beam_id]["nx_position"])
                    tmp_segment.append(beam_states[sent_id][beam_id]["nx_segment_id"])
            with torch.no_grad():
                input = torch.cat(
                    [
                        input,
                        torch.tensor(tmp_input, dtype=torch.int32, device="cuda").view(
                            batch_size * beam_size, 1
                        ),
                    ],
                    dim=-1,
                )
                logits, _, past_key_values = self.model.llm.inference(
                    input=input[:, -1:],
                    input_sub=torch.tensor(tmp_input_sub, dtype=torch.int32, device="cuda").view(
                        batch_size * beam_size, 1
                    ),
                    position=torch.tensor(tmp_position, dtype=torch.int32, device="cuda").view(
                        batch_size * beam_size, 1
                    ),
                    context=torch.ones(
                        batch_size * beam_size, dtype=torch.bool, device="cuda"
                    ).view(batch_size * beam_size, 1),
                    sample_ids=torch.zeros(
                        batch_size * beam_size, dtype=torch.int32, device="cuda"
                    ).view(batch_size * beam_size, 1),
                    num_segments=num_segments[:, -1:],
                    segment=torch.tensor(tmp_segment, dtype=torch.int32, device="cuda").view(
                        batch_size * beam_size, 1
                    ),
                    segment_rel_offset=segment_rel_offset[:, -1:],
                    segment_rel=segment_rel,
                    ext_table_ids=ext_table_ids,
                    ext_table_sub=ext_table_sub,
                    past_key_values=past_key_values,
                )
                logits = logits[:, -1, :]

            # skip all steps when we are done with each sentence
            if all(done):
                break

            for sent_id in range(batch_size):
                if self.tokenizer.unk_id not in other_info[sent_id]["ext_table"]:
                    # unk is not allowed, mask unk
                    logits[
                        sent_id * beam_size: (sent_id + 1) * beam_size, self.tokenizer.unk_id
                    ] = -10000
                ext_ids = set()
                for v in other_info[sent_id]["ext_table"].keys():
                    ext_ids.add(v)
                for ext_id in range(
                    self.tokenizer.vocab_size, self.tokenizer.vocab_size + ext_table_ids.size(0)
                ):
                    if ext_id not in ext_ids:
                        logits[sent_id * beam_size: (sent_id + 1) * beam_size, ext_id] = -10000

            apply_repetition_penalty(
                logits,
                batch_size,
                beam_size,
                input,
                repetition_penalty,
                pred_start_index,
                input.size(-1) - 1,
                repetition_window,
            )
            logits = logits / temperature
            scores = F.log_softmax(logits, dim=-1)
            next_scores = scores + beam_scores[:, None].expand_as(
                scores
            )  # (batch_size * beam_size, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            next_scores, next_words = torch.topk(
                next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)
            next_beam_states = []

            for sent_id in range(batch_size):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item(), i
                )
                if done[sent_id]:
                    next_beam_states.append(
                        [
                            (
                                {
                                    "idx": 0,
                                    "ans": [],
                                    "nx_token_id": 0,
                                    "nx_token_sub": 0,
                                    "nx_segment_id": 0,
                                    "nx_position": 0,
                                },
                                0,
                                0,
                            )
                        ]
                        * beam_size
                    )  # pad the batch
                    continue

                # next sentence beam content
                next_instance_beam_states = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = torch.div(idx, scores.size(-1), rounding_mode="floor").item()
                    word_id = (idx % scores.size(-1)).item()

                    curr_info = beam_states[sent_id][beam_id]
                    # end of sentence, or next word
                    if (
                        word_id == self.tokenizer.eos_id
                        and (curr_info["idx"] + 1 == len(other_info[sent_id]["predict_segments"]))
                    ) or i == max_length:
                        generated_hyps[sent_id].add(
                            beam_states[sent_id][beam_id]["ans"]
                            + [
                                (
                                    word_id,
                                    other_info[sent_id]["predict_segments"][curr_info["idx"]][1],
                                )
                            ],
                            value.item(),
                        )
                    elif word_id == self.tokenizer.eos_id:
                        next_instance_beam_states.append(
                            (
                                {
                                    "idx": curr_info["idx"] + 1,
                                    "ans": curr_info["ans"]
                                    + [
                                        (
                                            word_id,
                                            other_info[sent_id]["predict_segments"][
                                                curr_info["idx"]
                                            ][1],
                                        )
                                    ],
                                    "nx_token_id": self.tokenizer.bos_id,
                                    "nx_token_sub": 0,
                                    "nx_segment_id": other_info[sent_id]["predict_segments"][
                                        curr_info["idx"] + 1
                                    ][0],
                                    "nx_position": 0,
                                },
                                value.item(),
                                sent_id * beam_size + beam_id,
                            )
                        )

                    else:
                        raw_word_id = word_id
                        word_id_sub = 0
                        if word_id >= self.tokenizer.vocab_size:
                            word_id -= self.tokenizer.vocab_size
                            word_id_sub = int(ext_table_sub_cpu[word_id].item())
                            word_id = int(ext_table_ids_cpu[word_id].item())

                        next_instance_beam_states.append(
                            (
                                {
                                    "idx": curr_info["idx"],
                                    "ans": curr_info["ans"]
                                    + [
                                        (
                                            raw_word_id,
                                            other_info[sent_id]["predict_segments"][
                                                curr_info["idx"]
                                            ][1],
                                        )
                                    ],
                                    "nx_token_id": word_id,
                                    "nx_token_sub": word_id_sub,
                                    "nx_segment_id": curr_info["nx_segment_id"],
                                    "nx_position": curr_info["nx_position"] + 1,
                                },
                                value.item(),
                                sent_id * beam_size + beam_id,
                            )
                        )

                    # the beam for next step is full
                    if len(next_instance_beam_states) == beam_size:
                        break

                # update next beam content
                assert len(next_instance_beam_states) == 0 if i == max_length else beam_size
                next_beam_states.append(next_instance_beam_states)

            # we have reached the last step
            if i == max_length:
                break

            # sanity check / prepare next batch
            beam_reorder_idx = []
            beam_new_scores = []
            beam_states = []
            for sent_id in range(batch_size):
                instance_beam_states = []
                for beam_id in range(beam_size):
                    state, value, beam_idx = next_beam_states[sent_id][beam_id]
                    beam_reorder_idx.append(beam_idx)
                    beam_new_scores.append(value)
                    instance_beam_states.append(state)
                beam_states.append(instance_beam_states)

            input = input[beam_reorder_idx, :]
            beam_scores = torch.tensor(beam_new_scores, dtype=torch.float, device=input.device)
            for kw in past_key_values.keys():
                if kw == "buffer":
                    buf_list = past_key_values[kw]
                    nw_buf_list = []
                    for k_buf, v_buf in buf_list:
                        nw_buf_list.append((k_buf[beam_reorder_idx, :], v_buf[beam_reorder_idx, :]))
                    past_key_values[kw] = nw_buf_list
                else:
                    past_key_values[kw] = past_key_values[kw][beam_reorder_idx, :]

        # select the best hypotheses
        results = []
        for sent_id, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            results.append(best_hyp)
        return results
