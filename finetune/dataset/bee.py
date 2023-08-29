# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import TypedDict
import numpy as np
from numpy.typing import NDArray
import importlib.machinery
import importlib.util
import types
import random

from VisCPM.cpm_tokenizers import CPMBeeTokenizer



CPMBeeInputType = Union[str, Dict[str, "CPMBeeInputType"]]


class _DictTree(TypedDict):
    value: str
    children: List["_DictTree"]
    depth: int
    segment_id: int
    need_predict: bool
    is_image: bool


class _PrevExtTableStates(TypedDict):
    ext_table: Dict[int, str]
    token_id_table: Dict[str, Dict[int, int]]


class _TransformFuncDict(TypedDict):
    loader: importlib.machinery.SourceFileLoader
    module: types.ModuleType
    last_m: float


_TransformFunction = Callable[[CPMBeeInputType, int, random.Random], CPMBeeInputType]


class CPMBeeBatch(TypedDict):
    inputs: NDArray[np.int32]
    inputs_sub: NDArray[np.int32]
    length: NDArray[np.int32]
    context: NDArray[np.bool_]
    sample_ids: NDArray[np.int32]
    num_segments: NDArray[np.int32]
    segment_ids: NDArray[np.int32]
    segment_rel_offset: NDArray[np.int32]
    segment_rel: NDArray[np.int32]
    spans: NDArray[np.int32]
    target: NDArray[np.int32]
    ext_ids: NDArray[np.int32]
    ext_sub: NDArray[np.int32]
    task_ids: NDArray[np.int32]
    task_names: List[str]
    raw_data: List[Any]


def rel_to_bucket(n_up: int, n_down: int, max_depth: int = 8):
    ret = n_up * max_depth + n_down
    if ret == 0:
        return ret
    else:
        # bucket 1 is reserved for incontext samples
        return ret + 1


def convert_data_to_id(
    tokenizer: CPMBeeTokenizer,
    data: Any,
    prev_ext_states: Optional[_PrevExtTableStates] = None,
    shuffle_answer: bool = True,
    max_depth: int = 8
):
    root: _DictTree = {
        "value": "<root>",
        "children": [],
        "depth": 0,
        "segment_id": 0,
        "need_predict": False,
        "is_image": False
    }

    segments = [root]

    def _build_dict_tree(data: CPMBeeInputType, depth: int, need_predict: bool, is_image: bool) -> List[_DictTree]:
        if isinstance(data, dict):
            ret_list: List[_DictTree] = []
            curr_items = list(data.items())
            if need_predict and shuffle_answer:
                access_idx = np.arange(len(curr_items))
                np.random.shuffle(access_idx)
                curr_items = [curr_items[idx] for idx in access_idx]
            for k, v in curr_items:
                child_info: _DictTree = {
                    "value": k,
                    "children": [],
                    "depth": depth,
                    "segment_id": len(segments),
                    "need_predict": False,  # only leaves are contexts
                    "is_image": False,
                }
                segments.append(child_info)
                child_info["children"] = _build_dict_tree(
                    v, depth + 1,
                    need_predict=need_predict or (depth == 1 and k == "<ans>"),
                    is_image=is_image or (depth == 1 and k == "image")
                )  # elements in <root>.<ans>

                ret_list.append(child_info)
            return ret_list
        else:
            assert isinstance(data, str), "Invalid data {}".format(data)
            ret: _DictTree = {
                "value": data,
                "children": [],
                "depth": depth,
                "segment_id": len(segments),
                "need_predict": need_predict,
                "is_image": is_image,
            }
            segments.append(ret)
            return [ret]

    root["children"] = _build_dict_tree(data, 1, False, False)

    num_segments = len(segments)
    segment_rel = np.zeros((num_segments * num_segments,), dtype=np.int32)

    def _build_segment_rel(node: _DictTree) -> List[Tuple[int, int]]:
        ret: List[Tuple[int, int]] = [(node["segment_id"], node["depth"])]
        for child in node["children"]:
            sub = _build_segment_rel(child)
            for seg_id_1, depth_1 in sub:
                for seg_id_2, depth_2 in ret:
                    n_up = min(depth_1 - node["depth"], max_depth - 1)
                    n_down = min(depth_2 - node["depth"], max_depth - 1)
                    segment_rel[seg_id_1 * num_segments + seg_id_2] = rel_to_bucket(
                        n_up, n_down, max_depth=max_depth
                    )
                    segment_rel[seg_id_2 * num_segments + seg_id_1] = rel_to_bucket(
                        n_down, n_up, max_depth=max_depth
                    )
            ret.extend(sub)
        return ret

    _build_segment_rel(root)

    input_ids: List[int] = []
    input_id_subs: List[int] = []
    segment_bound: List[Tuple[int, int]] = []
    image_bound: List[Tuple[int, int]] = []

    ext_table: Dict[int, str] = {}
    token_id_table: Dict[str, Dict[int, int]] = {}

    if prev_ext_states is not None:
        ext_table = prev_ext_states["ext_table"]
        token_id_table = prev_ext_states["token_id_table"]

    for seg in segments:
        tokens, ext_table = tokenizer.encode(seg["value"], ext_table)

        token_id_subs = []
        reid_token_ids = []
        for idx in tokens:
            if idx in ext_table:
                # unk or special token
                token = ext_table[idx]
                if token.startswith("<") and token.endswith(">"):
                    # special token
                    if "_" in token:
                        token_name = token[1:-1].split("_", maxsplit=1)[0]
                    else:
                        token_name = token[1:-1]
                    token_name = "<{}>".format(token_name)
                else:
                    token_name = "<unk>"

                if token_name not in token_id_table:
                    token_id_table[token_name] = {}
                if idx not in token_id_table[token_name]:
                    token_id_table[token_name][idx] = len(token_id_table[token_name])
                if token_name not in tokenizer.encoder:
                    raise ValueError("Invalid token {}".format(token))
                reid_token_ids.append(tokenizer.encoder[token_name])
                token_id_subs.append(token_id_table[token_name][idx])
            else:
                reid_token_ids.append(idx)
                token_id_subs.append(0)
        tokens = [tokenizer.bos_id] + reid_token_ids
        token_id_subs = [0] + token_id_subs
        if not seg["need_predict"]:
            tokens = tokens + [tokenizer.eos_id]
            token_id_subs = token_id_subs + [0]
        else:
            # no eos
            pass
        begin = len(input_ids)
        input_ids.extend(tokens)
        input_id_subs.extend(token_id_subs)
        end = len(input_ids)
        segment_bound.append((begin, end))

    ids = np.array(input_ids, dtype=np.int32)
    id_subs = np.array(input_id_subs, dtype=np.int32)
    segs = np.zeros((ids.shape[0],), dtype=np.int32)
    context = np.zeros((ids.shape[0],), dtype=np.int8)
    for i, (begin, end) in enumerate(segment_bound):
        if not segments[i]["need_predict"]:
            context[begin:end] = 1
        if segments[i]["is_image"]:
            image_bound.append((begin+1, end-1))
        segs[begin:end] = i

    curr_ext_table_states: _PrevExtTableStates = {
        "ext_table": ext_table,
        "token_id_table": token_id_table,
    }
    image_bound = np.array(image_bound, dtype=np.int32)
    return ids, id_subs, context, segs, segment_rel, num_segments, curr_ext_table_states, image_bound
