import cv2
import numpy as np
import torch
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms
import urllib
from tqdm import tqdm
from cpm_live.tokenizers import CPMBeeTokenizer
from torch.utils.data import default_collate
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict
from numpy.typing import NDArray
import importlib.machinery
import importlib.util
import types
import random


CPMBeeInputType = Union[str, Dict[str, "CPMBeeInputType"]]


def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor


class CPMBeeCollater:
    """
    针对 cpmbee 输入数据 collate, 对应 cpm-live 的 _MixedDatasetBatchPacker
    目前利用 torch 的原生 Dataloader 不太适合改造 in-context-learning
    并且原来实现为了最大化提高有效 token 比比例, 会有一个 best_fit 操作, 这个目前也不支持
    todo: 重写一下 Dataloader or BatchPacker
    """

    def __init__(self, tokenizer: CPMBeeTokenizer, max_len):
        self.tokenizer = tokenizer
        self._max_length = max_len
        self.pad_keys = ['input_ids', 'input_id_subs', 'context', 'segment_ids', 'segment_rel_offset',
                         'segment_rel', 'sample_ids', 'num_segments']

    def __call__(self, batch):
        batch_size = len(batch)

        tgt = np.full((batch_size, self._max_length), -100, dtype=np.int32)
        # 目前没有 best_fit, span 为全 0
        span = np.zeros((batch_size, self._max_length), dtype=np.int32)
        length = np.zeros((batch_size,), dtype=np.int32)

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []
        raw_data_list: List[Any] = []

        for i in range(batch_size):
            instance_length = batch[i]['input_ids'][0].shape[0]
            length[i] = instance_length
            raw_data_list.extend(batch[i]['raw_data'])

            for j in range(instance_length):
                idx, idx_sub = batch[i]['input_ids'][0, j], batch[i]['input_id_subs'][0, j]
                tgt_idx = idx
                if idx_sub > 0:
                    # need to be in ext table
                    if (idx, idx_sub) not in batch_ext_table_map:
                        batch_ext_table_map[(idx, idx_sub)] = len(batch_ext_table_map)
                        batch_ext_table_ids.append(idx)
                        batch_ext_table_sub.append(idx_sub)
                    tgt_idx = batch_ext_table_map[(idx, idx_sub)] + self.tokenizer.vocab_size
                if j > 1 and batch[i]['context'][0, j - 1] == 0:
                    if idx != self.tokenizer.bos_id:
                        tgt[i, j - 1] = tgt_idx
                    else:
                        tgt[i, j - 1] = self.tokenizer.eos_id
            if batch[i]['context'][0, instance_length - 1] == 0:
                tgt[i, instance_length - 1] = self.tokenizer.eos_id

        if len(batch_ext_table_map) == 0:
            # placeholder
            batch_ext_table_ids.append(0)
            batch_ext_table_sub.append(1)

        # image
        if 'pixel_values' in batch[0]:
            data = {'pixel_values': default_collate([i['pixel_values'] for i in batch])}
        else:
            data = {}

        # image_bound
        if 'image_bound' in batch[0]:
            data['image_bound'] = default_collate([i['image_bound'] for i in batch])

        # bee inp
        for key in self.pad_keys:
            data[key] = pad(batch, key, max_length=self._max_length, padding_value=0, padding_side='right')

        data['context'] = data['context'] > 0
        data['length'] = torch.from_numpy(length)
        data['span'] = torch.from_numpy(span)
        data['target'] = torch.from_numpy(tgt)
        data['ext_table_ids'] = torch.from_numpy(np.array(batch_ext_table_ids))
        data['ext_table_sub'] = torch.from_numpy(np.array(batch_ext_table_sub))
        data['raw_data'] = raw_data_list

        return data


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


# aug functions
def identity_func(img):
    return img


def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256

    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            table = np.arange(n_bins) * scale - low * scale
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize_func(img):
    '''
        same output as PIL.ImageOps.equalize
        PIL's implementation is different from cv2.equalize
    '''
    n_bins = 256

    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0:
            return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def rotate_func(img, degree, fill=(0, 0, 0)):
    '''
    like PIL, rotate by degree, not radians
    '''
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out


def solarize_func(img, thresh=128):
    '''
        same output as PIL.ImageOps.posterize
    '''
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Color
    '''
    # implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = (
        np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * factor
        + np.float32([[0.114], [0.587], [0.299]])
    )
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    """
        same output as PIL.ImageEnhance.Contrast
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = np.array([(
        el - mean) * factor + mean
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def brightness_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def sharpness_func(img, factor):
    '''
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


def shear_x_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def posterize_func(img, bits):
    '''
        same output as PIL.ImageOps.posterize
    '''
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    pad_size = pad_size // 2
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


# level to args
def enhance_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        return ((level / MAX_LEVEL) * 1.8 + 0.1,)
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 0.3
        if np.random.random() > 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * float(translate_const)
        if np.random.random() > 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * cutout_const)
        return (level, replace_value)

    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 256)
        return (level, )
    return level_to_args


def none_level_to_args(level):
    return ()


def posterize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 4)
        return (level, )
    return level_to_args


def rotate_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 30
        if np.random.random() < 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args


func_dict = {
    'Identity': identity_func,
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Rotate': rotate_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    'ShearX': shear_x_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Posterize': posterize_func,
    'ShearY': shear_y_func,
}

translate_const = 10
MAX_LEVEL = 10
replace_value = (128, 128, 128)
arg_dict = {
    'Identity': none_level_to_args,
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
    'TranslateX': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'TranslateY': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
}


class RandomAugment(object):

    def __init__(self, N=2, M=10, isPIL=False, augs=[]):
        self.N = N
        self.M = M
        self.isPIL = isPIL
        if augs:
            self.augs = augs
        else:
            self.augs = list(arg_dict.keys())

    def get_random_ops(self):
        sampled_ops = np.random.choice(self.augs, self.N)
        return [(op, 0.5, self.M) for op in sampled_ops]

    def __call__(self, img):
        if self.isPIL:
            img = np.array(img)
        ops = self.get_random_ops()
        for name, prob, level in ops:
            if np.random.random() > prob:
                continue
            args = arg_dict[name](level)
            img = func_dict[name](img, *args)
        return img


def build_transform(is_train, randaug=True, input_size=224, interpolation='bicubic'):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(
                input_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        if randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "vissl"})
        ) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)
