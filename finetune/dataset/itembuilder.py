import io
import json
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image, PngImagePlugin

from VisCPM.cpm_tokenizers import CPMBeeTokenizer

from finetune.dataset.bee import convert_data_to_id
from finetune.utils.prompts import caption_zh, caption_en
from finetune.utils.utils import is_contain_chinese
from finetune.utils.logger import init_logger

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# logger = init_logger(level='INFO')
logger = init_logger(level='WARNING')


def maybe_select_text(raw_text):
    candidates = raw_text.split('<cap_sep>')
    return random.choice(candidates)


def maybe_parse_json(raw_text: str):
    # VG raw
    if raw_text.startswith('[{') and raw_text.endswith('}]'):
        try:
            data = json.loads(raw_text)
            text_list = [x['phrase'] for x in data if x['height'] > 160 and x['width'] > 160]
            if len(text_list) == 0:
                return max(data, key=lambda x: len(x['phrase'].split()))['phrase']
            else:
                return random.choice(text_list)
        except:
            return raw_text
    else:
        return raw_text


def clean_text(raw_text):
    text = raw_text.replace('<PERSON>', '')
    text = maybe_parse_json(maybe_select_text(text))
    return text


def check_text_valid(raw_text):
    if pd.isna(raw_text):
        return False
    if not is_contain_chinese(raw_text) and len(raw_text.split()) <= 3:
        return False
    if '<img' in raw_text or '<a href' in raw_text:
        return False
    return True


class ItemBuilder():
    def __init__(self, transform=None):
        self.transform = transform

    def build_item(self, data):
        if self.transform is not None:
            return self.transform(data)
        return data


class CPMBeeImageTextBuilder(ItemBuilder):
    """
    use case
    >>> tokenizer = CPMBeeTokenizer()
    >>> max_length = 64
    >>> builder = CPMBeeImageTextBuilder(tokenizer=tokenizer, max_len=max_length, transform=transform, query_len=32)
    >>> dataset_path = '/mnt/data/user/tc_agi/multi_modal/test_data/test_files.txt'

    >>> dataset= ParquetDataset(
    >>>     dataset_path,
    >>>     builder,
    >>>     data_queue_size=500,
    >>>     num_workers=2
    >>> )
    >>> dataloader = DataLoader(
    >>>     dataset,
    >>>     sampler=None,
    >>>     batch_size=8,
    >>>     pin_memory=True,
    >>>     shuffle=False,
    >>>     collate_fn=CPMBeeCollater(tokenizer=tokenizer, max_len=max_length)
    >>> )
    """

    def __init__(self, tokenizer: CPMBeeTokenizer, max_len, transform=None, query_len=32, extra_inp_dict=None, task='caption', min_resolution=224, skip_overlength=False):
        super().__init__(transform)
        if extra_inp_dict:
            assert isinstance(extra_inp_dict, dict)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.query_len = query_len
        self.extra_inp_dict = extra_inp_dict
        self.task = task
        self.min_resolution = min_resolution
        self.skip_overlength = skip_overlength

    def build_item(self, data):
        # show data
        logger.info(f'build_item data.key: {data.keys()}')
        task = self.task
        img_buffer = data['BUFFER']#
        # 中英双语随机选一个
        if 'TEXT' in data and 'ZH_TEXT' in data:
            text = random.choice([data['TEXT'], data['ZH_TEXT']])
        else:
            text = data['TEXT']
        # text = data['ZH_TEXT']

        text = clean_text(text)
        if text.startswith('[{') and text.endswith('}]') and 'human' in text and 'gpt' in text:
            task = 'llava_instruction'
            logger.info(f'LLAVA instruction: {text}')

        img_io = io.BytesIO(img_buffer)
        img_io.seek(0)
        try:
            assert text.strip() != ''
            image = Image.open(img_io).convert('RGB')
            # todo@wangshan: filter
            if min(image.size) < self.min_resolution:
                return None
            image = self.transform(image)
        except:
            return None

        inp_dicts = []

        if task == 'caption':
            # image caption 场景
            # 后续图文交错的话应该是 <image_0>、<image_1> 
            caption_prompt = random.choice(caption_zh) if is_contain_chinese(text) else random.choice(caption_en)

            inp_dict = {
                'image': self.tokenizer.unk_token * self.query_len,  # only placeholder
                'input': caption_prompt
            }
            if self.extra_inp_dict:
                inp_dict.update(self.extra_inp_dict)
            inp_dict['<ans>'] = self.tokenizer.escape(text)
            inp_dicts.append(inp_dict)
        elif task == 'llava_instruction':
            conversion = json.loads(text)
            if len(conversion) % 2 != 0 or len([c for c in conversion if c['from'] == 'human']) != len([c for c in conversion if c['from'] == 'gpt']):
                return None
            rounds = len(conversion) // 2
            context = ''  # 上下文
            for i in range(rounds):
                if i > 0:
                    for j in ((i - 1) * 2, i * 2):
                        role = 'User: ' if conversion[j]['from'] == 'human' else 'AI: '
                        context += role + conversion[j]['value'].replace('<image>', '').strip() + '\n'
                question = conversion[i * 2]['value'].replace('<image>', '').strip()
                ans = conversion[i * 2 + 1]['value']

                inp_dict = {
                    'image': self.tokenizer.unk_token * self.query_len,
                    'context': context,
                    'question': question,
                }
                if self.extra_inp_dict:
                    inp_dict.update(self.extra_inp_dict)
                inp_dict['<ans>'] = self.tokenizer.escape(ans)
                inp_dicts.append(inp_dict)

        res = []
        for inp_dict in inp_dicts:
            (
                input_ids,
                input_id_subs,
                context,
                segment_ids,
                segment_rel,
                n_segments,
                table_states,
                image_bound
            ) = convert_data_to_id(self.tokenizer, data=inp_dict, shuffle_answer=False, max_depth=8)

            if len(input_ids) > self.max_len:
                if self.skip_overlength:
                    if random.random() > 0.95:
                        logger.warn(f"overlength={len(input_ids)}, raw_inp={inp_dict}, skip data")
                    else:
                        logger.warn(f"overlength={len(input_ids)}, skip data")
                    continue
            input_ids = input_ids[: self.max_len]
            input_id_subs = input_id_subs[: self.max_len]
            context = context[: self.max_len]
            segment_ids = segment_ids[: self.max_len]

            sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
            segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
            num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)
            raw_data = {
                'text': text
            }
            # raw_data show
            logger.info(f'raw_data: {raw_data}')

            res.append({
                'pixel_values': image,
                'input_ids': torch.from_numpy(input_ids).unsqueeze(0),
                'input_id_subs': torch.from_numpy(input_id_subs).unsqueeze(0),
                'context': torch.from_numpy(context).unsqueeze(0),
                'segment_ids': torch.from_numpy(segment_ids).unsqueeze(0),
                'segment_rel_offset': torch.from_numpy(segment_rel_offset).unsqueeze(0),
                'segment_rel': torch.from_numpy(segment_rel).unsqueeze(0),
                'sample_ids': torch.from_numpy(sample_ids).unsqueeze(0),
                'num_segments': torch.from_numpy(num_segments).unsqueeze(0),
                'image_bound': torch.from_numpy(image_bound),
                'raw_data': raw_data,
            })
        
        return res