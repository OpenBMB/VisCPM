"""
图像生成推理代码
"""
import os
import PIL.Image as Image
import collections
import numpy as np
import torch
from VisCPM.cpm_tokenizers.bee import CPMBeeTokenizer
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
from transformers import pipeline, AutoProcessor, AutoModel, BertForSequenceClassification, BertTokenizer
from typing import Optional, List

from VisCPM.models import SDWrapper, VLG_CPMBee, CPMBeeConfig, CPMBeeTorch
from VisCPM.utils.utils import CPMBeeCollater, convert_data_to_id
import bminf


def grid_image(images: List[Image.Image]) -> Image.Image:
    n = len(images)
    nrow = min(n, 8)
    images_tensor = [to_tensor(image) for image in images]
    images_tensor_grid = [make_grid(x, nrow, padding=0) for x in images_tensor]
    images_grid = [to_pil_image(x) for x in images_tensor_grid]
    return images_grid


class VisCPMPaint:
    def __init__(self, model_path, image_safety_checker=True, prompt_safety_checker=True, add_ranker=True):
        llm_config = CPMBeeConfig.from_json_file('./config/cpm-bee-10b.json')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.llm = CPMBeeTorch(llm_config)
        self.tokenizer = CPMBeeTokenizer()
        self.sd = SDWrapper(image_safety_checker=image_safety_checker)
        self.model = VLG_CPMBee(self.llm, self.sd)
        self.load_model(model_path)

        if os.getenv('CUDA_MEMORY_CPMBEE_MAX', False):
            limit = os.getenv("CUDA_MEMORY_CPMBEE_MAX")
            try:
                assert limit.lower().endswith('g')
                memory_limit = int(limit.lower()[:-1]) * (1 << 30)
                print(f'use CUDA_MEMORY_CPMBEE_MAX={limit} to limit cpmbee cuda memory cost ')
            except:
                memory_limit = None
                print(f'environment CUDA_MEMORY_CPMBEE_MAX={limit} parse error')

            self.llm = bminf.wrapper(self.llm, memory_limit=memory_limit)
            self.sd.to(self.device)
        else:
            self.llm.to(self.device)
            self.sd.to(self.device)

        if prompt_safety_checker:
            model = BertForSequenceClassification.from_pretrained('openbmb/VisCPM-Paint', subfolder='text-security-checker')
            tokenizer = BertTokenizer.from_pretrained('openbmb/VisCPM-Paint', subfolder='text-security-checker')
            self.prompt_safety_checker = pipeline(
                'text-classification',
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )
        else:
            self.prompt_safety_checker = None

        if add_ranker:
            self.clip_ranker = AutoModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(self.device)
            self.clip_preprocessor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        else:
            self.clip_ranker = None

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        trans_block_ckpt = collections.OrderedDict()
        unet_ckpt = collections.OrderedDict()
        llm_ckpt = collections.OrderedDict()

        for key, value in ckpt.items():
            if key.startswith('trans_block'):
                trans_block_ckpt[key.replace('trans_block.', '')] = value
            elif key.startswith('unet'):
                unet_ckpt[key.replace('unet.', '')] = value
            elif key.startswith('llm.'):
                llm_ckpt[key.replace('llm.', '')] = value

        self.sd.trans_block.load_state_dict(trans_block_ckpt)
        self.sd.unet.load_state_dict(unet_ckpt)
        self.llm.load_state_dict(llm_ckpt)

    def build_input(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image_size: int = 512
    ):
        data_input = {'caption': prompt, 'objects': ''}
        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,
            image_bound
        ) = convert_data_to_id(self.tokenizer, data=data_input, shuffle_answer=False, max_depth=8)
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)
        data = {
            'pixel_values': torch.zeros(3, image_size, image_size).unsqueeze(0),
            'input_ids': torch.from_numpy(input_ids).unsqueeze(0),
            'input_id_subs': torch.from_numpy(input_id_subs).unsqueeze(0),
            'context': torch.from_numpy(context).unsqueeze(0),
            'segment_ids': torch.from_numpy(segment_ids).unsqueeze(0),
            'segment_rel_offset': torch.from_numpy(segment_rel_offset).unsqueeze(0),
            'segment_rel': torch.from_numpy(segment_rel).unsqueeze(0),
            'sample_ids': torch.from_numpy(sample_ids).unsqueeze(0),
            'num_segments': torch.from_numpy(num_segments).unsqueeze(0),
            'image_bound': image_bound,
            'raw_data': prompt,
        }

        uncond_data_input = {
            'caption': "" if negative_prompt is None else negative_prompt,
            'objects': ''
        }
        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,
            image_bound
        ) = convert_data_to_id(self.tokenizer, data=uncond_data_input, shuffle_answer=False, max_depth=8)
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)
        uncond_data = {
            'pixel_values': torch.zeros(3, image_size, image_size).unsqueeze(0),
            'input_ids': torch.from_numpy(input_ids).unsqueeze(0),
            'input_id_subs': torch.from_numpy(input_id_subs).unsqueeze(0),
            'context': torch.from_numpy(context).unsqueeze(0),
            'segment_ids': torch.from_numpy(segment_ids).unsqueeze(0),
            'segment_rel_offset': torch.from_numpy(segment_rel_offset).unsqueeze(0),
            'segment_rel': torch.from_numpy(segment_rel).unsqueeze(0),
            'sample_ids': torch.from_numpy(sample_ids).unsqueeze(0),
            'num_segments': torch.from_numpy(num_segments).unsqueeze(0),
            'image_bound': image_bound,
            'raw_data': "" if negative_prompt is None else negative_prompt,
        }
        packer = CPMBeeCollater(
            tokenizer=self.tokenizer,
            max_len=max(data['input_ids'].size(-1), uncond_data['input_ids'].size(-1))
        )
        data = packer([data])
        uncond_data = packer([uncond_data])
        return data, uncond_data

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 7.5,
        image_size: int = 512,
        generator: Optional[torch.Generator] = None,
    ):
        if self.prompt_safety_checker:
            res = self.prompt_safety_checker(prompt, top_k=None)[0]
            if res['label'] == 'LABEL_1' and res['score'] > 0.75:
                print('Your input has unsafe content, please correct it!')
                return
        data, uncond_data = self.build_input(prompt, negative_prompt, image_size=512)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        for key, value in uncond_data.items():
            if isinstance(value, torch.Tensor):
                uncond_data[key] = value.to(self.device)
        if self.clip_ranker:
            num_images_per_prompt = num_images_per_prompt * 4
            clip_text = [prompt] * 4
        output = self.model.generate(
            data, uncond_data,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            width=image_size,
            height=image_size,
            generator=generator
        )
        images, nsfw_content_detected = output.images, output.nsfw_content_detected
        if self.clip_ranker:
            n = num_images_per_prompt // 4
            new_images = []
            for i in range(n):
                clip_input = self.clip_preprocessor(text=clip_text, images=images[i * 4: (i + 1) * 4],
                                                    return_tensors='pt', padding=True).to(self.device)
                clip_output = self.clip_ranker(**clip_input)
                clip_score = torch.diag(clip_output.logits_per_image)
                _, indices = torch.sort(-clip_score)
                img = images[i * 4: (i + 1) * 4][indices[0]]
                new_images.append(img)
            return grid_image(new_images)
        else:
            return grid_image(images)
