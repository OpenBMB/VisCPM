import os
import numpy as np
import torch
from PIL import Image
from VisCPM.cpm_tokenizers import CPMBeeTokenizer
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from timm.models import create_model
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid
from transformers import CLIPImageProcessor
from typing import List

from VisCPM.generation.vllm_bee import VLLMCPMBeeBeamSearch
from VisCPM.models import VLU_CPMBee
from VisCPM.models.cpmbee import CPMBeeConfig, CPMBeeTorch
from VisCPM.utils import utils
import bminf

file_path = os.path.dirname(__file__)


def grid_image(images: List[Image.Image]) -> Image.Image:
    n = len(images)
    nrow = min(n, 8)
    images_tensor = [to_tensor(image) for image in images]
    images_tensor_grid = make_grid(images_tensor, nrow, padding=0)
    images_grid = to_pil_image(images_tensor_grid)
    return images_grid


class VisCPMChat(object):
    def __init__(self, model_path, config_path=None, image_safety_checker=False) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = utils.build_transform(is_train=False)
        self.tokenizer = CPMBeeTokenizer()

        self.beit3_wrapper = create_model("beit3_large_patch16_224")
        if config_path is None:
            config_path = os.path.join(file_path, '../config/cpm-bee-10b.json')
        self.config = CPMBeeConfig.from_json_file(config_path)
        self.cpm_model = CPMBeeTorch(self.config)

        self.vlu_cpmbee = VLU_CPMBee(
            llm=self.cpm_model,
            vpm=self.beit3_wrapper,
            vision_dim=self.beit3_wrapper.args.encoder_embed_dim,
            query_num=64,
            device=self.device
        )

        self.beam_search = VLLMCPMBeeBeamSearch(
            self.vlu_cpmbee, self.tokenizer, self.transform, device=self.device
        )

        vlu_state_dict = torch.load(model_path, map_location="cpu")
        self.vlu_cpmbee.load_state_dict(vlu_state_dict)
        self.vlu_cpmbee.half()

        if os.getenv('CUDA_MEMORY_CPMBEE_MAX', False):
            limit = os.getenv("CUDA_MEMORY_CPMBEE_MAX")
            try:
                assert limit.lower().endswith('g')
                memory_limit = int(limit.lower()[:-1]) * (1 << 30)
                print(f'use CUDA_MEMORY_CPMBEE_MAX={limit} to limit cpmbee cuda memory cost ')
            except:
                memory_limit = None
                print(f'environment CUDA_MEMORY_CPMBEE_MAX={limit} parse error')

            self.cpm_model = bminf.wrapper(self.cpm_model, memory_limit=memory_limit)
            self.vlu_cpmbee.query.data = self.vlu_cpmbee.query.data.to(self.device)
            self.vlu_cpmbee.mapping.to(self.device)
            self.vlu_cpmbee.vpm.to(self.device)
        else:
            self.vlu_cpmbee.to(self.device)

        self.vlu_cpmbee.eval()

        if image_safety_checker:
            self.image_safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            )
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )  # Download image_processing_config from huggingface.co and cache.
            self.image_safety_checker.to(self.device)
        else:
            self.image_safety_checker = None
            self.feature_extractor = None

    def chat(self, image, question, context='', vision_hidden_states=None):
        extra_inp_dict = {
            "context": context,
            "question": question,
        }

        images, has_nsfw_concept = self.run_image_safety_checker(
            [np.asarray(image)], self.device, torch.float
        )
        if has_nsfw_concept and has_nsfw_concept[0]:
            print("Content is not safe for work.")
            images = grid_image(np.asarray(image))

        res, vision_hidden_states = self.beam_search.generate(
            [image],
            max_inp_length=3000,
            max_length=512,
            extra_inp_dict=extra_inp_dict,
            vision_hidden_states=vision_hidden_states,
            return_vision_hidden_states=True,
            beam_size=3,
            temperature=0.7,
            repetition_penalty=1.1,
            length_penalty=3,
        )

        answer = res[0]["<ans>"]

        context += "User: " + question + "\n"
        context += "AI: " + answer + "\n"
        return answer, context, vision_hidden_states

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def run_image_safety_checker(self, image, device, dtype):
        if self.image_safety_checker is not None:
            image_safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(np.asarray(image)), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.image_safety_checker(
                images=image, clip_input=image_safety_checker_input.pixel_values.to(dtype)
            )
            if any(has_nsfw_concept):
                print(
                    "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                )
                for idx, _has_nsfw_concept in enumerate(has_nsfw_concept):
                    if _has_nsfw_concept:
                        image[idx] = np.zeros(image[idx].shape)  # black image
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

