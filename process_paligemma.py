from typing import Dict, List, Optional, Tuple, Union, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = {0.5, 0.5, 0.5}
IMAGENET_STANDARD_STD = {0.5, 0.5, 0.5}

def resize(image, size, resample, reducing_gap: Optional[int] = None) -> np.ndarray:
    h, w = size
    resized_img = image.resize((w, h), resample=resample, reducing_gap=reducing_gap)
    return resized_img

def rescale(image, scale, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_img = image * scale
    rescaled_img = rescaled_img.astype(dtype)
    return rescaled_img

def normalize(image, mean, std) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
        images: List[Image.Image],
        size: Dict[str, int] = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std:Optional[Union[float, List[float]]] = None 
) -> List[np.ndarray]:
    
    height, width = size[0], size[1]
    images = [resize(image=image, size=(height, width), resample=resample) for image in images]
    images = [np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    images = [image.transpose(2,0,1) for image in images] #[H,W,C] -> [C,H,W]

    return images


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_length, image_token):
    return f'{image_token * image_seq_length}{bos_token}{prefix_prompt}\n'


class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super.__init__()
        self.image_size = image_size
        self.image_seq_length = num_image_tokens

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add(tokens_to_add)
        
        EXTRA_TOKENS = [f'<loc{i:04d}' for i in range(1024)] # for object detection (bounding box coords)
        EXTRA_TOKENS += [f'<seg{i:03d}' for i in range(128)] # for segmentation
        tokenizer.add(EXTRA_TOKENS)
        self.image_token_id = tokenizer.conver_tokens_to_id(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(images, size = (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC,
                                      rescale_factor=1/255.0, img_mean=IMAGENET_STANDARD_MEAN, img_std=IMAGENET_STANDARD_STD)
        
        pixel_values = np.stack(pixel_values, axis=0) # [B, C, H, W]
        pixel_values = torch.tensor(pixel_values)

        # prepend self.image_seq_length number of image_tokens to the prompt
        input_strings = [add_image_tokens_to_prompt(prefix_prompt=prompt,
                                             bos_token=self.tokenizer.bos_token,
                                             image_seq_length=self.image_seq_length,
                                             image_token=self.IMAGE_TOKEN
                                             )
                        for prompt in text]
        
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding=padding, truncation=truncation)

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data

            


