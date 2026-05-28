from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from models.utils import bitsandbytes_8bit_config
from models.vlm_wrapper import VLMWrapperCaptioning


def init_llava(
        model_config: Dict[str, Any],
        device: str = "cuda",
        use_8bit: bool = False
    ):
    # bitsandbytes 8-bit loading is GPU-oriented; disable it automatically on CPU.
    use_8bit = use_8bit and torch.cuda.is_available() and str(device) != "cpu"
    model = model_config["model_class"].from_pretrained(
        model_config["model_id"],
        quantization_config=bitsandbytes_8bit_config() if use_8bit else None
    )
    model = model.to(device) if not use_8bit else model
    processor = model_config["processor_class"].from_pretrained(model_config["model_id"])

    vlm_wrapper = model_config["wrapper_class"](model=model, processor=processor)
    return vlm_wrapper

@dataclass
class LLaVaWrapper(VLMWrapperCaptioning):
    model: Any = field(
        default_factory=lambda: LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            device_map={"": 0},
            torch_dtype=torch.float16
        )
    )
    processor: Any = field(
        default_factory=lambda: AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        )
    )

    def __post_init__(self):
        self.processor.tokenizer.padding_side = "left"

    def process_inputs(self, apply_template=True, **kwargs):
        required_keys = {'image', 'prompt'}
        if not required_keys.issubset(kwargs.keys()):
            raise ValueError(f"Missing required arguments: {required_keys - set(kwargs.keys())}")

        if apply_template:
            prompts = []
            for prompt in kwargs["prompt"]:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )
                prompts.append(text)
        else:
            prompts = kwargs["prompt"]

        inputs = self.processor(
            images=kwargs['image'],
            text=prompts,
            padding=True,
            return_tensors="pt"
        )

        # Fix potential extra image dimension only when it's a singleton extra channel.
        # Llava's processor may return pixel_values as (B, N, C, H, W), where N can be >1.
        # In that case, the model expects the full 5D tensor and should not collapse it.
        pixel_values = inputs.get("pixel_values", None)

        if pixel_values is not None:
            if pixel_values.ndim == 5:
                if pixel_values.shape[1] == 1:
                    pixel_values = pixel_values[:, 0]
                # Otherwise keep the actual multi-patch 5D tensor.
            elif pixel_values.ndim == 4:
                pass
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

            inputs["pixel_values"] = pixel_values

        return inputs.to(self.model.device)

    def decode(self, outputs, **kwargs):
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        clean_up_tokenization_spaces = kwargs.get('clean_up_tokenization_spaces', False)
        return self.processor.batch_decode(
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

    def generate(self, inputs: Dict[str, Any], **kwargs) -> Any:
        max_new_tokens = kwargs.get('max_new_tokens', 100)

        return self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )