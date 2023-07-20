import contextlib
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import torch
from transformers.models.clip.modeling_clip import CLIPOutput, clip_loss
from transformers import CLIPConfig, CLIPModel, PreTrainedModel, PretrainedConfig

from .modeling_vq_adapter import VQAdapterModel, VQAdapterConfig


@dataclass
class VQCLIPOutput(CLIPOutput):
    text_codes: torch.LongTensor = None
    image_codes: torch.LongTensor = None
    quantization_loss: torch.FloatTensor = None
    contrastive_loss: torch.FloatTensor = None
    perplexity: torch.FloatTensor = None


class VQCLIPConfig(PretrainedConfig):
    model_type = "VQCLIP"

    def __init__(
        self,
        clip_config_dict: dict = CLIPConfig().to_dict(),
        vision_vq_adapter_config_dict: Optional[dict] = VQAdapterConfig().to_dict(),
        text_vq_adapter_config_dict: Optional[dict] = None,
        **kwargs,
    ):
        self.clip_config_dict = clip_config_dict
        self.vision_vq_adapter_config_dict = vision_vq_adapter_config_dict
        self.text_vq_adapter_config_dict = text_vq_adapter_config_dict
        super().__init__(**kwargs)


class VQCLIPModel(PreTrainedModel):
    config_class = VQCLIPConfig

    def __init__(self, config: VQCLIPConfig):
        super().__init__(config)

        self.clip_config = CLIPConfig.from_dict(config.clip_config_dict)
        self.clip_model = CLIPModel(self.clip_config)

        self.vision_vq_adapter, self.text_vq_adapter = None, None

        if config.vision_vq_adapter_config_dict:
            self.vision_vq_adapter = VQAdapterModel(
                VQAdapterConfig.from_dict(config.vision_vq_adapter_config_dict)
            )
        if config.text_vq_adapter_config_dict:
            self.text_vq_adapter = VQAdapterModel(
                VQAdapterConfig.from_dict(config.text_vq_adapter_config_dict)
            )

    def _init_weights(self, module):
        pass

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_codes: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, Union[torch.FloatTensor, None]]]:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.clip_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.clip_config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.clip_config.use_return_dict
        )

        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.clip_model.text_projection(pooled_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.text_vq_adapter:
            _res = self.text_vq_adapter(text_features)
            text_features = _res['z']
            text_codes = _res['codes']
        else:
            text_codes = None

        if return_codes:
            return text_features, text_codes

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_codes: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, Union[torch.FloatTensor, None]]]:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.clip_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.clip_config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.clip_config.use_return_dict
        )

        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.clip_model.visual_projection(pooled_output)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.vision_vq_adapter:
            _res = self.vision_vq_adapter(image_features)
            image_features = _res['z']
            image_codes = _res['codes']
        else:
            image_codes = None

        if return_codes:
            return image_features, image_codes

        return image_features

    @property
    def logit_scale(self):
        return self.clip_model.logit_scale

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_perplexity: Optional[bool]=None,
    ) -> Union[Tuple, VQCLIPOutput]:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.clip_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.clip_config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.clip_config.use_return_dict
        )

        vision_outputs = self.clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.clip_model.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.clip_model.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # quantization
        image_codes, text_codes = None, None
        perplexity = 0.0
        vq_loss = 0.0
        if self.text_vq_adapter:
            res = self.text_vq_adapter(text_embeds, return_perplexity=return_perplexity)
            text_embeds = res["z"]
            text_codes = res["codes"]
            perplexity += res["perplexity"] if return_perplexity else 0.0
            vq_loss += res["loss"]
        if self.vision_vq_adapter:
            res = self.vision_vq_adapter(image_embeds, return_perplexity=return_perplexity)
            image_embeds = res["z"]
            image_codes = res["codes"]
            perplexity += res["perplexity"] if return_perplexity else 0.0
            vq_loss += res["loss"]
        if self.vision_vq_adapter and self.text_vq_adapter:
            # averages
            perplexity = perplexity / 2.0
            vq_loss = vq_loss / 2.0

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        total_loss = None
        if loss is not None:
            total_loss = loss + vq_loss

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                vision_outputs,
            )
            return ((loss,) + output) if loss is not None else output

        return VQCLIPOutput(
            loss=total_loss,
            quantization_loss=vq_loss,
            contrastive_loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            text_codes=text_codes,
            image_embeds=image_embeds,
            image_codes=image_codes,
        )

    @staticmethod
    def from_pretrained_clip(clip_path: str, vision_vq_adapter_path: Optional[str] = None, text_vq_adapter_path: Optional[str] = None):
        """
        load only the adapter from the vq_clip_path, and load the clip model
        from the clip_path
        """

        assert not (text_vq_adapter_path is None and vision_vq_adapter_path is None)

        clip_config = CLIPConfig.from_pretrained(clip_path).to_dict()
        if vision_vq_adapter_path is not None:
            vision_vq_config = VQAdapterConfig.from_pretrained(vision_vq_adapter_path).to_dict()
        else: vision_vq_config = None
        if text_vq_adapter_path is not None:
            text_vq_config = VQAdapterConfig.from_pretrained(text_vq_adapter_path).to_dict()
        else: text_vq_config = None

        vq_clip_config = VQCLIPConfig(clip_config_dict=clip_config, vision_vq_adapter_config_dict=vision_vq_config, text_vq_adapter_config_dict=text_vq_config)

        init_provider = contextlib.suppress
        try:
            from accelerate import init_empty_weights
            init_provider = init_empty_weights
        except ImportError:
            print("Could not do vq-clip lazy init")

        with init_provider():
            vq_clip = VQCLIPModel(vq_clip_config)

        clip: CLIPModel = CLIPModel.from_pretrained(clip_path)
        vq_clip.clip_model = clip

        if vision_vq_adapter_path is not None:
            vision_vq_adapter = VQAdapterModel.from_pretrained(vision_vq_adapter_path)
            vq_clip.vision_vq_adapter = vision_vq_adapter
        if text_vq_adapter_path is not None:
            text_vq_adapter = VQAdapterModel.from_pretrained(text_vq_adapter_path)
            vq_clip.text_vq_adapter = text_vq_adapter

        return vq_clip
