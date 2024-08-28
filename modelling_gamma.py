import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modelling_siglip import SiglipVisionConfig, SiglipVisionModel


class GemmaConfig():
     
     def __init__(self, vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attentions_heads, num_key_value_heads, head_dim=256, max_position_embeddings=8192,
                  rms_norm_eps=1e-6, rope_theta=10000.0, attention_bias=False, attention_dropout=0.0, pad_token_id=None, **kwargs):
          super.__init__()
          self.vocab_size = vocab_size
          self.max_position_embeddings = max_position_embeddings
          self.hidden_size = hidden_size
          self.intermediate_size = intermediate_size
          self.num_hidden_layers = num_hidden_layers
          self.num_attentions_heads = num_attentions_heads
          self.num_key_value_heads = num_key_value_heads
          self.rms_norm_eps = rms_norm_eps
          self.rope_theta = rope_theta
          self.head_dim = head_dim
          self.attention_bias = attention_bias
          self.attention_dropout = attention_dropout
          self.pad_token_id = pad_token_id
          

class PaliGemmaConfig():
     
     def __init__(self, vision_config=None, text_config=None, ignore_index=-100, img_token_idx=256000, vocab_size=257152, projection_dim=2048, hidden_dim=2048, pad_token_id=None, **kwargs):
          super.__init__()
          self.ignore_index = ignore_index
          self.image_token_index = img_token_idx
          self.vocab_size = vocab_size
          self.projection_dim = projection_dim
          self.hidden_size = hidden_dim
          self.text_config = text_config
          self.is_encoder_decoder = False
          self.pad_token_id = pad_token_id

          self.vision_config = SiglipVisionConfig(**vision_config)
          self.text_config = text_config

          self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
          self.vocab_size = self.text_config.vocab_size

          self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
          self.vision_config.projection_dim = projection_dim


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        self.language_model.tie_weights()


    def _merge_input_ids_with_image_features(self, image_features, input_embeds, input_ids, attention_mask, kv_cache):

        _,_, embed_dim = image_features.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        final_embedding = torch.zeros(batch_size, seq_len, embed_dim, dtype=dtype, device=device)

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = (input_ids == self.config.image_token_index)
        pad_mask = (input_ids == self.config.pad_token_id)

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        final_embedding = torch.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        





    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
            
            assert torch.all(attention_mask == 1), "The input cannot be padded"

            input_embeds = self.language_model.get_input_embeddings()(input_ids) # text

            img_features = self.vision_tower(pixel_values.to(input_embeds.dtype))
            img_features = self.multi_modal_projector(img_features) # convert to text embedding_dim

            input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(img_features, input_embeds, input_ids, attention_mask, kv_cache)

            outputs = self.language_model(
                 attention_mask=attention_mask,
                 position_ids=position_ids,
                 input_embeds=input_embeds,
                 kv_cache=kv_cache
            )

            return outputs
