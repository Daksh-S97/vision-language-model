import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modelling_siglip import SiglipVisionConfig, SiglipVisionModel


class KVCache:

     def __init__(self) -> None:
          self.key_cache: List[torch.Tensor] = []
          self.value_cache: List[torch.Tensor] = []

     def num_items(self) -> int:
          if len(self.key_cache) == 0:
               return 0
          else:
               # shape of KVCache-> [b, kv_heads, seq_len, head_dim] (Return seq_len)
               return self.key_cache.shape[-2]     

     def update(
               self,
               k: torch.Tensor,
               v:torch.Tensor,
               layer_idx: int,
     ) -> Tuple[torch.Tensor, torch.Tensor]:
          
          if len(self.key_cache) <= layer_idx: # not added anything for this layer
               self.key_cache.append(k)
               self.value_cache.append(v)
          else:
               self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=-2)
               self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=-2)

          return self.key_cache[layer_idx], self.value_cache[layer_idx]

          

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


class PaliGemmaMultiModalProjector(nn.Module):
     
     def __init__(self, config:PaliGemmaConfig) -> None:
          super().__init__()
          self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

     def forward(self, image_features):
          hidden_states = self.linear(image_features)
          return hidden_states


# faster than layer_norm because calc. only 1 statistic (RMS)
class GemmaRMSNorm(nn.Module):
     
     def __init__(self, dim: int, eps: float = 1e-6) -> None:
          super().__init__()
          self.eps = eps
          self.weight = nn.Parameter(torch.zeros(dim))

     #torch.rsqrt = reciprocal of sqrt
     # eps to prevent division by 0
     def _norm(self, x):
          return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

     def forward(self, x):
          output = self._norm(x.float())
          output = output * (1.0 + self.weight.float())
          return output.type_as(x)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
     if n_rep == 1:
          return hidden_states
     b, kv_heads, seq_len, head_dim = hidden_states.shape

     # repeat eahc (s x d) matrix n_rep times to get kv_heads = q_heads
     hidden_states = hidden_states[:, :, None, :, :].expand(b, kv_heads, n_rep, seq_len, head_dim)
     return hidden_states.reshape(b, kv_heads * n_rep, seq_len, head_dim)



class GemmaMLP(nn.Module):

     def __init__(self, config: GemmaConfig) -> None:
          super().__init__()
          self.config = config
          self.hidden_size = config.hidden_size
          self.intermediate_size = config.intermediate_size
          self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
          self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
          self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

     def forward(self, x):
          return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)) 
         


class GemmaAttention(nn.Module):

     def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None) -> None:
          super().__init__()
          self.config = config
          self.layer_idx = layer_idx
          self.attention_dropout = config.attention_dropout
          self.hidden_size = config.hidden_size
          self.num_heads = config.num_attentions_heads
          self.head_dim = config.head_dim
          self.num_key_value_heads = config.num_key_value_heads
          self.num_key_value_groups = self.num_heads // self.num_key_value_heads
          self.max_position_embeddings = config.max_position_embeddings
          self.rope_theta = config.rope_theta
          self.is_causal = True

          assert self.hidden_size % self.num_heads == 0

          self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
          self.k_proj - nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
          self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
          self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

          self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

     def forward(
               self,
               hidden_states:torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.LongTensor] = None,
               kv_cache: Optional[KVCache] = None,
               **kwargs
     )-> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
          
          b, q_len, _ = hidden_states.shape
          q = self.q_proj(hidden_states)
          k = self.k_proj(hidden_states)
          v = self.v_proj(hidden_states)

          q = q.view(b, q_len, self.num_heads, self.head_dim).transpose(1,2)
          k = k.view(b, q_len, self.num_key_value_heads, self.head_dim).tranpose(1,2)
          v = v.view(b, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

          cos, sin = self.rotary_emb(v, position_ids, seq_len=None)
          q, k = apply_rotary_pos_emb(q, k, cos, sin)

          if kv_cache is not None:
               k, v = kv_cache.update(k, v, self.layer_idx)

          # repeat k and v heads to match number of q heads; basically, reversing GQA
          #  while using custom CUDA kernels, do not have to repeat head as each kv head for a group is copied into the local GPU memory so it can be reused
          k = repeat_kv(k, self.num_key_value_groups)
          v = repeat_kv(v, self.num_key_value_groups)

          # [b, q_heads, s_q, head_dim] X [b, q_heads, head_dim, s_kv] -> [b, q_heads, s_q, s_kv]
          attn_scores = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.head_dim)

          assert attention_mask is not None
          attn_scores = attn_scores + attention_mask

          attn_scores = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
          attn_scores = nn.functional.dropout(attn_scores, p=self.config.attention_dropout, training=self.training) # only apply during training
          
          # [b, q_heads, s_q, s_kv] X [b, q_heads, s_kv, head_dim] -> [b, q_heads, s_q, head_dim]
          attn_out = torch.matmul(attn_scores, v)

          if attn_out.size != (b, self.num_heads, q_len, self.head_dim):
               raise ValueError(
                    f"attn_output should be of size {(b, self.num_heads, q_len, self.head_dim)}, but is"
                    f"{attn_out.size()}"
               )
          
          attn_out = attn_out.transpose(1,2).contiguous()
          attn_out = attn_out.view(b, q_len, -1)
          attn_out = self.out_proj(attn_out)

          return attn_out, attn_scores




class GemmaDecoderLayer(nn.Module):
     
     def __init__(self, config:GemmaConfig, layer_idx: int) -> None:
          super().__init__()
          self.hidden_size = config.hidden_size
          self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
          self.mlp = GemmaMLP(config)

          self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
          self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

     def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
          residual = hidden_states
          hidden_states = self.input_layernorm(hidden_states)
          hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)

          hidden_states = residual + hidden_states

          residual = hidden_states
          hidden_states = self.post_attention_layernorm(hidden_states)
          hidden_states = self.mlp(hidden_states)

          hidden_states = residual + hidden_states

          return hidden_states
     
     

class GemmaModel(nn.Module):

     def __init__(self, config:GemmaConfig) -> None:
          super().__init__()
          self.config = config
          self.padding_idx = config.pad_token_id
          self.vocab_size = config.vocab_size

          self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
          self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
          self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

     def get_input_embeddings(self):
          return self.embed_tokens


     def forward(
               self,
               attention_mask: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.LongTensor] = None,
               input_embeds: Optional[torch.FloatTensor] = None,
               kv_cache: Optional[KVCache] = None,
     ) -> torch.FloatTensor:
          
          hidden_states = input_embeds
          normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
          hidden_states = hidden_states * normalizer

          for decoder_layer in self.layers:
               hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)

          hidden_states = self.norm(hidden_states)

          return hidden_states
     


class GemmaForCausalLM(nn.Module):
     
     def __init__(self, config) -> None:
          super().__init__()
          self.config = config
          self.model = GemmaModel(config)
          self.vocab_size = config.vocab_size
          self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

     def get_input_embeddings(self):
          return self.model.embed_tokens

     def tie_weights(self):
          self.lm_head.weight = self.model.embed_tokens.weight

     def forward(
               self,
               attention_mask: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.LongTensor] = None,
               input_embeds: Optional[torch.FloatTensor] = None,
               kv_cache: Optional[KVCache] = None,
     ) -> Tuple:
          
          outputs = self.model(attention_mask=attention_mask, position_ids=position_ids, input_embeds=input_embeds, kv_cache=kv_cache)

          hidden_states = hidden_states
          logits = self.lm_head(hidden_states)
          logits = logits.float()

          return_data = {"logits": logits}
          if kv_cache is not None:
               return_data["kv_cache"] = kv_cache

          return return_data     
                   



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
          

        # MASK: 0 -> not masked, -inf -> masked (we're not masking anything)
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # 1st token
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            assert q_len == 1 # query values are fed one-by-one
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        # [b, q_len, kv_len] -> [b, num_heads_q, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                 position_ids - position_ids.unsqueeze(0)

        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)


        return final_embedding, causal_mask, position_ids




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
