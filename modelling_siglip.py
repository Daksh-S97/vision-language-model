from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
            self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_layers=12, num_channels=3, img_size=224, patch_size=16,
            layer_norm_eps=1e-6, attention_dropout=0.0, num_image_tokens: int = None, **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_layers = num_attention_layers
        self.num_channels = num_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.img_size = config.img_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, 
                                         stride=self.patch_size, padding="valid")
        
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.pos_embeds = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1,-1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        b, c, h, w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)   # [B, Embed_dim, patches_h, patches_w]
        embeddings = patch_embeds.flatten(2)    # [B, embed_dim, #patches]
        embeddings = embeddings.transpose(1, 2)     # [B, #patches, embed_dim]
        embeddings += self.pos_embeds(self.position_ids)

        return embeddings


class SiglipMLP(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh") 
        hidden_states = self.fc2(hidden_states) 
        return hidden_states

class SiglipAttention(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)    


class SiglipEncoder(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # CHANGED: check if error (vid timestamp - 1:13:05)
        #resd = hidden_states
        x = self.layer_norm1(hidden_states)
        x, _ = self.self_attn(hidden_states=x)
        x += hidden_states
        x2 = self.layer_norm2(x)
        x2 = self.mlp(x2)
        x += x2
        return x    



class SiglipVisionTransformer(nn.Module):

    def __init__(self, config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(self.config) # get (embeddings of patches + pos_enc)
        self.encoder = SiglipEncoder(self.config)
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, #Patches, Embed_dim]
        patch_embeds = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds=patch_embeds)
        last_hidden_state = self.post_layer_norm(last_hidden_state)

        return last_hidden_state



class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(self.config)

    def forward(self, pixel_values) -> Tuple:
        # [B, C, H, W] -> [B, #Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)    