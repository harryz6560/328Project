import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbeddings(nn.Module):
    """TODO: (0.5 out of 8) Calculates patch embedding
    of shape `(batch_size, seq_length, hidden_size)`.
    """
    def __init__(
        self, 
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_channels: int = 3,      # 3 for RGB, 1 for Grayscale
        ):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################
        embed_dem = hidden_size
        self.projection = nn.Conv2d(in_channels = num_channels, out_channels = hidden_size, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) **2
        
        # #########################

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

    def forward(
        self, 
        x: torch.Tensor,
        ) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # #########################
        # Finish Your Code HERE
        # #########################
        if torch.cuda.is_available():
            x = x.cuda()

        # Calculate Patch Embeddings, then flatten into
        # batched 1D sequence (batch_size, seq_length, hidden_size)
        embeddings = self.projection(x)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # print(f"embeddings: {embeddings.shape}")
        # #########################
        return embeddings

class PositionEmbedding(nn.Module):
    def __init__(
        self,
        num_patches,
        hidden_size,
        ):
        """TODO: (0.5 out of 8) Given patch embeddings, 
        calculate position embeddings with [CLS] and [POS].
        """
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################
        
        # Specify [CLS] and positional embedding as learnable parameters

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches+1, hidden_size))

        # #########################

    def forward(
        self,
        embeddings: torch.Tensor
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################
        if torch.cuda.is_available():
            embeddings = embeddings.cuda()
        
        # Concatenate [CLS] token with embedded patch tokens
        cls_token = self.cls_token.expand(embeddings.size(0), -1, -1)
        # print(f"embeddings: {embeddings.shape}")
        # print(f"position_embeddings: {cls_token.shape}")
        embeddings = torch.cat((cls_token, embeddings), dim=1)
        
        # Then add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        # #########################
        return embeddings

class GELU(nn.Module):
    def forward(self, x: torch.Tensor):
        if torch.cuda.is_available():
            x = x.cuda()
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """TODO: (0.25 out of 8) Residual Attention Block.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.attn = nn.MultiheadAttention(d_model, n_head)    # Refer to nn.MultiheadAttention
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            GELU(),
            nn.Linear(4 * d_model, d_model))     # A trick is to use nn.Sequential to specify multiple layers at once
        self.ln_2 = nn.LayerNorm(d_model)

        # #########################

    def forward(self, x: torch.Tensor):

        # #########################
        # Finish Your Code HERE
        # #########################
        # LayerNorm -> Multi-head Attention
        if torch.cuda.is_available():
            x = x.cuda()
        attn_output, _ = self.attn(x, x, x)  # Q, K, V all set to x
        # Residual connection against x
        x = x + attn_output
        x = self.ln_1(x)

        # LayerNorm -> MLP Block
        mlp_output = self.mlp(x)
        # Residual connection against x
        out = x + mlp_output
        out = self.ln_2(out)

        # #########################

        return out

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        if torch.cuda.is_available():
            x = x.cuda()
        x = x.permute(1, 0, 2)  # (batch_size, seqlen, dim) -> (seqlen, batch_size, dim)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # (seqlen, batch_size, dim) -> (batch_size, seqlen, dim)
        return x

class ViT(nn.Module):
    """TODO: (0.5 out of 8) Vision Transformer.
    """
    def __init__(
        self, 
        image_size: int, 
        patch_size: int, 
        num_channels: int,
        hidden_size: int, 
        layers: int, 
        heads: int,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.patch_embed = PatchEmbeddings(image_size, patch_size, hidden_size, num_channels)

        self.pos_embed = PositionEmbedding(self.patch_embed.num_patches, hidden_size)

        self.ln_pre = nn.LayerNorm(hidden_size)

        self.transformer = Transformer(hidden_size, layers, heads)     # TODO: Use the provided transformer codeblock

        self.ln_post = nn.LayerNorm(hidden_size)

        # #########################


    def forward(self, x: torch.Tensor):
        # #########################
        # Finish Your Code HERE
        # #########################
        if torch.cuda.is_available():
            x = x.cuda()
        
        x = self.patch_embed(x)
        # Position Embedding
        x = self.pos_embed(x)
        # LayerNorm before transformer
        x = self.ln_pre(x)
        # Transformer
        x = self.transformer(x)
        # reduce size of (batchsize, hidden size)
        x = torch.mean(x, dim=1)
        # #########################
        return x

class ClassificationHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_classes: int = 10,
        ):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        if torch.cuda.is_available():
            feats = feats.cuda()
        out = self.classifier(feats)
        return out

class LinearEmbeddingHead(nn.Module):
    """TODO: (0.25 out of 8) Given features from ViT, generate linear embedding vectors.
    """
    def __init__(
        self, 
        hidden_size: int,
        embed_size: int = 64,
        ):
        super().__init__()
        self.embed_size = embed_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.projection = nn.Linear(hidden_size, embed_size)
        # #########################

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################
        if torch.cuda.is_available():
            feats = feats.cuda()
        out = self.projection(feats)
        # out = torch.mean(out, dim=1)    # reduce size
        # #########################
        return out