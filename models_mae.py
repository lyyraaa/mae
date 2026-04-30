# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_dims, patch_dims, pos_encoding_params,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.img_dims = img_dims
        self.patch_dims = patch_dims
        self.token_grid_dims = np.floor_divide(img_dims,patch_dims)
        self.num_spatial_patches = np.prod(self.token_grid_dims[1:])  # number of tokens along the spatial dimensions
        self.num_spectral_patches = self.token_grid_dims[0]  # number of tokens along the spectral dimension
        self.num_patches = np.prod(self.token_grid_dims)  # total number of tokens in the spatial-spectral input

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.use_spatial_pos_encoding, self.use_spectral_pos_encoding, self.learnable_spatial_pos_encoding,self.learnable_spectral_pos_encoding = pos_encoding_params

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.Linear(in_features=np.prod(self.patch_dims), out_features=self.embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.encoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches, embed_dim), requires_grad=self.learnable_spatial_pos_encoding)
        self.encoder_spectral_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spectral_patches, embed_dim), requires_grad=self.learnable_spectral_pos_encoding)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches, decoder_embed_dim),requires_grad=self.learnable_spatial_pos_encoding)
        self.decoder_spectral_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spectral_patches, decoder_embed_dim),requires_grad=self.learnable_spectral_pos_encoding)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(self.decoder_embed_dim, np.prod(self.patch_dims), bias=True)  # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.learnable_spatial_pos_encoding:
            torch.nn.init.normal_(self.encoder_spatial_pos_embed, std=.02)
            torch.nn.init.normal_(self.decoder_spatial_pos_embed, std=.02)
        else:
            encoder_spatial_pos_embed = get_2d_sincos_pos_embed(self.embed_dim, int(self.num_spatial_patches**0.5),cls_token=False)
            decoder_spatial_pos_embed = get_2d_sincos_pos_embed(self.decoder_embed_dim, int(self.num_spatial_patches ** 0.5),cls_token=False)
            self.encoder_spatial_pos_embed.data.copy_(torch.from_numpy(encoder_spatial_pos_embed).float().unsqueeze(0))
            self.decoder_spatial_pos_embed.data.copy_(torch.from_numpy(decoder_spatial_pos_embed).float().unsqueeze(0))

        if self.learnable_spectral_pos_encoding:
            torch.nn.init.normal_(self.encoder_spectral_pos_embed, std=.02)
            torch.nn.init.normal_(self.decoder_spectral_pos_embed, std=.02)
        else:
            encoder_spectral_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.embed_dim, pos=np.arange(0, self.num_spectral_patches))
            decoder_spectral_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.decoder_embed_dim, pos=np.arange(0, self.num_spectral_patches))
            self.encoder_spectral_pos_embed.data.copy_(torch.from_numpy(encoder_spectral_pos_embed).float().unsqueeze(0))
            self.decoder_spectral_pos_embed.data.copy_(torch.from_numpy(decoder_spectral_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, prod(patch_dims))
        """
        B, C, H, W = imgs.shape # image batch size, height, width, depth
        pc, ph, pw = self.patch_dims # patch height, width, depth # todo could have this be an argument, set C = 1 for 2d map
        c, h, w = self.token_grid_dims # token map height, width, depth

        # B, c*pc, h*ph, w*pw -> B, c, pc, h, ph, w, pw, -> B, c, h, w, pc, ph, pw -> B, c*h*w, pc*ph*pw
        x = imgs.reshape(B, c, pc, h, ph, w, pw)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, c*h*w, pc*ph*pw)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, prod(patch_dims))
        imgs: (N, C, H, W)
        """
        B, L, D = x.shape
        pc, ph, pw = self.patch_dims # patch height, width, depth
        c, h, w = self.token_grid_dims  # token map height, width, depth

        # B, c*h*w, pc*ph*pw -> B, c, h, w, pc, ph, pw -> B, c, pc, h, ph, w, pw -> B, c*pc, h*ph, w*pw
        imgs = x.reshape(B, c, h, w, pc, ph, pw)
        imgs = imgs.permute(0,1,4,2,5,3,6).reshape(B, c*pc, h*ph, w*pw)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """
        Do forward pass of the encoder
        x: [N, L, D = ph*pw*pc], sequence
        """
        # embed patches
        x = self.patchify(x)
        x = self.patch_embed(x)
        c, h, w = self.token_grid_dims

        # add pos embed w/o cls token
        if self.use_spatial_pos_encoding:
            # spatial pos encoding size = [1, h*w, D]
            x = x + self.encoder_spatial_pos_embed.unsqueeze(1).repeat(1,c,1,1).flatten(start_dim=1,end_dim=2)
        if self.use_spectral_pos_encoding:
            # spectral pos encoding size = [1, c, D]
            x = x + self.encoder_spectral_pos_embed.unsqueeze(2).repeat(1,1,h*w,1).flatten(start_dim=1,end_dim=2) # todo if this doesn't work, maybe do each h and w dim separaesly

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        c, h, w = self.token_grid_dims

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed w/o cls token
        if self.use_spatial_pos_encoding:
            # spatial pos encoding size = [1, h*w, D]
            x[:,1:] = x[:,1:] + self.decoder_spatial_pos_embed.unsqueeze(1).repeat(1, c, 1, 1).flatten(start_dim=1, end_dim=2)
        if self.use_spectral_pos_encoding:
            # spectral pos encoding size = [1, c, D]
            x[:,1:] = x[:,1:] + self.decoder_spectral_pos_embed.unsqueeze(2).repeat(1, 1, h*w, 1).flatten(start_dim=1, end_dim=2)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        x = self.decoder_pred(x) # B, L, D -> B, L, pc*ph*pw
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [B, C, H, W]
        pred: [B, L, pc*ph*pw]
        mask: [B, L], 0 is keep, 1 is remove,
        target: [B, L=c*h*w, pc*ph*pw]
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B, L, D]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class MaskedImageModellingViT(MaskedAutoencoderViT):
    """ Masked Image modelling with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, input_norm=True, decoder_pred_intermediate=32):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                 embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss, input_norm=input_norm, decoder_pred_intermediate=decoder_pred_intermediate)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # keep the first subset
        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], 1)
        x_masked = torch.where(mask.unsqueeze(-1)==0, x, mask_tokens)

        return x_masked, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        N, L, D  = x.shape
        x = self.decoder_pred_coarse(x).reshape(
            N, L, self.patch_size, self.patch_size,self.decoder_pred_intermediate
        ) # N x L x D -> N x L x patch_dim x patch_dim x decoder_pred_intermediate
        x = self.decoder_pred_fine(x).reshape(
            N, L, self.patch_size * self.patch_size * self.in_chans
        ) # N x L x patch_dim x patch_dim x decoder_pred_intermediate -> N x L x (patch_dim * patch_dim * wavenumber_channels)

        return x

