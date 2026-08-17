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

    def __init__(self, img_dims, patch_dims, pos_encoding_params,learn_bg_encoding=False,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path=0.0):
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
        self.learn_bg_encoding = learn_bg_encoding

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.Linear(in_features=np.prod(self.patch_dims), out_features=self.embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.encoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches, embed_dim), requires_grad=self.learnable_spatial_pos_encoding)
        self.encoder_spectral_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spectral_patches, embed_dim), requires_grad=self.learnable_spectral_pos_encoding)

        self.background_embed = nn.Parameter(
            torch.zeros(1, self.img_dims[0], 1, 1), requires_grad=self.learn_bg_encoding)

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

        if self.learn_bg_encoding:
            torch.nn.init.normal_(self.background_embed, std=.02)

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

    def reset_pos_encodings(self, new_image_size, device): # todo this should be tidied if here to stay
        pc, ph, pw = self.patch_dims
        # Get 2d sincos embedding for the new grid size implied by new image size
        new_pos_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size=new_image_size // ph,
                                                   cls_token=False)
        new_decoder_pos_encoding = get_2d_sincos_pos_embed(self.decoder_embed_dim,
                                                           grid_size=new_image_size // ph, cls_token=False)

        # Set position embed to this new encoding
        self.encoder_spatial_pos_embed.data = torch.from_numpy(new_pos_encoding).float().unsqueeze(0).to(device)
        self.decoder_spatial_pos_embed.data = torch.from_numpy(new_decoder_pos_encoding).float().unsqueeze(0).to(device)

        # Reset model expected image size
        self.token_grid_dims = [self.token_grid_dims[0], new_image_size // ph, new_image_size // pw]
        self.img_dims = [self.img_dims[0], new_image_size, new_image_size]

        self.num_spatial_patches = np.prod(self.token_grid_dims[1:])  # number of tokens along the spatial dimensions
        self.num_spectral_patches = self.token_grid_dims[0]  # number of tokens along the spectral dimension
        self.num_patches = np.prod(self.token_grid_dims)  # total number of tokens in the spatial-spectral input

        self.patch_embed.strict_img_size = False

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

    def encode_background(self, imgs, tissue_mask):
        """
        imgs: [B, C, H, W]
        tissue_mask: [B, 1, H, W]
        """
        if not self.learn_bg_encoding: return imgs
        B, C, H, W = imgs.shape

        # place learnable background vector at all non-tissue pixels
        encoded_bg = (1-tissue_mask) * self.background_embed.expand(B, -1, H, W)

        # Zero pixels in background, then add the encoded vectors
        imgs = (imgs * tissue_mask) + encoded_bg

        return imgs

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

    def encode(self, x, finalnorm=True):
        """
        Do forward pass of the encoder, without any of the pretraining masking
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

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if finalnorm:
            x = self.norm(x)

        return x

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

    def forward(self, imgs, tissue_mask, mask_ratio=0.75):
        imgs = self.encode_background(imgs, tissue_mask) # todo try moving this inside the next line, so encoded bg is not predicted
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B, L, D]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class MaskedAutoencoderViT_RepeatSample(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_dims, patch_dims, pos_encoding_params,learn_bg_encoding=False,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path=0.0, sample_repeat=None):
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
        self.learn_bg_encoding = learn_bg_encoding
        self.sample_repeat = sample_repeat

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.Linear(in_features=np.prod(self.patch_dims), out_features=self.embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.encoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches, embed_dim), requires_grad=self.learnable_spatial_pos_encoding)
        self.encoder_spectral_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spectral_patches, embed_dim), requires_grad=self.learnable_spectral_pos_encoding)

        self.background_embed = nn.Parameter(
            torch.zeros(1, self.img_dims[0], 1, 1), requires_grad=self.learn_bg_encoding)

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

        if self.learn_bg_encoding:
            torch.nn.init.normal_(self.background_embed, std=.02)

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

    def reset_pos_encodings(self, new_image_size, device): # todo this should be tidied if here to stay
        pc, ph, pw = self.patch_dims
        # Get 2d sincos embedding for the new grid size implied by new image size
        new_pos_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size=new_image_size // ph,
                                                   cls_token=False)
        new_decoder_pos_encoding = get_2d_sincos_pos_embed(self.decoder_embed_dim,
                                                           grid_size=new_image_size // ph, cls_token=False)

        # Set position embed to this new encoding
        self.encoder_spatial_pos_embed.data = torch.from_numpy(new_pos_encoding).float().unsqueeze(0).to(device)
        self.decoder_spatial_pos_embed.data = torch.from_numpy(new_decoder_pos_encoding).float().unsqueeze(0).to(device)

        # Reset model expected image size
        self.token_grid_dims = [self.token_grid_dims[0], new_image_size // ph, new_image_size // pw]
        self.img_dims = [self.img_dims[0], new_image_size, new_image_size]

        self.num_spatial_patches = np.prod(self.token_grid_dims[1:])  # number of tokens along the spatial dimensions
        self.num_spectral_patches = self.token_grid_dims[0]  # number of tokens along the spectral dimension
        self.num_patches = np.prod(self.token_grid_dims)  # total number of tokens in the spatial-spectral input

        self.patch_embed.strict_img_size = False

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        if self.training and self.sample_repeat is not None: # todo this is messy, duplicated code
            N, L, D = x.shape  # batch, length, dim
            x = x.unsqueeze(1).expand(-1,self.sample_repeat,-1,-1).reshape(N*self.sample_repeat, L, D)
            N, L, D = x.shape  # batch, length, dim

            len_keep = int(L * (1 - mask_ratio))

            # REDO AFTER LUNCH WITH JUST MAKING BIGGER NOISE ARRAY
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return x_masked, mask, ids_restore
        else:
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))

            # REDO AFTER LUNCH WITH JUST MAKING BIGGER NOISE ARRAY
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return x_masked, mask, ids_restore

    def encode_background(self, imgs, tissue_mask):
        """
        imgs: [B, C, H, W]
        tissue_mask: [B, 1, H, W]
        """
        if not self.learn_bg_encoding: return imgs
        B, C, H, W = imgs.shape

        # place learnable background vector at all non-tissue pixels
        encoded_bg = (1-tissue_mask) * self.background_embed.expand(B, -1, H, W)

        # Zero pixels in background, then add the encoded vectors
        imgs = (imgs * tissue_mask) + encoded_bg

        return imgs

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

    def encode(self, x, finalnorm=True):
        """
        Do forward pass of the encoder, without any of the pretraining masking
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

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if finalnorm:
            x = self.norm(x)

        return x

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

        if self.training and self.sample_repeat is not None:
            B,L,D = target.shape
            target = target.unsqueeze(1).expand(-1, self.sample_repeat, -1, -1).reshape(B*self.sample_repeat, L, D)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, tissue_mask, mask_ratio=0.75):
        imgs = self.encode_background(imgs, tissue_mask) # todo try moving this inside the next line, so encoded bg is not predicted
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B, L, D]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class MaskedAutoencoderViT_ConvComplex(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_dims, patch_dims, pos_encoding_params,learn_bg_encoding=False,
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
        self.learn_bg_encoding = learn_bg_encoding

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

        self.background_embed = nn.Parameter(
            torch.zeros(1, self.img_dims[0], 1, 1), requires_grad=self.learn_bg_encoding)

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
        # Conv test stuff

        self.in_conv = self.DoubleConv(self.img_dims[0], 64) # B 405 224 224 -> B 64 224 224
        self.down1 = self.Down(64, 128) # B 64 224 224 -> B 128 112 112
        self.down2 = self.Down(128, 256)  # B 128 112 112 -> B 256 56 56
        self.down3 = self.Down(256, 512)  # B B 256 56 56 -> B 512 28 28
        self.down4 = self.Down(512, 768)  # B 512 28 28 -> B 768 14 14 # todo embed dim


        self.up1 = self.Up(768, 512)
        self.up2 = self.Up(512, 256)
        self.up3 = self.Up(256, 128)
        self.up4 = self.Up(128, 64)

        self.out_conv = nn.Conv2d(64, 7, 1)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()


    class DoubleConv(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        """Downscaling with maxpool then double conv"""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                MaskedAutoencoderViT_ConvComplex.DoubleConv(in_channels, out_channels)
            )

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        """Upscaling then double conv"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = MaskedAutoencoderViT_ConvComplex.DoubleConv(out_channels * 2, out_channels)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                              diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)



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

        if self.learn_bg_encoding:
            torch.nn.init.normal_(self.background_embed, std=.02)

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

    def reset_pos_encodings(self, new_image_size, device): # todo this should be tidied if here to stay
        pc, ph, pw = self.patch_dims
        # Get 2d sincos embedding for the new grid size implied by new image size
        new_pos_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size=new_image_size // ph,
                                                   cls_token=False)
        new_decoder_pos_encoding = get_2d_sincos_pos_embed(self.decoder_embed_dim,
                                                           grid_size=new_image_size // ph, cls_token=False)

        # Set position embed to this new encoding
        self.encoder_spatial_pos_embed.data = torch.from_numpy(new_pos_encoding).float().unsqueeze(0).to(device)
        self.decoder_spatial_pos_embed.data = torch.from_numpy(new_decoder_pos_encoding).float().unsqueeze(0).to(device)

        # Reset model expected image size
        self.token_grid_dims = [self.token_grid_dims[0], new_image_size // ph, new_image_size // pw]
        self.img_dims = [self.img_dims[0], new_image_size, new_image_size]

        self.num_spatial_patches = np.prod(self.token_grid_dims[1:])  # number of tokens along the spatial dimensions
        self.num_spectral_patches = self.token_grid_dims[0]  # number of tokens along the spectral dimension
        self.num_patches = np.prod(self.token_grid_dims)  # total number of tokens in the spatial-spectral input

        self.patch_embed.strict_img_size = False

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

    def encode_background(self, imgs, tissue_mask):
        """
        imgs: [B, C, H, W]
        tissue_mask: [B, 1, H, W]
        """
        if not self.learn_bg_encoding: return imgs
        B, C, H, W = imgs.shape

        # place learnable background vector at all non-tissue pixels
        encoded_bg = (1-tissue_mask) * self.background_embed.expand(B, -1, H, W)

        # Zero pixels in background, then add the encoded vectors
        imgs = (imgs * tissue_mask) + encoded_bg

        return imgs

    def forward_encoder(self, x, mask_ratio):
        """
        Do forward pass of the encoder
        x: [N, L, D = ph*pw*pc], sequence
        """
        # embed patches
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x).flatten(-2, -1).permute(0, 2, 1)  # B 768 14 14

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

    def predict(self, x, tissue_mask, finalnorm=True):
        """
        Do forward pass of the encoder, without any of the pretraining masking
        x: [N, L, D = ph*pw*pc], sequence
        """

        B, C, H, W = x.shape
        c, h, w = self.token_grid_dims

        x = self.encode_background(x, tissue_mask)

        # embed patches
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # B 768 14 14
        xtrans = x5.flatten(-2,-1).permute(0,2,1)

        #x = self.patchify(x)
        #x = self.patch_embed(x)
        c, h, w = self.token_grid_dims

        # add pos embed w/o cls token
        if self.use_spatial_pos_encoding:
            # spatial pos encoding size = [1, h*w, D]
            xtrans = xtrans + self.encoder_spatial_pos_embed.unsqueeze(1).repeat(1,c,1,1).flatten(start_dim=1,end_dim=2)
        if self.use_spectral_pos_encoding:
            # spectral pos encoding size = [1, c, D]
            xtrans = xtrans + self.encoder_spectral_pos_embed.unsqueeze(2).repeat(1,1,h*w,1).flatten(start_dim=1,end_dim=2) # todo if this doesn't work, maybe do each h and w dim separaesly

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(xtrans.shape[0], -1, -1)
        xtrans = torch.cat((cls_tokens, xtrans), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            xtrans = blk(xtrans)
        xtrans = self.norm(xtrans)

        # cut off class token
        xtrans = xtrans[:, 1:]

        # upwards path
        xtrans = xtrans.permute(0,2,1).reshape(B,self.embed_dim,h,w)

        x = self.up1(xtrans, x4) #
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

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

    def forward(self, imgs, tissue_mask, mask_ratio=0.75):
        imgs = self.encode_background(imgs, tissue_mask) # todo try moving this inside the next line, so encoded bg is not predicted
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B, L, D]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class MaskedAutoencoderViT_ConvStack(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_dims, patch_dims, pos_encoding_params,learn_bg_encoding=False,
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
        self.learn_bg_encoding = learn_bg_encoding

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

        self.background_embed = nn.Parameter(
            torch.zeros(1, self.img_dims[0], 1, 1), requires_grad=self.learn_bg_encoding)

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


        # --- Conv patchify layer
        # this is currently locked to assume the patches are spatially 16x16

        self.patchify_conv1 = nn.Conv2d(img_dims[0], 256, kernel_size=4, stride=4, padding=0)
        self.patchify_conv2 = nn.Conv2d(256, self.embed_dim, kernel_size=4, stride=4, padding=0)

        self.patchify_conv_stack = nn.Sequential(
            self.patchify_conv1,
            nn.ReLU(inplace=True),
            self.patchify_conv2,
            nn.ReLU(inplace=True),
        )

        self.unpatchify_conv1 = nn.ConvTranspose2d(self.decoder_embed_dim, 256,4, 4)
        self.unpatchify_conv2 = nn.ConvTranspose2d(256, img_dims[0],4, 4)

        self.unpatchify_conv_stack = nn.Sequential(
            self.unpatchify_conv1,
            nn.ReLU(inplace=True),
            self.unpatchify_conv2,
            #nn.ReLU(inplace=True),
        )

        # --------------------------------------------------------------------------

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

        if self.learn_bg_encoding:
            torch.nn.init.normal_(self.background_embed, std=.02)

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

    def reset_pos_encodings(self, new_image_size, device): # todo this should be tidied if here to stay
        pc, ph, pw = self.patch_dims
        # Get 2d sincos embedding for the new grid size implied by new image size
        new_pos_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size=new_image_size // ph,
                                                   cls_token=False)
        new_decoder_pos_encoding = get_2d_sincos_pos_embed(self.decoder_embed_dim,
                                                           grid_size=new_image_size // ph, cls_token=False)

        # Set position embed to this new encoding
        self.encoder_spatial_pos_embed.data = torch.from_numpy(new_pos_encoding).float().unsqueeze(0).to(device)
        self.decoder_spatial_pos_embed.data = torch.from_numpy(new_decoder_pos_encoding).float().unsqueeze(0).to(device)

        # Reset model expected image size
        self.token_grid_dims = [self.token_grid_dims[0], new_image_size // ph, new_image_size // pw]
        self.img_dims = [self.img_dims[0], new_image_size, new_image_size]

        self.num_spatial_patches = np.prod(self.token_grid_dims[1:])  # number of tokens along the spatial dimensions
        self.num_spectral_patches = self.token_grid_dims[0]  # number of tokens along the spectral dimension
        self.num_patches = np.prod(self.token_grid_dims)  # total number of tokens in the spatial-spectral input

        self.patch_embed.strict_img_size = False

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

    def encode_background(self, imgs, tissue_mask):
        """
        imgs: [B, C, H, W]
        tissue_mask: [B, 1, H, W]
        """
        if not self.learn_bg_encoding: return imgs
        B, C, H, W = imgs.shape

        # place learnable background vector at all non-tissue pixels
        encoded_bg = (1-tissue_mask) * self.background_embed.expand(B, -1, H, W)

        # Zero pixels in background, then add the encoded vectors
        imgs = (imgs * tissue_mask) + encoded_bg

        return imgs

    def forward_encoder(self, x, mask_ratio):
        """
        Do forward pass of the encoder
        x: [N, L, D = ph*pw*pc], sequence
        """
        # embed patches
        # x = self.patchify(x)
        #x = self.patch_embed(x)
        x = self.patchify_conv_stack(x).flatten(-2,-1).permute(0,2,1) # B C H W -> B D h w - > B hw D
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

    def encode(self, x, finalnorm=True):
        """
        Do forward pass of the encoder, without any of the pretraining masking
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

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if finalnorm:
            x = self.norm(x)

        return x

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
        #x = self.decoder_pred(x) # B, L, D -> B, L, pc*ph*pw
        x = self.unpatchify_conv_stack(x.reshape(x.shape[0],h,w,x.shape[-1]).permute(0,3,1,2))
        return self.patchify(x)

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

    def forward(self, imgs, tissue_mask, mask_ratio=0.75):
        imgs = self.encode_background(imgs, tissue_mask) # todo try moving this inside the next line, so encoded bg is not predicted
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B, L, D]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class MaskedAutoencoderViT_BetterPos(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_dims, patch_dims, pos_encoding_params,learn_bg_encoding=False,
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
        self.learn_bg_encoding = learn_bg_encoding

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = nn.Linear(in_features=np.prod(self.patch_dims), out_features=self.embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.encoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches, embed_dim // 2), requires_grad=self.learnable_spatial_pos_encoding)
        self.encoder_spectral_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spectral_patches, embed_dim // 2), requires_grad=self.learnable_spectral_pos_encoding)

        self.background_embed = nn.Parameter(
            torch.zeros(1, self.img_dims[0], 1, 1), requires_grad=self.learn_bg_encoding)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spatial_patches, decoder_embed_dim // 2),requires_grad=self.learnable_spatial_pos_encoding)
        self.decoder_spectral_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_spectral_patches, decoder_embed_dim // 2),requires_grad=self.learnable_spectral_pos_encoding)

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
            encoder_spatial_pos_embed = get_2d_sincos_pos_embed(self.embed_dim // 2 , int(self.num_spatial_patches**0.5),cls_token=False)
            decoder_spatial_pos_embed = get_2d_sincos_pos_embed(self.decoder_embed_dim // 2, int(self.num_spatial_patches ** 0.5),cls_token=False)
            self.encoder_spatial_pos_embed.data.copy_(torch.from_numpy(encoder_spatial_pos_embed).float().unsqueeze(0))
            self.decoder_spatial_pos_embed.data.copy_(torch.from_numpy(decoder_spatial_pos_embed).float().unsqueeze(0))

        if self.learnable_spectral_pos_encoding:
            torch.nn.init.normal_(self.encoder_spectral_pos_embed, std=.02)
            torch.nn.init.normal_(self.decoder_spectral_pos_embed, std=.02)
        else:
            encoder_spectral_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.embed_dim // 2, pos=np.arange(0, self.num_spectral_patches))
            decoder_spectral_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=self.decoder_embed_dim // 2, pos=np.arange(0, self.num_spectral_patches))
            self.encoder_spectral_pos_embed.data.copy_(torch.from_numpy(encoder_spectral_pos_embed).float().unsqueeze(0))
            self.decoder_spectral_pos_embed.data.copy_(torch.from_numpy(decoder_spectral_pos_embed).float().unsqueeze(0))

        if self.learn_bg_encoding:
            torch.nn.init.normal_(self.background_embed, std=.02)

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

    def reset_pos_encodings(self, new_image_size, device): # todo this should be tidied if here to stay
        pc, ph, pw = self.patch_dims
        # Get 2d sincos embedding for the new grid size implied by new image size
        new_pos_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size=new_image_size // ph,
                                                   cls_token=False)
        new_decoder_pos_encoding = get_2d_sincos_pos_embed(self.decoder_embed_dim,
                                                           grid_size=new_image_size // ph, cls_token=False)

        # Set position embed to this new encoding
        self.encoder_spatial_pos_embed.data = torch.from_numpy(new_pos_encoding).float().unsqueeze(0).to(device)
        self.decoder_spatial_pos_embed.data = torch.from_numpy(new_decoder_pos_encoding).float().unsqueeze(0).to(device)

        # Reset model expected image size
        self.token_grid_dims = [self.token_grid_dims[0], new_image_size // ph, new_image_size // pw]
        self.img_dims = [self.img_dims[0], new_image_size, new_image_size]

        self.num_spatial_patches = np.prod(self.token_grid_dims[1:])  # number of tokens along the spatial dimensions
        self.num_spectral_patches = self.token_grid_dims[0]  # number of tokens along the spectral dimension
        self.num_patches = np.prod(self.token_grid_dims)  # total number of tokens in the spatial-spectral input

        self.patch_embed.strict_img_size = False

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

    def encode_background(self, imgs, tissue_mask):
        """
        imgs: [B, C, H, W]
        tissue_mask: [B, 1, H, W]
        """
        if not self.learn_bg_encoding: return imgs
        B, C, H, W = imgs.shape

        # place learnable background vector at all non-tissue pixels
        encoded_bg = (1-tissue_mask) * self.background_embed.expand(B, -1, H, W)

        # Zero pixels in background, then add the encoded vectors
        imgs = (imgs * tissue_mask) + encoded_bg

        return imgs

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
            catencoding = torch.cat([
                self.encoder_spatial_pos_embed.unsqueeze(1).repeat(1, c, 1, 1).flatten(start_dim=1,end_dim=2),
                self.encoder_spectral_pos_embed.unsqueeze(2).repeat(1, 1, h * w, 1).flatten(start_dim=1, end_dim=2)],
                dim=-1)
            x = x + catencoding

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

    def encode(self, x, finalnorm=True):
        """
        Do forward pass of the encoder, without any of the pretraining masking
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

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if finalnorm:
            x = self.norm(x)

        return x

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
            catencoding = torch.cat([
                self.decoder_spatial_pos_embed.unsqueeze(1).repeat(1, c, 1, 1).flatten(start_dim=1, end_dim=2),
                self.decoder_spectral_pos_embed.unsqueeze(2).repeat(1, 1, h * w, 1).flatten(start_dim=1, end_dim=2)],
                dim=-1)
            x[:,1:] = x[:,1:] + catencoding

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

    def forward(self, imgs, tissue_mask, mask_ratio=0.75):
        imgs = self.encode_background(imgs, tissue_mask) # todo try moving this inside the next line, so encoded bg is not predicted
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [B, L, D]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

class MaskedAutoencoderViT_Cosine(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_dims, patch_dims, pos_encoding_params,learn_bg_encoding=False,
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
        self.learn_bg_encoding = learn_bg_encoding

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

        self.background_embed = nn.Parameter(
            torch.zeros(1, self.img_dims[0], 1, 1), requires_grad=self.learn_bg_encoding)

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

        if self.learn_bg_encoding:
            torch.nn.init.normal_(self.background_embed, std=.02)

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

    def reset_pos_encodings(self, new_image_size, device): # todo this should be tidied if here to stay
        pc, ph, pw = self.patch_dims
        # Get 2d sincos embedding for the new grid size implied by new image size
        new_pos_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size=new_image_size // ph,
                                                   cls_token=False)
        new_decoder_pos_encoding = get_2d_sincos_pos_embed(self.decoder_embed_dim,
                                                           grid_size=new_image_size // ph, cls_token=False)

        # Set position embed to this new encoding
        self.encoder_spatial_pos_embed.data = torch.from_numpy(new_pos_encoding).float().unsqueeze(0).to(device)
        self.decoder_spatial_pos_embed.data = torch.from_numpy(new_decoder_pos_encoding).float().unsqueeze(0).to(device)

        # Reset model expected image size
        self.token_grid_dims = [self.token_grid_dims[0], new_image_size // ph, new_image_size // pw]
        self.img_dims = [self.img_dims[0], new_image_size, new_image_size]

        self.num_spatial_patches = np.prod(self.token_grid_dims[1:])  # number of tokens along the spatial dimensions
        self.num_spectral_patches = self.token_grid_dims[0]  # number of tokens along the spectral dimension
        self.num_patches = np.prod(self.token_grid_dims)  # total number of tokens in the spatial-spectral input

        self.patch_embed.strict_img_size = False

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

    def encode_background(self, imgs, tissue_mask):
        """
        imgs: [B, C, H, W]
        tissue_mask: [B, 1, H, W]
        """
        if not self.learn_bg_encoding: return imgs
        B, C, H, W = imgs.shape

        # place learnable background vector at all non-tissue pixels
        encoded_bg = (1-tissue_mask) * self.background_embed.expand(B, -1, H, W)

        # Zero pixels in background, then add the encoded vectors
        imgs = (imgs * tissue_mask) + encoded_bg

        return imgs

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

    def encode(self, x, finalnorm=True):
        """
        Do forward pass of the encoder, without any of the pretraining masking
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

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if finalnorm:
            x = self.norm(x)

        return x

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


        B, L, D = pred.shape
        loss = torch.nn.functional.cosine_embedding_loss(
            pred.reshape(B*L,D),
            target.reshape(B*L,D),
            torch.ones(B*L).to(pred.device),
            reduction='none',
        ).reshape(B,L)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, tissue_mask, mask_ratio=0.75):
        imgs = self.encode_background(imgs, tissue_mask) # todo try moving this inside the next line, so encoded bg is not predicted
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

