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
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,input_norm=True,decoder_pred_intermediate=32,zero_tissue_mask=False):
        super().__init__()

        self.in_chans = in_chans
        self.input_norm = input_norm
        self.bn0 = nn.BatchNorm2d(in_chans,affine=False)
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.zero_tissue_mask = zero_tissue_mask

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_pred_intermediate = decoder_pred_intermediate
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_coarse = nn.Linear(decoder_embed_dim, patch_size ** 2 * self.decoder_pred_intermediate,bias=True)  # decoder to patch
        self.decoder_pred_fine = nn.Linear(self.decoder_pred_intermediate, in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    # Reset the size of the position encodings used in the model
    def reset_pos_encodings(self, new_image_size, device):
        # Get 2d sincos embedding for the new grid size implied by new image size
        new_pos_encoding = get_2d_sincos_pos_embed(self.embed_dim, grid_size=new_image_size // self.patch_size, cls_token=True)
        new_decoder_pos_encoding = get_2d_sincos_pos_embed(self.decoder_embed_dim, grid_size=new_image_size // self.patch_size, cls_token=True)

        # Set position embed to this new encoding
        self.pos_embed.data = torch.from_numpy(new_pos_encoding).float().unsqueeze(0).to(device)
        self.decoder_pos_embed.data = torch.from_numpy(new_decoder_pos_encoding).float().unsqueeze(0).to(device)

        # Reset model expected image size
        self.img_size = new_image_size
        self.patch_embed.strict_img_size = False

    def patchify(self, imgs, chans=None):
        if chans is None: chans = self.in_chans
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * chans))
        return x

    def unpatchify(self, x, chans=None):
        if chans is None: chans = self.in_chans
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], chans, h * p, h * p))
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
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
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

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        N, L, D = x.shape
        x = self.decoder_pred_coarse(x).reshape(
            N, L, self.patch_size, self.patch_size, self.decoder_pred_intermediate
        )  # N x L x D -> N x L x patch_dim x patch_dim x decoder_pred_intermediate
        x = self.decoder_pred_fine(x).reshape(
            N, L, self.patch_size * self.patch_size * self.in_chans
        )  # N x L x patch_dim x patch_dim x decoder_pred_intermediate -> N x L x (patch_dim * patch_dim * wavenumber_channels)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_zeromask(self, imgs, pred, token_mask, tissue_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        token_mask: [N, L], 0 is keep, 1 is remove,
        tissue_mask: [N, 3, H, W]
        """
        chans = imgs.shape[1]
        target = self.patchify(imgs) # [N, L, p*p*3]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # calculate MSE between pred and target
        loss = (pred - target) ** 2 # [N, L, p*p*3]

        # extend token mask to full image size
        token_mask = torch.repeat_interleave(
            torch.repeat_interleave(
                token_mask.reshape(token_mask.shape[0], 1, self.img_size // self.patch_size, self.img_size // self.patch_size), self.patch_size, dim=2
            ), self.patch_size, dim=3
        )

        # pixels we are concerned with are both: tissue (1 in tissue mask) AND removed during masking (1 in token mask)
        combined_mask = tissue_mask * token_mask # [N, 1, H, W]

        # calculate loss per pixel in image space
        loss_per_pixel = self.unpatchify(loss, chans=chans).mean(1,keepdim=True) # [N, 1, H, W]

        # ignore loss contribution from pixels we don't care about, calculate mean loss per pixel
        loss = (loss_per_pixel * combined_mask).sum() / combined_mask.sum() # [N, 1, H, W] -> 1

        return loss

    def forward(self, imgs, mask_ratio=0.75, tissue_mask=None):
        if self.input_norm:
            imgs = self.bn0(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        if self.zero_tissue_mask:
            loss = self.forward_loss_zeromask(imgs, pred, mask, tissue_mask)
        else:
            loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


class MaskedAutoencoderViT2(nn.Module):
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


class MaskedAutoencoderViT_discrete(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, input_norm=True,
                 decoder_pred_intermediate=32,num_quants=256):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                 embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss, input_norm=input_norm,
                 decoder_pred_intermediate=decoder_pred_intermediate)

        self.num_quants = num_quants
        self.decoder_pred_fine = nn.Linear(self.decoder_pred_intermediate, num_quants, bias=True)  # decoder to patch

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        N, L, D = x.shape
        x = self.decoder_pred_coarse(x).reshape(
            N, L, self.patch_size, self.patch_size, self.decoder_pred_intermediate
        )  # N x L x D -> N x L x patch_dim x patch_dim x decoder_pred_intermediate
        x = self.decoder_pred_fine(x).reshape(
            N, L, self.patch_size * self.patch_size * self.num_quants
        )  # N x L x patch_dim x patch_dim x decoder_pred_intermediate -> N x L x (patch_dim * patch_dim * wavenumber_channels)

        return x

    def forward(self, imgs, mask_ratio=0.75):
        if self.input_norm:
            imgs = self.bn0(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # N, L, 256 [N, L, p*p*3]
        return pred, mask

class MaskedAutoencoderViT_Cat(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,input_norm=True,decoder_pred_intermediate=32):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                 embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                 decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
                 mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss, input_norm=input_norm,
                 decoder_pred_intermediate=decoder_pred_intermediate)

        self.linear_proj = nn.Conv2d(in_chans, decoder_pred_intermediate, kernel_size=1)
        self.heads = nn.Sequential(
            nn.Conv2d(decoder_pred_intermediate * 2,
                      decoder_pred_intermediate * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(decoder_pred_intermediate * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_pred_intermediate * 2, decoder_pred_intermediate, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(decoder_pred_intermediate),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_pred_intermediate, in_chans, kernel_size=3, stride=1, padding=1),
        )
        self.learnable_spectral_mask_token = nn.Parameter(
            torch.tensor(np.random.normal(loc=0.0, scale=0.02, size=(1, decoder_pred_intermediate, 1, 1)),
                         dtype=torch.float32)) # todo make torch instead of numpy; remove numpy import

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # predictor projection
        N, L, D = x.shape
        x = self.decoder_pred_coarse(x).reshape(
            N, L, self.patch_size * self.patch_size * self.decoder_pred_intermediate
        )  # N x L x D -> N x L x patch_dim *patch_dim*decoder_pred_intermediate

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        target: [N, L, p*p*3]
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Project input image to pred intermediate channel dim
        img_projected = self.linear_proj(imgs) # [B, 32, H, W]
        img_projected = self.patchify(img_projected,chans=self.decoder_pred_intermediate) # [B, L, p*p*32]

        # Mask out projected input the same way that the input to encoder was. Masked pixels replaced with a mask spectral token
        spectral_mask_tokens = self.learnable_spectral_mask_token.repeat(target.shape[0], 1, self.img_size, self.img_size) # [B, H, W, 32]
        spectral_mask_tokens = self.patchify(spectral_mask_tokens,chans=self.decoder_pred_intermediate) # [B, L, p*p*32]
        img_projected = torch.where(
            mask.unsqueeze(-1).bool(),
            spectral_mask_tokens,# where mask ==1, tissue was hidden from encoder, so we use mask tokens
            img_projected # where mask ==0, tissue is kept for encoder, so we use the projected img
        ) # [B, L, p*p*32

        # Concatenate pred and img-projected, send through heads
        pred = self.heads(
            torch.cat([
                self.unpatchify(img_projected,chans=self.decoder_pred_intermediate),
                self.unpatchify(pred,chans=self.decoder_pred_intermediate),
            ], dim=1)
        ) # [B, 405, H, W]
        pred = self.patchify(pred, chans=self.in_chans) # [N, L, p*p*3]

        # Calculate loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, pred

    def forward(self, imgs, mask_ratio=0.75):
        if self.input_norm:
            imgs = self.bn0(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, pred = self.forward_loss(imgs, pred, mask)
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


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
