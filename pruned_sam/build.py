import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TinySAM'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TinySAM', 'tinysam'))

import quantization_layer
from tinysam.modeling import TinyViT, MaskDecoder, PromptEncoder, TwoWayTransformer, Sam

PRUNED_CONFIGS = {
    'pruned_l': {
        'description': 'Light pruning (~30% reduction)',
        'encoder': {
            'embed_dims': [48, 96, 128, 256],
            'depths': [2, 2, 4, 2],
            'num_heads': [2, 4, 4, 8],
            'window_sizes': [7, 7, 14, 7],
        },
        'decoder': {
            'depth': 2,
            'embedding_dim': 256,
            'mlp_dim': 2048,
            'num_heads': 8,
        },
    },
    'pruned_m': {
        'description': 'Medium pruning (~45% reduction)',
        'encoder': {
            'embed_dims': [32, 64, 128, 256],
            'depths': [2, 2, 4, 2],
            'num_heads': [2, 4, 4, 8],
            'window_sizes': [7, 7, 14, 7],
        },
        'decoder': {
            'depth': 2,
            'embedding_dim': 256,
            'mlp_dim': 2048,
            'num_heads': 8,
        },
    },
    'pruned_s': {
        'description': 'Aggressive pruning (~60% reduction)',
        'encoder': {
            'embed_dims': [32, 64, 128, 192],
            'depths': [1, 2, 4, 2],
            'num_heads': [2, 4, 4, 6],
            'window_sizes': [7, 7, 14, 7],
        },
        'decoder': {
            'depth': 1,
            'embedding_dim': 128,
            'mlp_dim': 1024,
            'num_heads': 4,
        },
    },
}

ORIGINAL_CONFIG = {
    'embed_dims': [64, 128, 160, 320],
    'depths': [2, 2, 6, 2],
    'num_heads': [2, 4, 5, 10],
    'window_sizes': [7, 7, 14, 7],
}

ORIGINAL_DECODER_CONFIG = {
    'depth': 2,
    'embedding_dim': 256,
    'mlp_dim': 2048,
    'num_heads': 8,
}

def _prune_conv_weight(w, old_in, new_in, old_out, new_out):
    if w.dim() == 0:
        return w
    if w.dim() == 4:
        return w[:new_out, :new_in, :, :].contiguous()
    elif w.dim() == 1:
        return w[:new_out].contiguous()
    return w

def _prune_linear_weight(w, old_in, new_in, old_out, new_out):
    if w.dim() == 0:
        return w
    if w.dim() == 2:
        return w[:new_out, :new_in].contiguous()
    elif w.dim() == 1:
        return w[:new_out].contiguous()
    return w

def _prune_norm_weight(w, old_c, new_c):
    if w.dim() == 0:
        return w
    return w[:new_c].contiguous()

def _prune_attention_qkv(qkv_weight, old_dim, new_dim, old_heads, new_heads, key_dim=32, attn_ratio=1):
    old_h_total = old_heads * (2 * key_dim + int(attn_ratio * key_dim))
    new_h_total = new_heads * (2 * key_dim + int(attn_ratio * key_dim))
    return qkv_weight[:new_dim, :new_h_total].contiguous()

def _prune_attention_proj(proj_weight, old_dim, new_dim, old_heads, new_heads, attn_ratio=1):
    old_dh = old_heads * int(attn_ratio * (old_dim // old_heads))
    new_dh = new_heads * int(attn_ratio * (new_dim // new_heads))
    return proj_weight[:new_dh, :new_dim].contiguous()

def _prune_attention_bias(bias, old_heads, new_heads):
    return bias[:new_heads, :].contiguous()

def _prune_encoder_state_dict(orig_sd, old_cfg, new_cfg):
    pruned = {}
    old_dims = old_cfg['embed_dims']
    new_dims = new_cfg['embed_dims']
    old_depths = old_cfg['depths']
    new_depths = new_cfg['depths']
    old_heads = old_cfg['num_heads']
    new_heads = new_cfg['num_heads']

    def prune_conv2d_bn(key, tensor, in_ch, out_ch):
        if key.endswith('c.weight'):
            return _prune_conv_weight(tensor, in_ch, in_ch, out_ch, out_ch)
        if any(key.endswith(x) for x in ['bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var']):
            return _prune_norm_weight(tensor, out_ch, out_ch)
        if key.endswith('bn.num_batches_tracked'):
            return tensor
        return tensor

    def prune_patch_embed(key, tensor, old_dim0, new_dim0):
        parts = key.split('.')
        seq_idx = int(parts[2])
        sub = parts[3]
        sub2 = parts[4] if len(parts) > 4 else ''
        if seq_idx == 0:
            old_out = 32
            new_out = new_dim0 // 2 if new_dim0 < 64 else 32
            if sub == 'c' and sub2 == 'weight':
                return _prune_conv_weight(tensor, 3, 3, old_out, new_out)
            if sub == 'bn':
                return _prune_norm_weight(tensor, old_out, new_out)
            return tensor
        elif seq_idx == 2:
            old_in = 32
            new_in = new_dim0 // 2 if new_dim0 < 64 else 32
            if sub == 'c' and sub2 == 'weight':
                return _prune_conv_weight(tensor, old_in, new_in, old_dim0, new_dim0)
            if sub == 'bn':
                return _prune_norm_weight(tensor, old_dim0, new_dim0)
            return tensor
        return tensor

    def prune_mbconv_block(key, tensor, old_dim, new_dim):
        parts = key.split('.')
        block_type = parts[4]
        param = '.'.join(parts[5:])
        old_hidden = int(old_dim * 4.0)
        new_hidden = int(new_dim * 4.0)
        if block_type == 'conv1':
            if param.endswith('c.weight'):
                return _prune_conv_weight(tensor, old_dim, new_dim, old_hidden, new_hidden)
            return prune_conv2d_bn(key, tensor, old_hidden, new_hidden)
        elif block_type == 'conv2':
            if param.endswith('c.weight'):
                return _prune_conv_weight(tensor, old_hidden, new_hidden, old_hidden, new_hidden)
            return prune_conv2d_bn(key, tensor, old_hidden, new_hidden)
        elif block_type == 'conv3':
            if param.endswith('c.weight'):
                return _prune_conv_weight(tensor, old_hidden, new_hidden, old_dim, new_dim)
            return prune_conv2d_bn(key, tensor, old_dim, new_dim)
        return tensor

    def prune_patch_merging(key, tensor, old_in_dim, new_in_dim, old_out_dim, new_out_dim):
        parts = key.split('.')
        block_type = parts[3]
        sub = parts[4]
        sub2 = parts[5] if len(parts) > 5 else ''
        if block_type == 'conv2':
            if sub == 'c' and sub2 == 'weight':
                return _prune_conv_weight(tensor, old_out_dim, new_out_dim, old_out_dim, new_out_dim)
            if sub == 'bn':
                return _prune_norm_weight(tensor, old_out_dim, new_out_dim)
            return tensor
        elif block_type == 'conv1':
            if sub == 'c' and sub2 == 'weight':
                return _prune_conv_weight(tensor, old_in_dim, new_in_dim, old_out_dim, new_out_dim)
            if sub == 'bn':
                return _prune_norm_weight(tensor, old_out_dim, new_out_dim)
            return tensor
        elif block_type == 'conv3':
            if sub == 'c' and sub2 == 'weight':
                return _prune_conv_weight(tensor, old_out_dim, new_out_dim, old_out_dim, new_out_dim)
            if sub == 'bn':
                return _prune_norm_weight(tensor, old_out_dim, new_out_dim)
            return tensor
        return tensor

    def prune_attention(key, tensor, old_dim, new_dim, old_h, new_h):
        head_dim = old_dim // old_h
        new_head_dim = new_dim // new_h
        if 'qkv.weight' in key:
            old_h_total = old_h * (2 * head_dim + head_dim)
            new_h_total = new_h * (2 * new_head_dim + new_head_dim)
            return tensor[:new_h_total, :new_dim].contiguous()
        if 'qkv.bias' in key:
            old_h_total = old_h * (2 * head_dim + head_dim)
            new_h_total = new_h * (2 * new_head_dim + new_head_dim)
            return tensor[:new_h_total].contiguous()
        if 'proj.weight' in key:
            old_dh = old_h * head_dim
            new_dh = new_h * new_head_dim
            return tensor[:new_dim, :new_dh].contiguous()
        if 'proj.bias' in key:
            old_dh = old_h * head_dim
            new_dh = new_h * new_head_dim
            return tensor[:new_dim].contiguous()
        if 'attention_biases' in key:
            return tensor[:new_h, :].contiguous()
        if 'attention_bias_idxs' in key:
            return tensor
        if 'norm' in key and ('weight' in key or 'bias' in key):
            return _prune_norm_weight(tensor, old_dim, new_dim)
        return tensor

    def prune_mlp(key, tensor, old_dim, new_dim):
        old_mlp_hidden = int(old_dim * 4.0)
        new_mlp_hidden = int(new_dim * 4.0)
        if 'fc1' in key:
            if 'weight' in key:
                return _prune_linear_weight(tensor, old_dim, new_dim, old_mlp_hidden, new_mlp_hidden)
            return tensor[:new_mlp_hidden].contiguous()
        if 'fc2' in key:
            if 'weight' in key:
                return _prune_linear_weight(tensor, old_mlp_hidden, new_mlp_hidden, old_dim, new_dim)
            return tensor[:new_dim].contiguous()
        if 'norm' in key and ('weight' in key or 'bias' in key):
            return _prune_norm_weight(tensor, old_dim, new_dim)
        return tensor

    def prune_local_conv(key, tensor, old_dim, new_dim):
        if 'c.weight' in key:
            return _prune_conv_weight(tensor, old_dim, new_dim, old_dim, new_dim)
        if 'bn' in key and ('weight' in key or 'bias' in key or 'running' in key):
            return _prune_norm_weight(tensor, old_dim, new_dim)
        return tensor

    for key, tensor in orig_sd.items():
        if not key.startswith('image_encoder.'):
            continue
        local_key = key[len('image_encoder.'):]

        if local_key.startswith('patch_embed.'):
            t = prune_patch_embed(local_key, tensor, old_dims[0], new_dims[0])
            pruned[key] = t
            continue

        if local_key.startswith('neck.'):
            parts = local_key.split('.')
            old_last = old_dims[-1]
            new_last = new_dims[-1]
            if parts[1] == '0' and 'weight' in local_key:
                t = _prune_conv_weight(tensor, old_last, new_last, 256, 256)
            elif parts[1] == '2' and 'weight' in local_key:
                t = tensor
            else:
                t = tensor
            pruned[key] = t
            continue

        if local_key.startswith('layers.'):
            parts = local_key.split('.')
            layer_idx = int(parts[1])
            old_dim = old_dims[layer_idx]
            new_dim = new_dims[layer_idx]
            old_depth = old_depths[layer_idx]
            new_depth = new_depths[layer_idx]

            if 'downsample' in local_key:
                old_out = old_dims[layer_idx + 1] if layer_idx + 1 < len(old_dims) else old_dim
                new_out = new_dims[layer_idx + 1] if layer_idx + 1 < len(new_dims) else new_dim
                t = prune_patch_merging(local_key, tensor, old_dim, new_dim, old_out, new_out)
                pruned[key] = t
                continue

            if 'blocks' in local_key:
                block_idx = int(parts[3])
                if block_idx >= new_depth:
                    continue

                if layer_idx == 0:
                    t = prune_mbconv_block(local_key, tensor, old_dim, new_dim)
                    pruned[key] = t
                    continue
                else:
                    sub = '.'.join(parts[4:])
                    if 'attn' in sub:
                        old_h = old_heads[layer_idx]
                        new_h = new_heads[layer_idx]
                        t = prune_attention(local_key, tensor, old_dim, new_dim, old_h, new_h)
                    elif 'mlp' in sub:
                        t = prune_mlp(local_key, tensor, old_dim, new_dim)
                    elif 'local_conv' in sub:
                        t = prune_local_conv(local_key, tensor, old_dim, new_dim)
                    else:
                        t = tensor
                    pruned[key] = t
                    continue

        if local_key.startswith('norm_head.'):
            if 'weight' in local_key or 'bias' in local_key:
                t = _prune_norm_weight(tensor, old_dims[-1], new_dims[-1])
                pruned[key] = t
                continue

        if local_key.startswith('head.'):
            if 'weight' in local_key:
                t = _prune_linear_weight(tensor, old_dims[-1], new_dims[-1], 1000, 1000)
                pruned[key] = t
                continue

        pruned[key] = tensor

    return pruned


def _prune_decoder_state_dict(orig_sd, old_cfg, new_cfg):
    pruned = {}
    old_ed = old_cfg['embedding_dim']
    new_ed = new_cfg['embedding_dim']
    old_depth = old_cfg['depth']
    new_depth = new_cfg['depth']
    old_heads = old_cfg['num_heads']
    new_heads = new_cfg['num_heads']
    old_mlp = old_cfg['mlp_dim']
    new_mlp = new_cfg['mlp_dim']

    for key, tensor in orig_sd.items():
        if not key.startswith('mask_decoder.'):
            continue
        local_key = key[len('mask_decoder.'):]

        if local_key.startswith('transformer.'):
            rest = local_key[len('transformer.'):]
            if rest.startswith('layers.'):
                parts = rest.split('.')
                layer_idx = int(parts[1])
                if layer_idx >= new_depth:
                    continue
                param = '.'.join(parts[2:])

                if 'self_attn' in param or 'cross_attn_token_to_image' in param or 'cross_attn_image_to_token' in param:
                    if 'qkv.weight' in param:
                        head_dim = old_ed // old_heads
                        new_head_dim = new_ed // new_heads
                        old_h = old_heads * (2 * head_dim + head_dim)
                        new_h = new_heads * (2 * new_head_dim + new_head_dim)
                        t = tensor[:new_ed, :new_h].clone() if tensor.dim() == 2 else tensor
                    elif 'qkv.bias' in param:
                        old_h = old_heads * (2 * (old_ed // old_heads) + (old_ed // old_heads))
                        new_h = new_heads * (2 * (new_ed // new_heads) + (new_ed // new_heads))
                        t = tensor[:new_h].clone()
                    elif 'proj.weight' in param:
                        old_dh = old_heads * (old_ed // old_heads)
                        new_dh = new_heads * (new_ed // new_heads)
                        t = tensor[:new_dh, :new_ed].clone()
                    elif 'proj.bias' in param:
                        old_dh = old_heads * (old_ed // old_heads)
                        new_dh = new_heads * (new_ed // new_heads)
                        t = tensor[:new_dh].clone()
                    elif 'rel_pos_h' in param or 'rel_pos_w' in param:
                        t = tensor
                    elif 'norm' in param and ('weight' in param or 'bias' in param):
                        t = _prune_norm_weight(tensor, old_ed, new_ed)
                    else:
                        t = tensor
                elif 'mlp' in param:
                    if 'lin1.weight' in param:
                        t = _prune_linear_weight(tensor, old_ed, new_ed, old_mlp, new_mlp)
                    elif 'lin1.bias' in param:
                        t = tensor[:new_mlp].clone()
                    elif 'lin2.weight' in param:
                        t = _prune_linear_weight(tensor, old_mlp, new_mlp, old_ed, new_ed)
                    elif 'lin2.bias' in param:
                        t = tensor[:new_ed].clone()
                    elif 'norm' in param and ('weight' in param or 'bias' in param):
                        t = _prune_norm_weight(tensor, old_ed, new_ed)
                    else:
                        t = tensor
                elif 'norm' in param and ('weight' in param or 'bias' in param):
                    t = _prune_norm_weight(tensor, old_ed, new_ed)
                else:
                    t = tensor
            elif 'final_attn_token_to_image' in rest:
                if 'qkv.weight' in rest:
                    old_h = old_heads * (2 * (old_ed // old_heads) + (old_ed // old_heads))
                    new_h = new_heads * (2 * (new_ed // new_heads) + (new_ed // new_heads))
                    t = tensor[:new_ed, :new_h].clone()
                elif 'qkv.bias' in rest:
                    old_h = old_heads * (2 * (old_ed // old_heads) + (old_ed // old_heads))
                    new_h = new_heads * (2 * (new_ed // new_heads) + (new_ed // new_heads))
                    t = tensor[:new_h].clone()
                elif 'proj.weight' in rest:
                    old_dh = old_heads * (old_ed // old_heads)
                    new_dh = new_heads * (new_ed // new_heads)
                    t = tensor[:new_dh, :new_ed].clone()
                elif 'proj.bias' in rest:
                    old_dh = old_heads * (old_ed // old_heads)
                    new_dh = new_heads * (new_ed // new_heads)
                    t = tensor[:new_dh].clone()
                elif 'norm' in rest and ('weight' in rest or 'bias' in rest):
                    t = _prune_norm_weight(tensor, old_ed, new_ed)
                else:
                    t = tensor
            elif 'norm' in rest and ('weight' in rest or 'bias' in rest):
                t = _prune_norm_weight(tensor, old_ed, new_ed)
            else:
                t = tensor
            pruned[key] = t
            continue

        if 'iou_token' in local_key or 'mask_tokens' in local_key:
            if 'weight' in local_key:
                t = tensor[:, :new_ed].clone()
            else:
                t = tensor
            pruned[key] = t
            continue

        if 'output_hypernetworks_mlps' in local_key:
            parts = local_key.split('.')
            mlp_idx = int(parts[1])
            param_name = '.'.join(parts[2:])
            if 'layers.0.weight' in param_name:
                t = _prune_linear_weight(tensor, old_ed, new_ed, old_ed, new_ed)
            elif 'layers.0.bias' in param_name:
                t = tensor[:new_ed].clone()
            elif 'layers.1.weight' in param_name:
                t = _prune_linear_weight(tensor, old_ed, new_ed, old_ed // 8, old_ed // 8)
            elif 'layers.1.bias' in param_name:
                t = tensor[:old_ed // 8].clone()
            elif 'layers.2' in param_name:
                pass
            else:
                t = tensor
            pruned[key] = t
            continue

        if 'iou_prediction_head' in local_key:
            parts = local_key.split('.')
            param_name = '.'.join(parts[1:])
            if 'layers.0' in param_name:
                if 'weight' in param_name:
                    t = _prune_linear_weight(tensor, old_ed, new_ed, 256, 256)
                elif 'bias' in param_name:
                    t = tensor[:256].clone()
                else:
                    t = tensor
            else:
                t = tensor
            pruned[key] = t
            continue

        if 'output_upscaling' in local_key:
            parts = local_key.split('.')
            param_name = '.'.join(parts[1:])
            if '0' in parts and len(parts) > 1 and parts[1] == '0':
                if 'weight' in param_name:
                    t = _prune_conv_weight(tensor, old_ed, new_ed, new_ed // 4, old_ed // 4)
                elif 'bias' in param_name:
                    t = tensor[:old_ed // 4].clone()
                else:
                    t = tensor
            elif '1' in parts and len(parts) > 1 and parts[1] == '1':
                if 'weight' in param_name or 'bias' in param_name:
                    t = _prune_norm_weight(tensor, old_ed // 4, old_ed // 4)
                else:
                    t = tensor
            elif '3' in parts and len(parts) > 1 and parts[1] == '3':
                if 'weight' in param_name:
                    t = _prune_conv_weight(tensor, old_ed // 4, old_ed // 4, old_ed // 8, old_ed // 8)
                elif 'bias' in param_name:
                    t = tensor[:old_ed // 8].clone()
                else:
                    t = tensor
            else:
                t = tensor
            pruned[key] = t
            continue

        pruned[key] = tensor

    return pruned


def _prune_prompt_encoder_state_dict(orig_sd, old_ed, new_ed):
    pruned = {}
    for key, tensor in orig_sd.items():
        if not key.startswith('prompt_encoder.'):
            continue
        local_key = key[len('prompt_encoder.'):]

        if 'point_embeddings' in local_key and 'weight' in local_key:
            t = tensor[:, :new_ed].clone()
        elif 'not_a_point_embed' in local_key and 'weight' in local_key:
            t = tensor[:new_ed].clone()
        elif 'mask_downscaling' in local_key:
            if '0.' in local_key:
                if 'weight' in local_key and tensor.dim() == 4:
                    t = _prune_conv_weight(tensor, 1, 1, 4, 4)
                elif 'bias' in local_key:
                    t = tensor[:4].clone()
                else:
                    t = tensor
            elif '1.' in local_key:
                if 'weight' in local_key:
                    t = _prune_conv_weight(tensor, 4, 4, 16, 16)
                elif 'bias' in local_key:
                    t = tensor[:16].clone()
                elif 'weight' in local_key:
                    t = _prune_norm_weight(tensor, 4, 4)
                elif 'bias' in local_key and tensor.dim() == 1:
                    t = _prune_norm_weight(tensor, 4, 4)
                else:
                    t = tensor
            elif '2.' in local_key:
                if 'weight' in local_key:
                    t = _prune_conv_weight(tensor, 16, 16, old_ed, new_ed)
                elif 'bias' in local_key:
                    t = tensor[:new_ed].clone()
                else:
                    t = tensor
            else:
                t = tensor
        elif 'no_mask_embed' in local_key and 'weight' in local_key:
            t = tensor[:new_ed].clone()
        elif 'pe_layer' in local_key:
            t = tensor
        else:
            t = tensor
        pruned[key] = t
    return pruned


def build_pruned_sam(config_name='pruned_m', checkpoint=None, original_checkpoint=None):
    if config_name not in PRUNED_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(PRUNED_CONFIGS.keys())}")

    cfg = PRUNED_CONFIGS[config_name]
    enc_cfg = cfg['encoder']
    dec_cfg = cfg['decoder']
    new_ed = dec_cfg['embedding_dim']

    prompt_embed_dim = new_ed
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    model = Sam(
        image_encoder=TinyViT(
            img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=enc_cfg['embed_dims'],
            depths=enc_cfg['depths'],
            num_heads=enc_cfg['num_heads'],
            window_sizes=enc_cfg['window_sizes'],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=dec_cfg['depth'],
                embedding_dim=dec_cfg['embedding_dim'],
                mlp_dim=dec_cfg['mlp_dim'],
                num_heads=dec_cfg['num_heads'],
            ),
            transformer_dim=dec_cfg['embedding_dim'],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    model.eval()

    enc = model.image_encoder
    res = list(enc.patches_resolution)
    for i in range(len(enc.layers)):
        layer = enc.layers[i]
        for block in layer.blocks:
            if hasattr(block, 'input_resolution'):
                block.input_resolution = tuple(res)
        if hasattr(layer, 'downsample') and layer.downsample is not None:
            conv2 = layer.downsample.conv2
            if hasattr(conv2, 'c') and conv2.c.stride[0] == 2:
                res[0] //= 2
                res[1] //= 2

    spatial_h, spatial_w = res
    orig_forward = enc.forward

    def patched_forward(x):
        x = enc.patch_embed(x)
        x = enc.layers[0](x)
        for i in range(1, len(enc.layers)):
            x = enc.layers[i](x)
        B, _, C = x.size()
        x = x.view(B, spatial_h, spatial_w, C)
        x = x.permute(0, 3, 1, 2)
        x = enc.neck(x)
        if spatial_h != 64 or spatial_w != 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x

    enc.forward = patched_forward

    if original_checkpoint is not None:
        print(f'Loading original weights from: {original_checkpoint}')
        with open(original_checkpoint, 'rb') as f:
            if original_checkpoint.endswith('tinysam_w8a8.pth'):
                orig_model = torch.load(f, map_location='cpu', weights_only=False)
                orig_sd = orig_model.state_dict()
            else:
                orig_sd = torch.load(f, map_location='cpu', weights_only=False)

        new_sd = {}
        new_sd.update(_prune_encoder_state_dict(orig_sd, ORIGINAL_CONFIG, enc_cfg))

        if dec_cfg != ORIGINAL_DECODER_CONFIG:
            new_sd.update(_prune_decoder_state_dict(orig_sd, ORIGINAL_DECODER_CONFIG, dec_cfg))
            new_sd.update(_prune_prompt_encoder_state_dict(orig_sd, 256, new_ed))
        else:
            for k, v in orig_sd.items():
                if k.startswith('mask_decoder.') or k.startswith('prompt_encoder.'):
                    new_sd[k] = v

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f'  Missing keys (from depth reduction, expected): {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')

        n_orig = sum(p.numel() for p in orig_sd.values() if hasattr(p, 'numel'))
        n_new = sum(p.numel() for p in model.parameters())
        ratio = (1 - n_new / n_orig) * 100
        print(f'  Original params: {n_orig/1e6:.2f}M → Pruned: {n_new/1e6:.2f}M ({ratio:.1f}% reduction)')

    if checkpoint is not None:
        print(f'Loading pruned checkpoint from: {checkpoint}')
        with open(checkpoint, 'rb') as f:
            sd = torch.load(f, map_location='cpu', weights_only=False)
        model.load_state_dict(sd, strict=False)

    return model


def list_available_configs():
    print('Available pruning configs:')
    for name, cfg in PRUNED_CONFIGS.items():
        enc = cfg['encoder']
        dec = cfg['decoder']
        total = 1
        for d in enc['depths']:
            total += d
        total += 1
        print(f'\n  [{name}] {cfg["description"]}')
        print(f'    Encoder: embed_dims={enc["embed_dims"]}, depths={enc["depths"]}, heads={enc["num_heads"]}')
        print(f'    Decoder: depth={dec["depth"]}, embed_dim={dec["embedding_dim"]}, heads={dec["num_heads"]}')


if __name__ == '__main__':
    list_available_configs()
    print('\n' + '=' * 60)
    print('Building pruned_m config from TinySAM weights...')
    model = build_pruned_sam(
        config_name='pruned_m',
        original_checkpoint='/home/zhang/vista-slam/TinySAM/weights/tinysam_42.3.pth',
    )
    print('\nDone! Pruned model ready for CPU inference.')
