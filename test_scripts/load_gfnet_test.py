import torch
import torch.nn as nn
from functools import partial
from typing import Union
from gfnet import GFNet, GFNetPyramid

def load_checkpoint(model: Union[GFNet, GFNetPyramid],
                    input_size: int,
                    arch_name: str,
                    ckpt_pth: str):
    checkpoint = torch.load(ckpt_pth, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

            # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]

    if arch_name in ['gfnet-ti', 'gfnet-xs', 'gfnet-s', 'gfnet-b']:
        num_patches = (input_size // 16) ** 2
    elif arch_name in ['gfnet-h-ti', 'gfnet-h-s', 'gfnet-h-b']:
        num_patches = (input_size // 4) ** 2
    else:
        raise NotImplementedError

    num_extra_tokens = 0
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    scale_up_ratio = new_size / orig_size
    # class_token and dist_token are kept unchanged
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model['pos_embed'] = pos_tokens

    for name in checkpoint_model.keys():
        if 'complex_weight' in name:
            h, w, num_heads = checkpoint_model[name].shape[0:3]  # h, w, c, 2
            origin_weight = checkpoint_model[name]
            upsample_h = h * new_size // orig_size
            upsample_w = upsample_h // 2 + 1
            origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
            new_weight = torch.nn.functional.interpolate(
                origin_weight, size=(upsample_h, upsample_w), mode='bicubic', align_corners=True).permute(0, 2, 3,
                                                                                                          1).reshape(
                upsample_h, upsample_w, num_heads, 2)
            checkpoint_model[name] = new_weight
    model.load_state_dict(checkpoint_model, strict=True)


if __name__ == '__main__':
    # test instantiating a model and loading a pretrained checkpoint for it
    INPUT_SIZE = 224
    ARCH_NAME = "gfnet-h-s"
    CKPT_PTH = f"/home/wendi/hdd/data/gfnet/{ARCH_NAME}.pth"

    # test instantiating gfnet-h-ti and loading a pretrained checkpoint for it
    model = GFNetPyramid(
        img_size=INPUT_SIZE,
        patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 10, 3],
        mlp_ratio=[4, 4, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.2, init_values=1e-5
    )

    load_checkpoint(model,
                    input_size=INPUT_SIZE,
                    arch_name=ARCH_NAME,
                    ckpt_pth=CKPT_PTH)
    print()
