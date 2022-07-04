# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Xu Cao (xc2057@nyu.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import openclip as open_clip
from cross_attention import CrossAttention, CrossBlock

logger = logging.getLogger(__name__)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MVLM(nn.Module):

    def __init__(self, clip_base):
        super(MVLM, self).__init__()

        self.vision_encoder = clip_base.encode_image
        self.language_encoder = clip_base.encode_text






def get_model(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


if __name__ == "__main__":
    import sys
    sys.path.append("/home/xucao/ICLR2023/MVLM-PyTorch/lib/")
    from PIL import Image
    from dataset.coco_caption import CocoCaptions
    import torchvision.transforms as transforms
    from dataset.randaugment import RandomAugment
    import numpy as np
    import matplotlib.pyplot as plt

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        plt.imsave('tmp.png', inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0),
                                        interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = CocoCaptions(root = '/media/data/dataset/coco/images/train2017', annFile='/media/data/dataset/coco/annotations/captions_train2017.json', transforms=train_transform)

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
    
    image, mask, text = dataset.__getitem__(4)
    imshow(image)
    image = image.unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    # target = open_clip.tokenize(["a diagram", "some people", "some food", "a giraffa", "a zebra", "a woman"] + text)
    target = open_clip.tokenize(text)

    trans = Mlp(768, out_features=512)

    with torch.no_grad():
        image_features, visual_features = model.encode_image(image, mask)
        text_features, language_features = model.encode_text(target)

        print(trans(visual_features, 14, 14).shape)

        print(visual_features.shape)
        print(language_features.shape)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]