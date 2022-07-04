# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Xu Cao
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

import re
import random
import numpy as np
import torch
from PIL import Image
import open_clip
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# import json_tricks as json
from .randaugment import RandomAugment


logger = logging.getLogger(__name__)


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask


class CocoCaptions(CocoDetection):
    """`MS Coco Captions Dataset with Masked Autoencoder.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def __init__(
        self,
        root,
        annFile,
        transform = None,
        target_transform = None,
        transforms = None
    ):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.prompt = ''
        self.generate_mask = RandomMaskingGenerator(14, 0.25)
        self.max_words = 77

    def _load_target(self, id):
        return [ann["caption"] for ann in super()._load_target(id)]
    
    def _pre_caption(self, caption, max_words):
        caption = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        if not len(caption):
            raise ValueError("pre_caption yields invalid text")

        return caption
    
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        captions = self._load_target(id)
        target = []
        for caption in captions:
            caption = self.prompt + self._pre_caption(caption, self.max_words)
            target.append(caption)

        if self.transforms is not None:
            # image, target = self.transforms(image, target)
            image = self.transforms(image)
        
        mask = self.generate_mask()
        
        target = random.sample(target, 1)
        # target = open_clip.tokenize(target, context_length=self.max_words)

        return image, mask, target


if __name__ == "__main__":
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

    print(dataset.__getitem__(0))


