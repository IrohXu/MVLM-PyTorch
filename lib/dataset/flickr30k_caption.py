# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Wenqian Ye
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
import torchvision.transforms as transforms
import json_tricks as json

logger = logging.getLogger(__name__)



