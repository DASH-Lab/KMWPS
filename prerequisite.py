# ------ LIBRARY -------#
"""
1. 라이브러리 로드
"""
import numpy as np
import pandas as pd
import os, sys
import json
import re
import random
import unicodedata
import math
from glob import glob
from collections import OrderedDict, defaultdict
import itertools as it
import itertools
import tqdm

from attrdict import AttrDict
from timeit import default_timer as timer
from time import time
import warnings

warnings.filterwarnings("ignore")
try:
    import cPickle as pickle
except ImportError:
    import pickle

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.cuda.amp as amp
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
from torch.autograd import Variable

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, OneCycleLR

# from torch.optim.optimizer import Optimizer, required

# nlp
from konlpy.tag import Okt
from konlpy.tag import Kkma

# scikit-learn
from sklearn.model_selection import KFold

# transformer model
from transformers import AdamW
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
okt = Okt()