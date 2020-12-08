import os

import torch
import torch.nn.functional as F
import numpy as np

from model import Mapping_network
from model import Generator
from model import Discriminator


class Solver():
    def __init__(self, config):
        # Dataloader
        self.dataloader = ""

        # Config - Model
        self.z_dim = config.z_dim
        self.w_dim = config.w_dim
        self.n_mapping = config.n_mapping

        # Config - Training

        # Config - Test

        # Config - Path

        # Config - Miscellanceous

    def build_model(self):
        pass
