"""Generic detector."""
import os
import pickle

import torch

from external.adaptors import yolox_adaptor


class Detector(torch.nn.Module):

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None

        self.initialize_model()

    def initialize_model(self):
        """Initialize the model."""
        self.model = yolox_adaptor.get_model(self.path)

    def forward(self, batch, tag=None):
        batch = batch.half()
        if self.model is None:
            self.initialize_model()
            
        with torch.no_grad():
            output = self.model(batch)

        return output
