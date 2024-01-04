import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import multiprocessing

import torch
import torch.nn as nn


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

