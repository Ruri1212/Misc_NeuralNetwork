## mlflowの結果を読み込み，必要に応じてソート
## 任意の結果を表示するコード

import os
import time
import copy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, models, transforms

from PIL import Image
import matplotlib.pyplot as plt

from omegaconf import OmegaConf, open_dict,DictConfig
import mlflow
from mlflow.models import infer_signature


client = mlflow.tracking.MlflowClient()
experiment_id = "878525502675615922"
best_run = client.search_runs(
    experiment_id, order_by=["metrics.val_loss"], max_results=1
)[0]
print(best_run)