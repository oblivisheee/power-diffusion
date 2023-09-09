import torch
from safetensors.torch import load_file
import os
import time

def load(model_path):
    if ".ckpt" in model_path:
        try:
            checkpoint = torch.load(model_path)
            model = torch.nn.Module()
            model.load_state_dict(checkpoint['state_dict'])
            print("Model has been successfully loaded.")
            return model
        except Exception as e:
            print("Error while loading model:", str(e))
            return None
    elif ".safetensors" in model_path:
        try:
            model = torch.nn.Module()
            checkpoint = load_file(model_path)
            model.load_state_dict(checkpoint)
            print("Model has been successfully loaded.")
            return model

        except Exception as e:
            print("Error while loading model:", str(e))
            return None
    else:
        raise ValueError('Your model is unsupported.\nUse model with ".ckpt" or ".safetensors" extensions.')


