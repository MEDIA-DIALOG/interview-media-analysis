import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


def load_dialogpt_zeroshot(weight_path: str):
    # Create object
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(config)
    print('{} defined'.format(model.__class__.__name__, ))

    # Obtain weights from file
    model_weights = torch.load(weight_path,
                               map_location=lambda storage, loc: storage)

    # Load model weights
    model_weights = {
        k.replace('module.', ''): v
        for k, v in model_weights.items()
    }
    if 'lm_head.decoder.weight' in model_weights:
        model_weights['lm_head.weight'] = model_weights.pop(
            'lm_head.decoder.weight'
        )  # Compatibility with newer versions of `transformers` package

    model.load_state_dict(model_weights, strict=True)
    print('Model loaded from {}'.format(weight_path))

    return model


def load_finetuned_gpt2(weight_path: str, zero_shot: bool = False):
    # Load the model
    if zero_shot:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        config = GPT2Config.from_pretrained('gpt2')
        model = GPT2LMHeadModel(config)
        print('{} defined'.format(model.__class__.__name__, ))

        model_weights = torch.load(
            weight_path,
            map_location=lambda storage, loc: storage)['state_dict']
        corrected_model_weights = {}
        for k, v in model_weights.items():
            corrected_model_weights[k.replace('model.', '')] = v
        print('Loaded model weights from {}'.format(weight_path))

        model.load_state_dict(corrected_model_weights, strict=True)
        print('{} loaded with checkpoint weights and sent to GPU!'.format(
            model.__class__.__name__, ))

    return model
