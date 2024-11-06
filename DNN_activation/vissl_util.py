from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from classy_vision.generic.util import load_checkpoint

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import numpy as np

from data_util import getDatasetFromLabel
import config as c
from model_configs import model_configs

batch_size = 128
embeding_dim = 2048
max_num_im = 100
device = torch.device('cuda')

def load_Vissl_Model(model_name):
    # model_name in [resnet_torchvision, resnet_caffe2, simclr_resnet, barlowtwin_resnet, swav_resnet, deepclusterv2_resnet, pirl_resnet]
    cfg =  model_configs[model_name]
    cfg = compose_hydra_configuration(cfg)
    _, cfg = convert_to_attrdict(cfg)
    
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
    init_model_from_consolidated_weights(
        config=cfg,
        model=model,
        state_dict=weights,
        replace_prefix = cfg['MODEL']['WEIGHTS_INIT']['REMOVE_PREFIX'],
        append_prefix = cfg['MODEL']['WEIGHTS_INIT']['APPEND_PREFIX'],
        state_dict_key_name=cfg['MODEL']['WEIGHTS_INIT']['STATE_DICT_KEY_NAME'],
        skip_layers=[],  # Use this if you do not want to load all layers
    )
    
    return model.to(device)

def features_of(model, label):
    transform = v2.Compose([
        v2.Resize(size=(256, 256), antialias=True),
        v2.CenterCrop(size=(224,224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = getDatasetFromLabel(label, top=max_num_im, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    activations = []
    for batch in dataloader:
        with torch.no_grad():
            output = model(batch.to(device))
            activation = [torch.flatten(layer.detach(), start_dim=1).cpu().numpy() for layer in output]
            if not hasattr(model, 'random_index'):
                random_index = [np.random.permutation(activation[i].shape[1])[0:embeding_dim] if activation[i].shape[1]>embeding_dim else np.arange(activation[i].shape[1]) for i in range(len(activation))]
                model.random_index = random_index
            activation = [activation[i][:, model.random_index[i]] for i in range(len(activation))]
        if activations == []:
            activations = activation
        else:
            activations = [np.concatenate((activations[i], activation[i]), axis=0) for i in range(len(activation))]
    return activations


from torchvision import models
import re
def load_torchvision_model(model_name):
    cfg =  model_configs[model_name]
    cfg = compose_hydra_configuration(cfg)
    _, cfg = convert_to_attrdict(cfg)

    weights = torch.load(cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE, map_location=torch.device('cpu'))
    # trunk = weights['classy_state_dict']['base_model']['model']['trunk']
    if 'classy_state_dict' in weights.keys():
        weights = weights['classy_state_dict']['base_model']['model']['trunk']
    if 'model_state_dict' in weights.keys():
        weights = weights['model_state_dict']
    if 'module' in list(weights.keys())[0]:
        weights = {re.sub('module.', '', key) : val for key, val in weights.items()}

    trunk = {re.sub('_feature_blocks\.', '', key) : val for key, val in weights.items()}
    dummy_weight = torch.rand((1000, 2048)) # for head
    dummy_bias = torch.rand((1000, ))
    trunk['fc.weight'] = dummy_weight
    trunk['fc.bias'] = dummy_bias
    model = models.resnet50()
    trunk = {k: trunk[k] for k in set(model.state_dict().keys()).intersection(trunk.keys())}
    model.load_state_dict(trunk)
    model = model.to(device)
    model.eval()
    
    return model

from torchvision.models.feature_extraction import create_feature_extractor
def get_layer_model(model, layer_name):
    #['x', 'layer1.0.relu', 'layer1.1.relu_2', 'layer2.0.relu_1', 'layer2.2.relu', 'layer2.3.relu_2', 'layer3.1.relu_1', 'layer3.3.relu', 'layer3.4.relu_2', 'layer4.0.relu_1', 'layer4.1.relu_1', 'layer4.2.relu', 'layer4.2.relu_1', 'layer4.2.relu_2', 'avgpool']
    return create_feature_extractor(model, return_nodes=[layer_name])
    
def features_of_torchvission(model, label, num_im=30, embeding_dim=embeding_dim):
    transform = v2.Compose([
        v2.Resize(size=(256, 256), antialias=True),
        v2.CenterCrop(size=(224,224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = getDatasetFromLabel(label, top=num_im, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    activations = []
    for batch in dataloader:
        with torch.no_grad():
            output = model(batch.to(device))
            output = output[list(output.keys())[0]]
            activation = torch.flatten(output.detach(), start_dim=1).cpu().numpy()
        if not hasattr(model, 'random_matrix'):
            if activation.shape[1] > embeding_dim:
                M = np.random.randn(embeding_dim, activation.shape[1])
                M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
            else:
                M = np.eye(activation.shape[1])
            model.random_matrix = M
        activation = np.dot(activation, model.random_matrix.T)
        if activations == []:
            activations = activation
        else:
            activations = np.concatenate((activations, activation), axis=0)
    return activations