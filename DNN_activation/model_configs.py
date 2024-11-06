# https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.mdS

import config as c

model_configs = {
    # supervised models
    "resnet_torchvision":[
        "config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear",
            "config.MODEL.TRUNK.RESNETS.DEPTH=50",
            f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={c.model_path}resnet50-19c8e357.pth",
            "config.MODEL.WEIGHTS_INIT.APPEND_PREFIX='trunk.base_model._feature_blocks.'",
            "config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=''",
            'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
            'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
            'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
            'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
            'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["conv1", ["Identity", []]],["res1", ["Identity", []]],["res2", ["Identity", []]],["res3", ["Identity", []]],["res4", ["Identity", []]],["res5avg", ["Identity", []]]]'
    ],
    "resnet_caffe2":[
        "config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear",
            "config.MODEL.TRUNK.RESNETS.DEPTH=50",
            f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={c.model_path}converted_vissl_rn50_supervised_in1k_caffe2.torch",
            "config.MODEL.WEIGHTS_INIT.APPEND_PREFIX='trunk.base_model.'",
            "config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME='model_state_dict'",
            'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
            'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
            'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
            'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
            'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["conv1", ["Identity", []]],["res1", ["Identity", []]],["res2", ["Identity", []]],["res3", ["Identity", []]],["res4", ["Identity", []]],["res5avg", ["Identity", []]]]'
    ],
    
    # self-supervised models
    "simclr_resnet":[
        'config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear',
        'config.MODEL.TRUNK.RESNETS.DEPTH=50',
        'config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=classy_state_dict',
        f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={c.model_path}model_final_checkpoint_phase999.torch ', # Specify path for the model weights.
        'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
        'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
        'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
        'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
        'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["conv1", ["Identity", []]],["res1", ["Identity", []]],["res2", ["Identity", []]],["res3", ["Identity", []]],["res4", ["Identity", []]],["res5avg", ["Identity", []]]]'
    ],
    "barlowtwin_resnet":[
        'config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear',
        'config.MODEL.TRUNK.RESNETS.DEPTH=50',
        'config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=classy_state_dict',
        f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={c.model_path}barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch', # Specify path for the model weights.
        'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
        'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
        'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
        'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
        'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["conv1", ["Identity", []]],["res1", ["Identity", []]],["res2", ["Identity", []]],["res3", ["Identity", []]],["res4", ["Identity", []]],["res5avg", ["Identity", []]]]'
    ],
    "swav_resnet":[
            "config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear",
            "config.MODEL.TRUNK.RESNETS.DEPTH=50",
            "config.MODEL.WEIGHTS_INIT.APPEND_PREFIX='trunk.base_model._feature_blocks.'",
            "config.MODEL.WEIGHTS_INIT.REMOVE_PREFIX='module.'",
            "config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=''",
            f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={c.model_path}model_final_checkpoint_phase799.torch",
            'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
            'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
            'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
            'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
            'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["conv1", ["Identity", []]],["res1", ["Identity", []]],["res2", ["Identity", []]],["res3", ["Identity", []]],["res4", ["Identity", []]],["res5avg", ["Identity", []]]]'
          ],
    "deepclusterv2_resnet":[
            "config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear",
            "config.MODEL.TRUNK.RESNETS.DEPTH=50",
            "config.MODEL.WEIGHTS_INIT.APPEND_PREFIX='trunk.base_model._feature_blocks.'",
            "config.MODEL.WEIGHTS_INIT.REMOVE_PREFIX='module.'",
            "config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=''",
            f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={c.model_path}deepclusterv2_800ep_pretrain.pth.tar",
            'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
            'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
            'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
            'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
            'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["conv1", ["Identity", []]],["res1", ["Identity", []]],["res2", ["Identity", []]],["res3", ["Identity", []]],["res4", ["Identity", []]],["res5avg", ["Identity", []]]]'
          ],
    "pirl_resnet":[
        'config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear',
        'config.MODEL.TRUNK.RESNETS.DEPTH=50',
        'config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=classy_state_dict',
        f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={c.model_path}model_final_checkpoint_phase799.torch.1', # Specify path for the model weights.
        'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
        'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
        'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
        'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
        'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["conv1", ["Identity", []]],["res1", ["Identity", []]],["res2", ["Identity", []]],["res3", ["Identity", []]],["res4", ["Identity", []]],["res5avg", ["Identity", []]]]'
    ],
    
}