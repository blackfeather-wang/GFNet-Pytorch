from PIL import Image

configurations = {
    'resnet50': {
        'feature_num': 2048,
        'feature_map_channels': 2048,
        'policy_conv': False,
        'policy_hidden_dim':1024,      
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bicubic'
    },
    'densenet121': {
        'feature_num': 1024,
        'feature_map_channels': 1024,
        'policy_conv': False,
        'policy_hidden_dim':1024,      
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bilinear'
    },
    'densenet169': {
        'feature_num': 1664,
        'feature_map_channels': 1664,
        'policy_conv': False,
        'policy_hidden_dim':1024,      
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bilinear'
    },
    'densenet201': {
        'feature_num': 1920,
        'feature_map_channels': 1920,
        'policy_conv': False,
        'policy_hidden_dim':1024,      
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bilinear'
    },
    'mobilenetv3_large_100': {
        'feature_num': 1280,
        'feature_map_channels': 960,
        'policy_conv': True,
        'policy_hidden_dim': 256,      
        'fc_rnn': False,
        'fc_hidden_dim': None,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bicubic'
    },
    'mobilenetv3_large_125': {
        'feature_num': 1280,
        'feature_map_channels': 1200,
        'policy_conv': True,
        'policy_hidden_dim': 256,      
        'fc_rnn': False,
        'fc_hidden_dim': None,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bicubic'
    },
    'efficientnet_b2': {
        'feature_num': 1408,
        'feature_map_channels': 1408,
        'policy_conv': True,
        'policy_hidden_dim': 256,      
        'fc_rnn': False,
        'fc_hidden_dim': None,
        'image_size': 260,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BICUBIC,
        'prime_interpolation': 'bicubic'
    },
    'efficientnet_b3': {
        'feature_num': 1536,
        'feature_map_channels': 1536,
        'policy_conv': True,
        'policy_hidden_dim': 256,      
        'fc_rnn': False,
        'fc_hidden_dim': None,
        'image_size': 300,
        'crop_pct': 0.904,
        'dataset_interpolation': Image.BICUBIC,
        'prime_interpolation': 'bicubic'
    },
    'regnety_600m': {
        'feature_num': 608,
        'feature_map_channels': 608,
        'policy_conv': True,
        'policy_hidden_dim': 256,      
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bilinear',
        'cfg_file': 'pycls/cfgs/RegNetY-600MF_dds_8gpu.yaml'
    },
    'regnety_800m': {
        'feature_num': 768,
        'feature_map_channels': 768,
        'policy_conv': True,
        'policy_hidden_dim': 256,      
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bilinear',
        'cfg_file': 'pycls/cfgs/RegNetY-800MF_dds_8gpu.yaml'
    },
    'regnety_1.6g': {
        'feature_num': 888,
        'feature_map_channels': 888,
        'policy_conv': True,
        'policy_hidden_dim': 256,      
        'fc_rnn': True,
        'fc_hidden_dim': 1024,
        'image_size': 224,
        'crop_pct': 0.875,
        'dataset_interpolation': Image.BILINEAR,
        'prime_interpolation': 'bilinear',
        'cfg_file': 'pycls/cfgs/RegNetY-1.6GF_dds_8gpu.yaml'
    }
}
