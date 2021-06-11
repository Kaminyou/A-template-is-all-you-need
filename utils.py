def get_encoder(base_encoder_name, contrastive_framework, **kwargs):
    """
    Get an encoder of specified framework.
    
    Input:
        - base_encoder_name (str): which resnet backbone to use.
            choices: ['resnet18', 'resnet34', 'resnet50', 'resnet101']
            
        - contrastive_framework (str): which framework to use.
            choices: ['simclr', 'moco', 'byol', 'simsiam']
            
        - kwargs: kwargs to be passed down to each framework, including
            pretrained (bool): whether to use pretrained weights (default: True)
            low_dim (int): the dimension of the encoder's projection (default: 128)
            m (float): momentum to update encoder if a momentum encoder is 
                       used in the framework (default: 0.99)
            T (float): softmax temperature in info_nce loss (default: 0.07)
            K (int): queue size in MoCo (default: 8192)
    """    
    # the framework of SimSiam is almost the same as BYOL.
    # the only difference is whether to use momentum encoder.
    if contrastive_framework == 'simsiam':
        contrastive_framework = 'byol'
        kwargs['use_momentum'] = False
        
    # import the specified framework
    class_map = {
        'simclr': 'SimCLR',
        'moco':   'MoCo',
        'byol':   'BYOL'
    }        
    from importlib import import_module
    module = import_module('encoders.'+contrastive_framework)
    framework = getattr(module, class_map[contrastive_framework])
    
    from encoders.resnet import ResNetEncoder
    encoder = framework(ResNetEncoder, **kwargs)
    
    return encoder