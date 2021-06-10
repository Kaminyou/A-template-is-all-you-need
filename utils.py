def get_encoder(base_encoder_name, contrastive_framework, **kwargs):
    """
    Get an encoder of specified framework.
    
    Input:
        - base_encoder_name (str): which resnet backbone to use.
            choices: ['resnet18', 'resnet34', 'resnet50', 'resnet101']
        - contrastive_framework (str): which framework to use.
            choices: ['simclr', 'moco', 'byol', 'simsiam']
    """
    # get the base encoder
    from encoders import resnet
    base_encoder = getattr(resnet, base_encoder_name)
    
    # the framework of SimSiam is almost the same as BYOL.
    # the only difference is whether to use momentum encoder.
    if contrastive_framework == 'simsiam':
        contrastive_framework = 'byol'
        kwargs['use_momentum'] = False
        
    # import the specified framework
    class_map = {
        'simclr': 'SimCLR',
        'moco':  'MoCo',
        'byol': 'BYOL'
    }        
    from importlib import import_module
    module = import_module('encoders.'+contrastive_framework)
    framework = getattr(module, class_map[contrastive_framework])
    
    
    encoder = framework(base_encoder, **kwargs)
    
    return encoder