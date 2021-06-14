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
    module = import_module('contrastive.encoders.'+contrastive_framework)
    framework = getattr(module, class_map[contrastive_framework])
    
    from contrastive.encoders.resnet import ResNetEncoder
    encoder = framework(ResNetEncoder, **kwargs)
    
    return encoder

def load_weights_from_contrastive_learning(encoder, state_dict, contrastive_framework):
    """
    Loading state_dict to encoder from the model learned by a contrastive framework.
    
    Input:
        - encoder: an Encoder instance from networks.encoder.
            
        - state_dict: the state_dict of a contrastive-learning model.
            
        - contrastive_framework (str): framework of the state_dict
            choices: ['simclr', 'moco', 'byol', 'simsiam']
    """  
    name_map = {
        'simclr' : 'encoder',
        'moco'   : 'encoder_q',
        'byol'   : 'online_encoder',
        'simsiam': 'online_encoder'
    }
    
    encoder_name = name_map[contrastive_framework]
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith(encoder_name) and not k.startswith(encoder_name+'.fc'):
            # remove prefix
            state_dict[k[len(encoder_name+'.'):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
        
    msg = encoder.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}