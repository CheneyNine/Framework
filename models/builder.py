from .STGCN import STGCN

def get_model_instance(config, data_module):
    model_name = config['model']['name']

    if model_name == 'STGCN':
        return STGCN(config, data_module)
    else:
        raise ValueError(f"Model {model_name} is not implemented.")