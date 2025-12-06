from .STGCN import STGCN
from .callst import CallST

def get_model_instance(config, data_module):
    model_name = config['model']['name']

    if model_name == 'STGCN':
        return STGCN(config, data_module)
    elif model_name == 'CallST':
        return CallST(config, data_module)
    else:
        raise ValueError(f"Model {model_name} is not implemented.")