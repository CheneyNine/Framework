import yaml

def load_config(global_config_path, model_config_path):
    def deep_merge(source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                node = destination.setdefault(key, {})
                deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    with open(global_config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    return deep_merge(model_config, config)