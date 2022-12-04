import os
import yaml
import json
import ml_collections
from collections import OrderedDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
data_root = os.path.join(os.path.dirname(__file__), 'data')


class YamlConf(object):
    def __init__(self, conf):
        self.conf = self.load(conf)

    @staticmethod
    def load(conf_path):
        if isinstance(conf_path, str):
            with open(conf_path, 'r') as f:
                conf = YamlConf._yaml_load(f)
        elif isinstance(conf_path, dict):
            conf = conf_path
        elif isinstance(conf_path, YamlConf):
            conf = conf_path.conf
        else:
            conf = None
        return conf

    @staticmethod
    def save(conf_path, data_dict: dict, **kwds):
        with open(conf_path, 'w') as f:
            YamlConf._yaml_dump(data_dict, f, **kwds)

    @staticmethod
    def _yaml_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(stream, OrderedLoader)

    @staticmethod
    def _yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
        class OrderedDumper(Dumper):
            pass

        def _dict_representer(dumper, data):
            return dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                data.items())

        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwds)

    def __str__(self):
        return json.dumps(self.conf, ensure_ascii=False, indent=4, sort_keys=True)

    def __getitem__(self, key):
        assert key in self.conf.keys()
        return ml_collections.FrozenConfigDict(self.conf[key])

    def remove(self, key):
        del self.conf[key]
