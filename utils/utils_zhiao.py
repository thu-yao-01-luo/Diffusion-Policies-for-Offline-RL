import argparse
from .builder import BuilderConfig
from typing import Union, TypeVar, cast, get_type_hints
from omegaconf import OmegaConf, DictConfig
from typing import Type, Any, Tuple, Dict
from dataclasses import is_dataclass


T = TypeVar('T')
ConfigType = Union[T, DictConfig]
from omegaconf import omegaconf


def dfs_builder(data_cls, config: DictConfig):
    if not is_dataclass(data_cls):
        return config

    if issubclass(data_cls, BuilderConfig):
        sub_config = data_cls.merge_config(config)
        # print(data_cls, sub_config)
        return OmegaConf.create(dict(
            type=config.type,
            config=dfs_builder(data_cls.CONFIGS[config.type], sub_config)
        ))

    hints = get_type_hints(data_cls)
    outs = {}
    # print(data_cls)
    # print(hints, "hints", hints is {})
    for k, v in config.items():
        # print(k, v, hints[str(k)])
        if isinstance(v, DictConfig):
            outs[k] = dfs_builder(hints[str(k)], v)
        else:
            outs[k] = v
    return OmegaConf.create(outs)


def load_config(config_type, other=None, expand=True, verbose=True):
    """
    TODO: load constraints using others and resolve equality constraints.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args, unknown = parser.parse_known_args()

    cfg: ConfigType[config_type] = OmegaConf.structured(config_type)

    if other is not None:
        cfg.merge_with(other)
    OmegaConf.resolve(cfg)

    if args.config is not None:
        cfg.merge_with(OmegaConf.load(args.config))

    input_cfg = OmegaConf.from_dotlist(unknown)

    cfg.merge_with(input_cfg)
    if expand:
        cfg = dfs_builder(config_type, cfg)

    if verbose:
        print(OmegaConf.to_yaml(cfg))
    return cast(config_type, cfg)