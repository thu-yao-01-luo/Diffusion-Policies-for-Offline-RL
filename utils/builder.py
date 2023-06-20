from omegaconf import OmegaConf, DictConfig, omegaconf
from typing import (
    TypeVar,
    Generic,
    MutableMapping,
    Type,
    Any,
    Tuple,
    Dict,
    cast,
    ClassVar,
)
from dataclasses import dataclass, field


def get_dict_key_value_types(ref_type: Any) -> Tuple[Any, Any]:
    args = getattr(ref_type, "__args__", None)
    if args is None:
        # bases = getattr(ref_type, "__orig_bases__", None)
        # if bases is not None and len(bases) > 0:
        #     args = getattr(bases[0], "__args__", None)
        ...

    key_type: Any
    element_type: Any
    if ref_type is None or ref_type == Dict:
        key_type = Any
        element_type = Any
    else:
        if args is not None:
            key_type = args[0]
            element_type = args[1]
        else:
            key_type = Any
            element_type = Any

    return key_type, element_type


omegaconf.get_dict_key_value_types = get_dict_key_value_types  # type: ignore


T = TypeVar("T")


@dataclass
class BuilderConfig(Generic[T]):
    type: str
    config: dict | None = None

    # factories ..
    CONFIGS = {}
    CLS = {}

    @classmethod
    def get_base_class(cls):
        return cls.__orig_bases__[0].__args__[0]  # type: ignore

    @classmethod
    def merge_config(cls, config: DictConfig) -> DictConfig:
        if config.type not in cls.CONFIGS:
            raise ValueError(f"Unknown config {config}.")
        config_type = cls.CONFIGS[config.type]
        cfg = OmegaConf.structured(config_type)
        if config.config is not None:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(config.config))
            assert isinstance(cfg, DictConfig)
        return cfg

    @classmethod
    def build_from_config(cls, config: "BuilderConfig[T]", *args, **kwargs) -> T:
        # print(config)
        # print(cls)
        # print(config.type)
        if config.type not in cls.CLS:
            raise ValueError(f"Unknown type {config.type}.")
        CLS = cls.CLS[config.type]
        return CLS(*args, config=cls.merge_config(cast(DictConfig, config)), **kwargs)

    @classmethod
    def register(cls, config_type, name: str | None = None):
        def wrapper(var):
            # print(var)
            # print(cls.get_base_class())
            assert issubclass(var, cls.get_base_class())
            _name = name or var.__name__
            cls.CONFIGS[_name] = config_type
            cls.CLS[_name] = var
            return var

        return wrapper
