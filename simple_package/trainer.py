import pprint

from ._utils import _filter_self, ConfigMeta
from .datasets import DataModule
from .models import Module


class Trainer(metaclass=ConfigMeta):
    def __init__(
        self,
        deterministic: bool,
        max_epochs: int = 1000,
        patience: int = 3,
        seed: int = 123,
        experiment: str = "default",
    ):
        print(f"called __init__ of {type(self).__name__} with arguments:")
        pprint.pp(_filter_self(locals()))

    def fit(self, module: Module, datamodule: DataModule) -> None:
        print(
            f"called fit of {type(self).__name__} "
            f"with module of type {type(module).__name__} "
            f"and datamodule of type {type(datamodule).__name__}"
        )
