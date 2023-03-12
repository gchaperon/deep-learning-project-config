import pprint
from ._utils import _filter_self

from .models import Module
from .datasets import DataModule


class Trainer:
    def __init__(
        self,
        max_epochs: int,
        patience: int,
        seed: int,
        experiment: str,
        deterministic: bool,
    ):
        print(f"called __init__ of {type(self).__name__} with arguments:")
        pprint.pp(_filter_self(locals()))

    def fit(self, module: Module, datamodule: DataModule) -> None:
        print(
            f"called fit of {type(self).__name__} "
            f"with module of type {type(module).__name__} "
            f"and datamodule of type {type(datamodule).__name__}"
        )
