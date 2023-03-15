import pathlib
import pprint

from ._utils import ConfigMeta, _filter_self

__all__ = ["LitSimpleArgs", "LitComplexArgs"]


class DataModule(metaclass=ConfigMeta):
    def __init__(self) -> None:
        print(f"called __init__ of {type(self).__name__} with arguments:")


class LitSimpleArgs(DataModule):
    def __init__(
        self, datadir: str | pathlib.Path, batch_size: int, num_workers: int
    ) -> None:
        super().__init__()
        pprint.pp(_filter_self(locals()))


class LitComplexArgs(DataModule):
    def __init__(
        self,
        datadir: str | pathlib.Path,
        batch_size: int,
        val_size: float,
        tokenizer_name: str,
        transforms: list[str],
    ):
        super().__init__()
        pprint.pp(_filter_self(locals()))
