from __future__ import annotations
import operator
import typing as tp

from . import datasets, models

if tp.TYPE_CHECKING:
    from omegaconf import DictConfig


def _identity(
    task_config: DictConfig, model_config: DictConfig
) -> tuple[DictConfig, DictConfig]:
    """Function that return configs without modifications. Used to register a
    task-model pair, but that doesn't require further processing, and all
    config values come from defaults, file, or command line.
    """
    return task_config, model_config


_KeyT = tuple[type[datasets.DataModule], type[models.Module]]
_ConfigFnT = tp.Callable[
    ["DictConfig", "DictConfig"], tuple["DictConfig", "DictConfig"]
]


class CompatibilityMatrix(dict[_KeyT, _ConfigFnT]):
    class RegisterCallback:
        def __init__(self, parent: dict[_KeyT, _ConfigFnT]) -> None:
            self.parent = parent

        def __getitem__(self, key: _KeyT) -> tp.Callable[[_ConfigFnT], None]:
            def callback(fn: _ConfigFnT) -> None:
                self.parent[key] = fn

            return callback

    @property
    def register(self) -> RegisterCallback:
        return self.RegisterCallback(self)

    @property
    def tasks(self) -> list[type[datasets.DataModule]]:
        return sorted(
            {task for task, _ in self.keys()}, key=operator.attrgetter("name")
        )

    @property
    def models(self) -> list[type[models.Module]]:
        return sorted(
            {model for _, model in self.keys()}, key=operator.attrgetter("name")
        )


matrix = CompatibilityMatrix()
matrix[datasets.LitSimpleArgs, models.LitRNN] = _identity
matrix[datasets.LitSimpleArgs, models.LitLSTM] = _identity
matrix[datasets.LitComplexArgs, models.LitConvNet] = _identity


@matrix.register[datasets.LitComplexArgs, models.LitLSTM]
def custom_initialization(
    task_config: DictConfig,
    model_config: DictConfig,
) -> tuple[DictConfig, DictConfig]:
    task_config.tokenizer_name = "custom-tokenizer-name"
    model_config.vocab_size = 5 * task_config.batch_size
    return task_config, model_config
