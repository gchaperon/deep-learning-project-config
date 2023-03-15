from omegaconf import DictConfig
import typing as tp

from . import datasets, models


class CompatibilityMatrix(dict):
    def register(
        self, *, task: type[datasets.DataModule], model: type[models.Module]
    ) -> tp.Callable[[tp.Any], None]:
        key = (task, model)

        def composed(
            task_config: DictConfig, model_config: DictConfig
        ) -> tuple[models.Module, datasets.DataModule]:
            return model(**model_config), task(**task_config)

        self[key] = composed

        def _override_callback(value: tp.Any) -> None:
            self[key] = value

        return _override_callback

    @property
    def task_names(self) -> list[str]:
        return sorted({task.name for task, _ in self.keys()})

    @property
    def model_names(self) -> list[str]:
        return sorted({model.name for _, model in self.keys()})


matrix = CompatibilityMatrix()
matrix.register(task=datasets.LitSimpleArgs, model=models.LitRNN)
matrix.register(task=datasets.LitSimpleArgs, model=models.LitLSTM)
matrix.register(task=datasets.LitComplexArgs, model=models.LitConvNet)


@matrix.register(task=datasets.LitComplexArgs, model=models.LitLSTM)
def custom_initialization(task_config: DictConfig, model_config: DictConfig) -> tuple:
    task_config.tokenizer_name = "custom-tokenizer-name"
    model_config.vocab_size = 5 * task_config.batch_size
    return models.LitLSTM(**model_config), datasets.LitComplexArgs(**task_config)
