import dataclasses
import types
import click
import pathlib
import typing as tp

from omegaconf import OmegaConf
from . import datasets, models, training
from .trainer import Trainer


@click.group
def cli() -> None:
    pass


class _BaseCamelCaseModuleLookup(click.ParamType):
    module: types.ModuleType
    base: type[tp.Any]

    def convert(self, value, param, ctx):
        if isinstance(value, type) and issubclass(value, self.base):
            return value

        try:
            return next(
                cls
                for attr in self.module.__all__
                if getattr(cls := getattr(self.module, attr), "name", None) == value
            )
        except StopIteration:
            self.fail(
                f"There is no member with name {value} in {self.module}", param, ctx
            )


class DatasetsLookup(_BaseCamelCaseModuleLookup):
    module = datasets
    base = datasets.DataModule


class ModelsLookup(_BaseCamelCaseModuleLookup):
    module = models
    base = models.Module


@cli.command()
@click.option("--task", "dataset_cls", type=DatasetsLookup(), required=True)
@click.option("--model", "model_cls", type=ModelsLookup(), required=True)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default="conf/conf.yaml",
)
@click.option("-o", "--option", "options", multiple=True)
def train(
    dataset_cls: type[datasets.DataModule],
    model_cls: type[models.Module],
    config_file: pathlib.Path,
    options: list[str],
) -> None:
    @dataclasses.dataclass
    class Config:
        task: dataset_cls.config = dataclasses.field(default_factory=dataset_cls.config)
        model: model_cls.config = dataclasses.field(default_factory=model_cls.config)
        trainer: Trainer.config = dataclasses.field(default_factory=Trainer.config)

    conf = OmegaConf.merge(
        OmegaConf.structured(Config),
        OmegaConf.load(config_file),
        OmegaConf.from_dotlist(options),
    )
    print(OmegaConf.to_yaml(conf))

    module, datamodule = training.matrix[dataset_cls, model_cls](
        conf.task, conf.model
    )
    trainer = Trainer(**conf.trainer)
    trainer.fit(module, datamodule)
