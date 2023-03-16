import dataclasses
import pathlib
import types
import typing as tp

import click
import rich.console
import rich.box
import rich.syntax
import rich.table
from omegaconf import OmegaConf

from . import datasets, models, training
from .trainer import Trainer

T = tp.TypeVar("T")


class _BaseChoiceAndLookup(click.Choice, tp.Generic[T]):
    module: types.ModuleType
    base: type[T]

    def convert(
        self, value: tp.Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> type[T]:
        value = super().convert(value, param, ctx)
        if isinstance(value, type) and issubclass(value, self.base):
            return value

        try:
            cls = next(
                cls
                for attr in self.module.__all__
                if getattr(cls := getattr(self.module, attr), "name", None) == value
            )
            if isinstance(cls, type) and issubclass(cls, self.base):
                return cls
            else:
                self.fail(
                    f"There is no subclass of {self.base.__name__} "
                    f"in module {self.module}"
                )

        except StopIteration:
            self.fail(
                f"There is no member with name {value} in {self.module}", param, ctx
            )


class TaskChoice(_BaseChoiceAndLookup[datasets.DataModule]):
    name = "task"
    module = datasets
    base = datasets.DataModule

    def __init__(self, case_sensitive: bool = True) -> None:
        registered_tasks = {
            datamodule_cls.name for datamodule_cls, _ in training.matrix.keys()
        }
        super().__init__(sorted(registered_tasks), case_sensitive)


class ModelChoice(_BaseChoiceAndLookup[models.Module]):
    name = "model"
    module = models
    base = models.Module

    def __init__(self, case_sensitive: bool = True) -> None:
        registered_tasks = {module_cls.name for _, module_cls in training.matrix.keys()}
        super().__init__(sorted(registered_tasks), case_sensitive)


def print_compatibility_matrix(
    ctx: click.Context, param: click.Parameter, value: tp.Any
) -> None:
    if not value or ctx.resilient_parsing:
        return
    symbols = {
        None: "\u2205",
        training._identity: "[bold]\u03bb[/bold]x.x",
    }
    table = rich.table.Table(title="task/model compatibility", box=rich.box.MARKDOWN)
    table.add_column("", style="bold")
    for model in training.matrix.models:
        table.add_column(model.name)

    for task in training.matrix.tasks:
        row = [task.name]
        for model in training.matrix.models:
            function = training.matrix.get((task, model), None)
            # NOTE: on ignore[arg-type], this works but mypy kinda chokes on
            # callable types
            symbol = symbols.get(
                function, "[bold]\u03bb[/bold]x.y"  # type: ignore[arg-type]
            )
            row.append(symbol)
        table.add_row(*row)

    console = rich.console.Console()
    console.print(table)
    ctx.exit()


@click.group(context_settings={"show_default": True})
def cli() -> None:
    pass


@cli.command()
@click.option("-t", "--task", "datamodule_cls", type=TaskChoice(), required=True)
@click.option("-m", "--model", "module_cls", type=ModelChoice(), required=True)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default="/dev/null",
    help="A yaml file with config values for the specific task and model chosen.",
)
@click.option(
    "-o",
    "--option",
    "options",
    multiple=True,
    help=(
        "Extra configuration overrides. The syntax should follow omegaconf's "
        "dotted notation. Ex: -o model.learn_rate=1e-3"
    ),
)
@click.option(
    "--task-compatibility",
    is_flag=True,
    callback=print_compatibility_matrix,
    expose_value=False,
    is_eager=True,
    help="""Show the task-model compatibility table and exit. \u2205 means the
    task-model pair is not compatible, \u03bbx.y means the code has some config
    values hardcoded (i.e. the given config is modified in the code) and
    \u03bbx.x means the config is used as is.""",
)
@click.option(
    "--print-config",
    is_flag=True,
    help="""Print the configuration that is going to be used for this
    experiment and exit. Missing values are shown as `???`. Note that a config
    with missing values is invalid, but it's still shown here.""",
)
def train(
    datamodule_cls: type[datasets.DataModule],
    module_cls: type[models.Module],
    config_file: pathlib.Path,
    options: list[str],
    print_config: bool = False,
) -> None:
    if (datamodule_cls, module_cls) not in training.matrix:
        raise click.ClickException(
            "Invalid combination of task and model. "
            "Run with --task-compatibility to see the available task-model pairs."
        )

    # NOTE: on ignore[name-defined]. This is probably because the config classes
    # are created dynamically, since I didn't want to repeat all the arguments
    # of each model and dataset __init__. The code works because omegaconf is evaluating
    # the types at runtime, where the config classes are indeed defined. This
    # could be fixed (i think) by writing each config class individually.
    @dataclasses.dataclass
    class Config:
        task: datamodule_cls.config = dataclasses.field(  # type: ignore[name-defined]
            default_factory=datamodule_cls.config
        )
        model: module_cls.config = dataclasses.field(  # type: ignore[name-defined]
            default_factory=module_cls.config
        )
        trainer: Trainer.config = dataclasses.field(  # type: ignore[name-defined]
            default_factory=Trainer.config
        )

    conf = OmegaConf.merge(
        OmegaConf.structured(Config),
        OmegaConf.load(config_file),
        OmegaConf.from_dotlist(options),
    )
    # The training matrix is allowed to further edit the config obtained from
    # defaults + config file + command line.
    # The matrix holds functions that receive task and model configs and update
    # them as needed.
    # This final config is then used to instantiate the task and model classes
    conf.task, conf.model = training.matrix[datamodule_cls, module_cls](
        conf.task, conf.model
    )
    if print_config:
        console = rich.console.Console()
        syntax = rich.syntax.Syntax(
            OmegaConf.to_yaml(conf).strip(), "yaml", background_color="default"
        )
        console.print(syntax)
        return

    # validate config
    if any(missing := OmegaConf.missing_keys(conf)):
        raise click.ClickException(
            f"The following config keys are missing: {sorted(missing)}"
        )

    module, datamodule = module_cls(**conf.model), datamodule_cls(**conf.task)
    trainer = Trainer(**conf.trainer)
    trainer.fit(module, datamodule)
