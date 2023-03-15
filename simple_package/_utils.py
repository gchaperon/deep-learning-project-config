import functools
import re
import inspect
import typing as tp
import dataclasses


def _filter_self(d: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Filters self and dunder names."""
    return {k: v for k, v in d.items() if not (k == "self" or re.match(r"^__.+__$", k))}


class ConfigMeta(type):
    @property
    def name(cls) -> str:
        """Naive method to convert to convert CamelCase to lisp-case. Doesn't
        handle acronyms well, i.e HTTP -> h-t-t-p"""
        if "name" in cls.__dict__:
            return cls.__dict__["name"]
        return re.sub(r"(?<!^)(?=[A-Z])", "-", cls.__name__).lower()

    @property
    @functools.cache
    def config(cls) -> type:
        init_parameters = {
            k: v
            for k, v in inspect.signature(cls.__init__).parameters.items()
            if k != "self"
        }
        annotations = {k: v.annotation for k, v in init_parameters.items()}
        class_dict = {
            k: "???" if v.default is inspect.Parameter.empty else v.default
            for k, v in init_parameters.items()
        }
        return dataclasses.dataclass(
            type(
                f"{cls.__name__}Config",
                (),
                {"__annotations__": annotations, **class_dict},
            )
        )
