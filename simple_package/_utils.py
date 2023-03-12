import typing as tp


def _filter_self(d: dict[str, tp.Any]) -> dict[str, tp.Any]:
    return {k: v for k, v in d.items() if k != "self"}
