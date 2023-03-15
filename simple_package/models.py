import pprint
import typing as tp

from ._utils import _filter_self, ConfigMeta

__all__ = ["LitRNN", "LitLSTM", "LitConvNet"]


class Module(metaclass=ConfigMeta):
    def __init__(self) -> None:
        print(f"called __init__ of {type(self).__name__} with arguments:")


class LitRNN(Module):
    name = "lit-rnn"

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        # NOTE: this should  be Literal["tanh", "relu"], but is not supported
        # by omegaconf
        nonlinearity: str,
        dropout: float,
        learn_rate: float,
    ):
        super().__init__()
        pprint.pp(_filter_self(locals()))


class LitLSTM(Module):
    name = "lit-lstm"

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        projection_size: int,
        learn_rate: float,
    ) -> None:
        super().__init__()
        pprint.pp(_filter_self(locals()))


class LitConvNet(Module):
    def __init__(self, input_size: int, output_size: int, learn_rate: int) -> None:
        super().__init__()
        pprint.pp(_filter_self(locals()))
