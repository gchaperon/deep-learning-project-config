import pprint

from ._utils import ConfigMeta, _filter_self

__all__ = ["LitRNN", "LitLSTM", "LitConvNet"]


class Module(metaclass=ConfigMeta):
    def __init__(self) -> None:
        print()
        print(f"called __init__ of {type(self).__name__} with arguments:")


class LitRNN(Module):
    _name = "lit-rnn"

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        # NOTE: this should  be Literal["tanh", "relu"], but it is not
        # supported by omegaconf
        nonlinearity: str,
        dropout: float,
        learn_rate: float,
    ):
        super().__init__()
        pprint.pp(_filter_self(locals()))


class LitLSTM(Module):
    _name = "lit-lstm"

    def __init__(
        self,
        vocab_size: int,
        projection_size: int,
        embedding_dim: int = 300,
        learn_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        pprint.pp(_filter_self(locals()))


class LitConvNet(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learn_rate: float = 0.1,
    ) -> None:
        super().__init__()
        pprint.pp(_filter_self(locals()))
