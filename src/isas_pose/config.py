from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    window_size: int = 90
    stride: int = 15
    bag_size: int = 3
    triplets: int = 10000
    encoder_epochs: int = 10
    mil_epochs: int = 10
    encoder_batch_size: int = 128
    mil_batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    smoothing_window: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "learning_rate":
                if value <= 0:
                    raise ValueError("learning_rate must be positive")
            elif name != "seed" and value < 1:
                raise ValueError(f"{name} must be positive")
