""" In this scripts are defined the optimizers used in the project """

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    """
    @abstractmethod
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
