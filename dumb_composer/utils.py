from typing import Callable


def attr_compiler(attr_name: str, attr_flag: str) -> Callable:
    """A decorater that finds all attributes of a class that themselves have an
    attribute called "attr_flag" and stores them in a new attribute called
    "attr_name".
    """

    def decorater(cls):
        out = tuple(
            attr for attr, val in vars(cls).items() if hasattr(val, attr_flag)
        )
        setattr(cls, attr_name, out)
        return cls

    return decorater
