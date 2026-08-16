"""S1 training support for the pinned BeatNet observation model.

`model` needs torch, which CI does not install. Importing it here eagerly made
*every* module in the package need torch transitively, including `c1_subsets`,
`c1_summarise` and `c1_launch`, which import nothing heavier than numpy -- so
collection of their tests failed in CI. The obvious repair, an
`importorskip("torch")` in each test file after the pattern
`test_s1_training.py` uses for a module that genuinely needs torch, would have
made that skip permanent. Those are the tests guarding the registered subset
membership rule and the schedule ordering; skipping them buys a green tick and
no protection.

The two torch-backed names are therefore exposed lazily instead, so
`from training.beatnet import BeatNetTrainable` still works and importing a
torch-free module no longer drags torch in behind it.
"""

__all__ = ["BeatNetTrainable", "configure_a3"]


def __getattr__(name: str):
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
