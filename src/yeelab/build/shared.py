"""A method that evaluates each of its parts once.

A method built from blocks often holds the same method several times: in a condition and
in the branch it chooses, as a finalist of two runoffs, as the poll of a strategy and in
the method itself. Each block evaluates its parts when asked, so such a part is worked
out as many times as it is written. shared(method) is the same method with every Winner
and Condition in it remembering what it gave for the voters it was asked about, so that it
elects the same candidates with the same margins, each part computed once.

A part is known by its repr, the expression that builds it, and the voters by the object:
a strategy's method runs on voters of its own (strategic.py), which are other voters.
"""

from copy import copy
from dataclasses import dataclass, field, fields

import numpy as np

from yeelab.build.blocks import Block, Condition, Winner


class _Results:
    """What the parts of one method gave, by (voters, part); the voters are held, so that
    no other ones get their id."""

    def __init__(self):
        self.entries: dict = {}

    def get(self, voters, key, compute) -> tuple:
        full = (id(voters), key)
        entry = self.entries.get(full)
        if entry is None:
            entry = self.entries[full] = (voters, tuple(np.array(part) for part in compute()))
        return tuple(part.copy() for part in entry[1])  # the caller may change it in place


@dataclass(frozen=True, eq=False)
class _SharedWinner(Winner):
    inner: Winner
    key: str = field(repr=False)
    results: _Results = field(repr=False)

    @property
    def seats(self):
        return self.inner.seats

    @property
    def needs(self):
        return self.inner.needs

    def select(self, voters):
        return self.results.get(voters, ("select", self.key), lambda: self.inner.select(voters))

    def evaluate(self, voters):
        return self.results.get(voters, ("evaluate", self.key), lambda: self.inner.evaluate(voters))

    def __repr__(self):
        return repr(self.inner)


@dataclass(frozen=True, eq=False)
class _SharedCondition(Condition):
    inner: Condition
    key: str = field(repr=False)
    results: _Results = field(repr=False)

    @property
    def needs(self):
        return self.inner.needs

    def evaluate(self, voters):
        return self.results.get(voters, ("condition", self.key), lambda: self.inner.evaluate(voters))

    def __repr__(self):
        return repr(self.inner)


def _wrapped(block, results: _Results):
    key = repr(block)
    changes = {item.name: _wrapped(value, results) for item in fields(block)
               if isinstance(value := getattr(block, item.name), Block)}
    if changes:  # a copy with the fields set: the constructors of some do not take them all
        block = copy(block)
        for name, value in changes.items():
            object.__setattr__(block, name, value)
    if isinstance(block, Winner):
        return _SharedWinner(block, key, results)
    return _SharedCondition(block, key, results) if isinstance(block, Condition) else block


def repeats(block) -> bool:
    """Whether some Winner or Condition is in the block more than once."""
    seen: set = set()

    def twice(part) -> bool:
        if isinstance(part, Winner | Condition):
            key = repr(part)
            if key in seen:
                return True
            seen.add(key)
        return any(twice(value) for item in fields(part)
                   if isinstance(value := getattr(part, item.name), Block))

    return twice(block)


def shared(method: Winner) -> Winner:
    """The method with each of its parts evaluated once for the same voters; the method
    itself where no part repeats. For one evaluation: it remembers every result."""
    return _wrapped(method, _Results()) if repeats(method) else method
