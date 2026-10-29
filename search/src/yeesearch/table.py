"""Every method a run scored, best first, as a Python file to read and search (Ctrl+F) in
an editor: a line (J, sincere, burial, size, method) each, the methods written short, a
subtree that is a method of yeelab.build.methods by its name (fptp, nanson, star, ...) and
the frequent ones without a name by the ALIASES defined at the top of the file."""

import re
from pathlib import Path

from yeelab.build.methods import METHODS

from yeesearch.pool import read_jsonl

ALIASES = {  # the subtrees the search uses most that have no name in yeelab.build.methods
    "score6": 'Score(6, power=1.5)',
    "score2": 'Score(2, power=1.5)',
    "margins": 'Margins(Pairwise())',
    "cmp6": 'ScoreComparisons(Score(6, power=1.5))',
    "cmp6_ties": 'ScoreComparisons(Score(6, power=1.5), tie_weight=1.0)',
    "cmp2": 'ScoreComparisons(Score(2, power=1.5))',
    "cmp2_ties": 'ScoreComparisons(Score(2, power=1.5), tie_weight=1.0)',
    "median6": 'Tally(Score(6, power=1.5), statistic="median")',
    "median2": 'Tally(Score(2, power=1.5), statistic="median")',
    "top2": 'Highest(Tally(Plurality()), n=2)',
    "top3": 'Highest(Tally(Plurality()), n=3)',
}


def _names() -> list[tuple[re.Pattern, str]]:
    """(the pattern of a subtree, its name) of the methods and the aliases, the longest
    first, so that a method is named before the shorter ones inside it."""
    named = {repr(method): name for name, method in METHODS.items()} | {text: name for name, text in ALIASES.items()}
    return [(re.compile(r"(?<![A-Za-z0-9_])" + re.escape(text)), name)
            for text, name in sorted(named.items(), key=lambda item: -len(item[0]))]


NAMES = _names()


def short(source: str, names=NAMES) -> str:
    """The repr of a method with every subtree that has a name written by it. A match
    starts at a block's name and ends with its closing bracket: a whole subtree."""
    for pattern, name in names:
        source = pattern.sub(name, source)
    return source


def _entries(runs: list[Path]) -> list[dict]:
    """The methods scored on the training set of the runs, the last run's entry of one
    scored in several."""
    entries = {}
    for run in runs:
        for entry in read_jsonl(run / "evals.jsonl"):
            entries[entry["method"]] = entry
    return sorted(entries.values(), key=lambda entry: (entry["J"], entry["method"]))


def write_table(runs: list[Path], out: Path, max_j: float | None = None) -> int:
    """`out` with a line of every method of the runs (of J at most `max_j`), best first; those
    that failed (J inf) last. The number of methods written."""
    entries = [entry for entry in _entries(runs) if max_j is None or entry["J"] <= max_j]
    aliases = [(name, short(text, [each for each in NAMES if each[1] != name])) for name, text in ALIASES.items()]
    with out.open("w") as file:
        file.write(f"# {', '.join(str(run) for run in runs)}: {len(entries)} methods scored on the training set,"
                   " best first\n# names: the methods of yeelab.build.methods and these\n\n")
        file.writelines(f"{name} = {text}\n" for name, text in aliases)
        file.write('inf = float("inf")  # failed\n\n#  J(0.3)   sincere   burial   size\n')
        for entry in entries:
            numbers = ", ".join(f"{entry[key]:8.6f}" if key in entry else "    None"
                                for key in ("J", "sincere", "burial"))
            file.write(f"({numbers}, {entry.get('size', 0):3d}, {short(entry['method'])})\n")
    return len(entries)


COLOURS = (  # (pattern, rich style) of highlighted(), the later over the earlier
    (r"\b\d+(\.\d+)?\b|\binf\b", "cyan"),
    (r"\b[A-Z][A-Za-z0-9]*(?=\()", "bold bright_blue"),  # a block
    (r"\b[a-z_]+(?==)", "grey50"),  # a keyword argument
    (r'"[^"]*"', "dark_orange"),
    (r"==|!=|<|>|\||&", "bold red"),
    (r"\b(" + "|".join(sorted(ALIASES, key=len, reverse=True)) + r")\b", "green"),
    (r"\b(" + "|".join(sorted(METHODS, key=len, reverse=True)) + r")\b", "bold yellow"),
)


def highlighted(line: str, matches=()):
    """A line of write_table() as rich Text: the blocks blue, the methods of
    yeelab.build.methods yellow, the aliases green, and what `matches` find underlined."""
    from rich.text import Text

    text = Text(line)
    for pattern, style in COLOURS:
        text.highlight_regex(pattern, style)
    for match in matches:
        text.highlight_regex(match, "underline")
    return text


INLINE = 50  # characters of a part of a method kept on one line (layout)
LINE = 80  # characters of a line of layout() at most, but of a part of INLINE deep in
OPERATORS = (" | ", " & ", " == ", " != ", " < ", " > ")


def _split(text: str, separators) -> list[tuple[str, str]]:
    """[(separator before it, part)] of `text` split at the separators outside brackets
    and strings."""
    parts, depth, start, quoted, i, before = [], 0, 0, False, 0, ""
    while i < len(text):
        char = text[i]
        if char == '"':
            quoted = not quoted
        elif not quoted and char in "([":
            depth += 1
        elif not quoted and char in ")]":
            depth -= 1
        elif not quoted and depth == 0:
            separator = next((each for each in separators if text.startswith(each, i)), None)
            if separator is not None:
                parts.append((before, text[start:i]))
                before, i = separator.strip(), i + len(separator)
                start = i
                continue
        i += 1
    return parts + [(before, text[start:])]


def _closes_at_end(text: str, opening: int) -> bool:
    """Whether the bracket at `opening` closes at the end of `text`."""
    depth, quoted = 0, False
    for i in range(opening, len(text)):
        if text[i] == '"':
            quoted = not quoted
        elif not quoted and text[i] == "(":
            depth += 1
        elif not quoted and text[i] == ")":
            depth -= 1
            if depth == 0:
                return i == len(text) - 1
    return False


def layout(method: str, indent: int = 0) -> str:
    """A method written short on several lines to read: a part longer than INLINE goes on
    a line of its own, 4 deeper, after the argument it is (among=), the arguments after it
    on lines of their own 2 deeper; a part of A | B (==, & ...) on a line each."""
    if len(method) <= INLINE:
        return method
    terms = _split(method, OPERATORS)
    if len(terms) > 1:
        lines = [f"{operator} {layout(term, indent + len(operator) + 1)}" if operator else layout(term, indent)
                 for operator, term in terms]
        return ("\n" + " " * indent).join(lines)
    if method.startswith("(") and _closes_at_end(method, 0):
        return "(" + layout(method[1:-1], indent + 1) + ")"
    call = re.match(r"[A-Za-z_]\w*\(", method)
    if not call or not _closes_at_end(method, call.end() - 1):
        return method
    out, broken = call.group(), False
    for k, (_, argument) in enumerate(_split(method[call.end():-1], (", ",))):
        keyword = re.match(r"[a-z_]+=(?!=)", argument)
        keyword = keyword.group() if keyword else ""
        value = argument[len(keyword):]
        column = len(out.rsplit("\n", 1)[-1]) + (0 if "\n" in out else indent)
        fits = len(argument) <= INLINE
        if broken or (k and fits and column + len(argument) + 2 > LINE):
            out += ",\n" + " " * (indent + 2)
            broken = True
        elif k:
            out += ", "
        if fits:
            out += argument
        else:
            out += keyword + "\n" + " " * (indent + 4) + layout(value, indent + 4)
            broken = True
    return out + ")"


def row(line: str) -> tuple[str, str, str, str, str]:
    """(J, sincere, burial, size, method) of a line of write_table(), as written."""
    numbers, method = line[1:-1].split(", ", 4)[:4], line[1:-1].split(", ", 4)[4]
    return (*(number.strip() for number in numbers), method)


def matchers(patterns: list[str], regex: bool = False) -> list[re.Pattern]:
    return [re.compile(pattern if regex else re.escape(pattern)) for pattern in patterns]


def find(table: Path, patterns: list[str], regex: bool = False) -> tuple[list[str], list[tuple[int, str]]]:
    """(the lines of the aliases, [(rank, line)] of the methods of a file of write_table()
    that hold every one of `patterns`, as text or with `regex` as regular expressions)."""
    matches = matchers(patterns, regex)
    aliases, found, rank = [], [], 0
    with table.open() as file:
        for line in file:
            line = line.rstrip("\n")
            if not line.startswith("("):
                if " = " in line and not line.startswith("inf"):
                    aliases.append(line)
                continue
            rank += 1
            if all(match.search(line) for match in matches):
                found.append((rank, line))
    return aliases, found
