# yee-search

Searches for voting methods built from yee-lab's blocks that stay close to the nearest
candidate both when everyone votes sincerely and when 30 % of the voters bury.

A separate uv project with its own lock and venv, so that yee-lab itself does not depend
on what the search needs.

```sh
cd search
uv run yee-search gp --out runs/first --hours 10      # overnight; the same --out continues a run
uv run yee-search report runs/first                   # the best methods and the known ones on the test set
uv run yee-search score 'Unbeaten(StrongestPaths(Margins(Pairwise())))'
uv run yee-search grids                               # does the coarse training grid order methods as a fine one?
```

A run directory holds `evals.jsonl` (every method scored on the training set),
`validation.jsonl`, `generations.jsonl`, `hall.json` and the report.
