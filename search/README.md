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
uv run yee-search conditional --out runs/cond         # every Conditional(c, A, B) of the small methods, from their maps
./night.sh 06:00                                      # a night: the Conditionals, then gp from their best, then the report
```

What is scored, on which configurations and how the searches work: `docs/search.pdf`
(`typst compile docs/search.typ`, its data from `uv run python docs/data.py <maps>`).

A run directory holds `evals.jsonl` (every method scored on the training set),
`validation.jsonl`, `generations.jsonl`, `hall.json` and the report.

`conditional` enumerates every method of the grammar up to 8 blocks, computes once where
each one elects whom on every configuration (sincerely and buried), and scores every
Conditional of two of them from those maps (`yeesearch.conditional`); its directory
holds the maps, `train.json`, `validation.json`, the test report and `summary.md`.
