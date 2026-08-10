# Contributing

Bug reports and patches are welcome. A few things specific to this project.

## Every engineering number must be reproducible

No number in the documentation, the README or a docstring is written by hand.
`docs/scripts/collect_doc_numbers.py` regenerates all of them; if you change
something that moves a quoted figure, re-run it and update the text in the same
commit.

The same rule applies to claims. "Faster", "more accurate" and "converges
better" need a measurement in the commit message or the PR, not an assertion.

## Validate against closed forms, not against previous output

`tests/test_analytical.py` checks the simulator against answers that exist
independently of it — Ohm's law, the Shockley equation, `gm = 2·Id/Vov`,
`GBW = gm1/2πCc`, Pelgrom's √area. A regression test that pins today's output
catches changes but cannot catch a bug that was there yesterday. Prefer the
former kind when adding coverage for physics.

If you add a device or an analysis, add a closed-form case for it.

## Running the suite

```bash
pip install -e ".[all]"
pytest -q                    # everything
pytest -q -m "not slow"      # the fast subset
pytest -q -m ngspice         # cross-validation; skips without an ngspice binary
```

`tests/test_ngspice.py` compares against a real ngspice install. It skips
cleanly when the binary is absent, so it will not block you, but CI installs
ngspice and does run it.

## Frontend

```bash
cd frontend
npm install
npx tsc --noEmit -p tsconfig.json
npm run build
```

The dashboard renders computed results only. Do not add placeholder data,
seeded arrays or a chart that is not backed by an API response — if a value
cannot be computed yet, the UI should say so.

## Style

Match the surrounding code. Comments explain *why*, especially where something
non-obvious was necessary — the tricky parts of this codebase are documented
inline for a reason, and a change that removes the reasoning is worse than one
that leaves it.
