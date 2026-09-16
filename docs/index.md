# CPP-MDANCE

A C++ implementation of [MDANCE](https://github.com/mqcomplab/MDANCE), a flexible n-ary
clustering package for molecular dynamics and general-purpose data.

The library exposes the clustering algorithms (KMeans/NANI, HELM, eQUAL, and the optional
DIVINE), extended n-ary similarity analysis, PRIME representative-frame prediction, and a
set of frame-selection tools. All of it is reachable from the `mdance-cli` command-line
front end.

```{toctree}
:maxdepth: 2
:caption: Contents

mdance-cli-quickstart
```

## Getting started

```shell
cmake -S . -B build
cmake --build build -j

./build/cli/mdance-cli --algorithm kmeans \
                       --input tests/data/sim.csv --output result.json \
                       --natoms 50 --nclusters 10
```

Eigen and GoogleTest are downloaded automatically if they are not already installed.

See the [`mdance-cli` quickstart](mdance-cli-quickstart.md) for the input format, all five
modes, the output schema, and the pitfalls worth knowing about.

## Building these docs

```shell
.venv/bin/pip install sphinx myst-parser furo
.venv/bin/sphinx-build -b html docs docs/_build/html
```

The rendered site lands in `docs/_build/html/index.html`.
