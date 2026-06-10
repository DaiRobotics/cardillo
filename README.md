# Cardillo - Discrete Rod

> A research fork of **Cardillo** for reproducing the experiments presented in our manuscript.

---

## Overview

This repository is a **research-oriented fork** of the original
[Cardillo](https://github.com/cardilloproject/cardillo) framework.

Although this package is installed as `cardillo`, this repository is a
research fork of the original Cardillo project and is not the official
upstream distribution.

It contains the source code used for the numerical experiments presented
in the manuscript:

> **"A Mixed Discrete Cosserat Rod Formulation"**  
> *(currently under review)*

Although this package is installed as `cardillo`, this repository is a
research fork of the original Cardillo project and is not the official
upstream distribution.
To improve reproducibility and reduce repository size, many components of
the original project (examples, visualization assets, and unrelated
modules) have been removed, while new implementations required for the
paper have been added.

---

## Relationship to the original project

The original Cardillo project is available at:

https://github.com/cardilloproject/cardillo

This repository is **not intended to replace the upstream project**.
Users interested in the complete simulation framework and the official
examples should refer to the original repository.

We gratefully acknowledge the original authors for making Cardillo
publicly available.

---

## Main modifications

Compared with the upstream repository, this fork includes:

- Implementation of **A Mixed Discrete Cosserat Rod Formulation**
- Additional functionality developed for the manuscript
- Scripts used to reproduce the numerical experiments
- Problem-specific examples and benchmark cases
- Bug fixes and project-specific adaptations

At the same time, the following components have been intentionally
removed:

- Unrelated demonstration examples
- Visualization assets and animations
- Unused modules
- Experimental utilities not required for the paper

The directory structure may vary depending on the development
version.

---

## Installation

Clone the repository

```bash
git clone https://github.com/DaiRobotics/cardillo.git
cd cardillo
git checkout v1.0.0-pamm2026
```

Install using pip

```bash
pip install .
```

It is recommended to use a dedicated virtual environment.

---

## Reproducing the paper

The released version of this repository corresponds to the implementation
used in our manuscript.

```bash
python examples/example_helix.py
```

Please refer to the scripts inside the `examples/` directory for reproducing the numerical results.

---

## Release information

The GitHub Release

> **v1.0.0-pamm2026**

corresponds to the version used to generate the results reported in the
submitted manuscript.

Later commits may include additional changes and therefore may not
reproduce the published results exactly.

---

## Citation

If you use this repository in your research, please cite our paper.

The BibTeX entry will be added after publication.

If your work relies on the underlying Cardillo framework, please also
cite the original Cardillo project and its associated publications.

---

## Acknowledgements

This work is based on the open-source **Cardillo** framework developed by
its original authors.

We sincerely thank the Cardillo developers for providing the software
that made this research possible.

---

## License

This repository follows the license of the original Cardillo project
unless explicitly stated otherwise.

Please refer to the `LICENSE` file for details.
