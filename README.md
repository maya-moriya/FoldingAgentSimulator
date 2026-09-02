# FoldingAgent Simulator

Origami folding simulation engine for **FoldingAgent: Inferring Parametric
Origami Procedures from Demonstration Videos**.

`origami` lets you build up a piece of virtual paper vertex by vertex, apply
mountain/valley folds, and render both the folded model and its crease
pattern, either interactively or to `.png` files. It can also import/export
crease patterns and fold sequences in the
[FOLD](https://github.com/edemaine/fold) format.

## Installation

```bash
pip install git+https://github.com/maya-moriya/FoldingAgentSimulator.git
```

Or install from a local clone:

```bash
git clone https://github.com/maya-moriya/FoldingAgentSimulator.git
cd FoldingAgentSimulator
pip install -e .
```

Optional extras:

```bash
pip install "foldingagent-simulator[notebook] @ git+https://github.com/maya-moriya/FoldingAgentSimulator.git"  # Jupyter support for usage_example.ipynb
```

## Usage

```python
from origami.origami import Origami

origami = Origami()
origami.plot()

```

See [`example.ipynb`](example.ipynb) for a complete walkthrough.

## License

MIT — see [LICENSE](LICENSE).
