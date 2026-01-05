# tiny-autograd

building a small autograd engine from scratch so i actually understand how
backprop works instead of just calling `loss.backward()`.

inspired by micrograd, but i'm extending it with a numpy tensor engine and
checking the gradients more carefully than a tutorial usually does.

## the claim i'm trying to prove

> every gradient my engine computes matches both central finite-differences and
> PyTorch autograd to within a tiny tolerance, and a neural net built on top of
> it trains on a real toy dataset.

two independent checks matter here: finite-differences and PyTorch fail in
different ways, so if my gradient agrees with both, it's probably right.

## what's inside

- `tinygrad_scratch/engine.py` - the scalar `Value` type + `backward()`
- `tinygrad_scratch/nn.py` - neurons, layers, an MLP, and a couple of losses
- `tinygrad_scratch/optim.py` - SGD, Momentum, Adam
- `tinygrad_scratch/tensor.py` - a numpy version with broadcasting
- `tests/` - per-op gradient checks, finite-diff, pytorch parity
- `scripts/` - train an MLP on two-moons

## running it

```bash
pip install -e .[dev]
pytest -q
```

## results

coming soon (still building).
