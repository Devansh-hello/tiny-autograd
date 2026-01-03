# tiny-autograd

building a small autograd engine from scratch so i actually understand how
backprop works instead of just calling `loss.backward()`.

the plan:

- a scalar `Value` type that records the operations done to it
- a `backward()` that walks the graph and fills in gradients
- check every gradient two ways: finite differences and pytorch
- build a tiny neural net on top and train it on a real toy dataset

if the gradients are right and the net trains, the engine works.

still building this.
