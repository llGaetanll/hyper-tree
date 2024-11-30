# `hyper-tree`

A zero-cost generalization over quadtrees and octrees that store points.

- No dependencies!
- 100% safe code
- Small footprint

This is **not** a [kd-tree](https://en.wikipedia.org/wiki/K-d_tree)! While
kd-trees are also used for binary space partitioning, their splits are not
necessarily equal, as they are in quadtrees, octrees, and more general forms.

## What can I do with this?

Right now, not much. This is still very much a work in progress. 

## Work in Progress!

The library needs to be tested and benched.

- [ ] Write tests
- [ ] Benchmark
- [ ] Allow creating padded trees
      Right now, new points can't be added to a tree without having to
      completely rebuild it.
