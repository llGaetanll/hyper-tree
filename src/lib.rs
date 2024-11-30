#![deny(unsafe_code)]

mod bound;
mod traits;
mod util;

use bound::Bound;

#[derive(Clone)]
pub struct Point<T, const N: usize>(pub [T; N]);

pub struct Tree<T, const N: usize> {
    bound: Bound<T, N>,
    root: usize,
    nodes: Vec<usize>,
    points: Vec<Point<T, N>>,
}

pub type QuadTree<T> = Tree<T, 2>;
pub type OcTree<T> = Tree<T, 3>;

impl<T, const N: usize> Tree<T, N> {
    fn uninit(points: Vec<Point<T, N>>) -> Option<Self>
    where
        T: ::core::cmp::Ord + ::core::marker::Copy,
    {
        let bound = Bound::from_points(&points)?;

        Some(Self {
            bound,
            root: usize::MAX,
            nodes: vec![],
            points,
        })
    }

    pub fn new(points: Vec<Point<T, N>>) -> Option<Self>
    where
        T: traits::Mean
            + traits::Epsilon
            + ::core::ops::Sub<Output = T>
            + ::core::cmp::Ord
            + ::core::fmt::Debug,
    {
        let mut tree = Self::uninit(points)?;

        tree.root = tree.build(tree.bound.clone(), 0..tree.points.len());

        Some(tree)
    }

    fn build(&mut self, bound: Bound<T, N>, rng: ::core::ops::Range<usize>) -> usize
    where
        T: traits::Mean
            + traits::Epsilon
            + ::core::ops::Sub<Output = T>
            + ::core::cmp::Ord
            + ::core::fmt::Debug,
    {
        if rng.is_empty() {
            return usize::MAX;
        }

        let idx = self.nodes.len();
        let children = vec![usize::MAX; util::num_divs::<N>()];
        self.nodes.extend(children);

        if rng.len() == 1 {
            return idx;
        }

        let mid = self.bound.center();

        let ::core::ops::Range { start: lo, end: hi } = rng;

        let mut ranges = vec![0usize; util::num_divs::<N>() + 1];
        self.partition_tree(&mut ranges, mid, 0, N - 1, lo..hi);

        if let Some(bounds) = bound.split() {
            for (i, (bound, range)) in bounds.zip(ranges.windows(2)).enumerate() {
                let &[lo, hi] = range else { unreachable!() };
                self.nodes[idx + i] = self.build(bound, lo..hi);
            }
        }

        idx
    }

    fn partition_tree(
        &mut self,
        rngs: &mut [usize],
        mid: Point<T, N>,
        rng_lo: usize,
        i: usize,
        rng: ::core::ops::Range<usize>,
    ) where
        T: ::core::marker::Copy + ::core::cmp::Ord,
    {
        let ::core::ops::Range { start: lo, end: hi } = rng;
        let m = util::partition_in_place(&mut self.points[lo..hi], |p| p.0[i] < mid.0[i]);

        rngs[rng_lo + i / 2] = m;

        if i == 0 {
            return;
        }

        // lo <= .. <= m <= .. <= hi
        self.partition_tree(rngs, mid.clone(), rng_lo, i - 1, lo..m);
        self.partition_tree(rngs, mid, rng_lo + i / 2, i - 1, m..hi);
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::Point;
    use super::Tree;

    fn p2s(ps: &[[i32; 2]]) -> Vec<Point<i32, 2>> {
        ps.iter().map(|&p| Point(p)).collect()
    }

    fn range2(xs: Range<i32>, ys: Range<i32>) -> Vec<Point<i32, 2>> {
        xs.flat_map(|x| ys.clone().map(move |y| Point([x, y])))
            .collect()
    }

    #[test]
    fn no_panic() {
        Tree::new(p2s(&[[0, 0]]));
        Tree::new(p2s(&[[0, 0], [0, 1], [1, 0], [1, 1]]));
        Tree::new(range2(-5..5, -5..5)).unwrap();
    }
}
