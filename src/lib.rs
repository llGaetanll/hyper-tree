#![deny(unsafe_code)]

mod bound;
mod point;
mod traits;
mod util;

use core::cmp::Ord;
use core::marker::Copy;
use core::ops::Sub;
use std::collections::VecDeque;

use bound::Bound;

use crate::point::Point;
use crate::traits::Epsilon;
use crate::traits::Mean;

pub type QuadTree<T> = Tree<T, 2>;
pub type OcTree<T> = Tree<T, 3>;

pub struct Tree<T, const N: usize> {
    /// The points stored by our tree. These are sorted into 2^N regions by their respective
    /// orthants, and each subregion is sorted in the same way, etc `depth` times.
    points: Vec<Point<T, N>>,

    /// The splits of our tree allow us to index into `points` to recover a particular orthant.
    ///
    /// They are stored in level-order in groups of size 2^N and represent the starting index in
    /// `points` for that orthant.
    splits: Vec<usize>,

    /// The max depth of our tree.
    depth: u32,
}

impl<T, const N: usize> Tree<T, N> {
    fn uninit(points: Vec<Point<T, N>>, depth: u32) -> Self
    where
        T: Ord + Copy,
    {
        // * On depth `d`, we get `2^N^d` splits
        // * 2^N^0 + 2^N^1 + ... + 2^N^d = 2^N^(d + 1) / (2^N - 1)
        // * approximate with 2^N^(d + 1)
        let num_splits = util::num_divs::<N>().pow(depth + 1);
        let splits = Vec::with_capacity(num_splits);

        Self {
            points,
            splits,
            depth,
        }
    }

    pub fn new(points: Vec<Point<T, N>>, depth: u32) -> Self
    where
        T: Mean + Epsilon + Sub<Output = T> + Ord,
    {
        let mut tree = Self::uninit(points, depth);

        tree.build();

        tree
    }

    fn build(&mut self)
    where
        T: Ord + Mean + Epsilon + Sub<Output = T>,
    {
        let n = self.points.len();

        let Some(bound) = Bound::from_points(&self.points) else {
            return;
        };

        let mut keys = vec![0; n];
        let mut split_queue = VecDeque::with_capacity(n);
        let mut bound_queue = VecDeque::with_capacity(n);

        // Buf of indices used for sorting points
        let mut swaps = Vec::with_capacity(n);

        // Small buffer to hold the splits for the current layer
        let mut splits = vec![0; util::num_divs::<N>()];

        split_queue.push_back(0);
        bound_queue.push_back(bound);

        for d in 0..self.depth {
            // A whole layer of the tree contains 2^N^d regions
            let regions = util::num_divs::<N>().pow(d);

            for _ in 0..regions {
                let Some(lo) = split_queue.pop_front() else {
                    unreachable!()
                };

                let Some(bound) = bound_queue.pop_front() else {
                    unreachable!()
                };

                self.splits.push(lo);

                // Splits are only ever not monotone when we are
                // about to process a new layer of the tree
                let hi = split_queue.front().copied().unwrap_or(n);
                let hi = if hi < lo { n } else { hi };

                let points = &mut self.points[lo..hi];
                let keys = &mut keys[lo..hi];

                let mid = bound.center();
                Self::sort_layer(mid, points, keys, &mut swaps, &mut splits);

                split_queue.extend(splits.iter().copied().map(|s| s + lo));
                splits.fill(0);

                // If the bound is minimal we can't do this
                let Some(bounds) = bound.split() else {
                    continue;
                };

                bound_queue.extend(bounds);
            }
        }

        self.splits.extend(split_queue);
    }

    fn sort_layer(
        mid: Point<T, N>,
        points: &mut [Point<T, N>],
        keys: &mut [usize],
        swaps: &mut Vec<usize>,
        splits: &mut [usize],
    ) where
        T: Ord,
    {
        debug_assert_eq!(points.len(), keys.len());
        let n = points.len();

        // For each dimensional axis
        for i in 0..N {
            // For each indexed point
            for (j, p) in points.iter().enumerate() {
                if p.0[i] >= mid.0[i] {
                    keys[j] |= 1 << i;
                }
            }
        }

        // Sort by keys order
        swaps.extend(0..n);
        util::argsort(keys, swaps);
        util::sort_by_argsort(points, swaps);

        Self::compute_splits(keys, splits);

        keys.fill(0);
        swaps.clear();
    }

    fn compute_splits(keys: &[usize], splits: &mut [usize]) {
        for &k in keys {
            splits[k] += 1; // 0 <= k < 2^N
        }

        // Accumulate the list
        for i in 1..util::num_divs::<N>() {
            splits[i] += splits[i - 1];
        }

        splits.rotate_right(1);
        splits[0] = 0;
    }
}

#[cfg(test)]
mod test_sort_layer {
    use std::collections::VecDeque;
    use std::fmt::Debug;

    use crate::Tree;
    use crate::bound::Bound;
    use crate::point::Point;
    use crate::traits::Mean;
    use crate::util;

    /// Returns `mid` and the `split_queue`
    fn sort_layer_wrapper<T: Copy + Ord + Mean + Debug, const N: usize>(
        points: &mut [Point<T, N>],
        lo: usize,
    ) -> (Point<T, N>, VecDeque<usize>) {
        let n = points.len();

        let Some(bound) = Bound::from_points(points) else {
            panic!("Provide at least one point")
        };

        let mid = bound.center();
        let mut keys = vec![0; n];
        let mut swaps = Vec::with_capacity(n);
        let mut splits = vec![0; util::num_divs::<N>()];
        let mut split_queue = VecDeque::with_capacity(n);

        Tree::sort_layer(mid.clone(), points, &mut keys, &mut swaps, &mut splits);

        split_queue.extend(splits.iter().copied().map(|s| s + lo));

        (mid, split_queue)
    }

    #[test]
    fn no_offset() {
        let mut points = [[0, 2], [2, 2], [2, 0], [0, 0]].map(Into::into).to_vec();
        let exp_points = &[[0, 0], [2, 0], [0, 2], [2, 2]].map(Into::into);

        let n = points.len();

        let lo = 0;
        let (mid, mut split_queue) = sort_layer_wrapper(&mut points, lo);

        assert_eq!(points, exp_points);
        assert_eq!(split_queue, [0, 1, 2, 3]);

        let split_queue = split_queue.make_contiguous();
        let Ok([a, b, c, d]) = TryInto::<[usize; 4]>::try_into(split_queue) else {
            unreachable!()
        };

        assert_eq!(mid, (1, 1).into());

        let nw = &points[a..b];
        let ne = &points[b..c];
        let sw = &points[c..d];
        let se = &points[d..n];

        for p in nw {
            assert!(p.0[0] < mid.0[0], "{p:?} not in NW");
            assert!(p.0[1] < mid.0[1], "{p:?} not in NW");
        }

        for p in ne {
            assert!(p.0[0] >= mid.0[0], "{p:?} not in NE");
            assert!(p.0[1] < mid.0[1], "{p:?} not in NE");
        }

        for p in sw {
            assert!(p.0[0] < mid.0[0], "{p:?} not in SW");
            assert!(p.0[1] >= mid.0[1], "{p:?} not in SW");
        }

        for p in se {
            assert!(p.0[0] >= mid.0[0], "{p:?} not in SE");
            assert!(p.0[1] >= mid.0[1], "{p:?} not in SE");
        }
    }

    #[test]
    fn with_offset() {
        let mut points = [[0, 0], [0, 0], [0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]
            .map(Into::into)
            .to_vec();
        let exp_points = &[[0, 0], [0, 0], [0, 0], [0, 0], [2, 0], [0, 2], [2, 2]].map(Into::into);

        let n = points.len();

        let lo = 3;
        let (mid, mut split_queue) = sort_layer_wrapper(&mut points[lo..], lo);

        assert_eq!(points, exp_points);
        assert_eq!(split_queue, [3, 4, 5, 6]);

        let split_queue = split_queue.make_contiguous();
        let Ok([a, b, c, d]) = TryInto::<[usize; 4]>::try_into(split_queue) else {
            unreachable!()
        };

        assert_eq!(mid, (1, 1).into());

        let nw = &points[a..b];
        let ne = &points[b..c];
        let sw = &points[c..d];
        let se = &points[d..n];

        for p in nw {
            assert!(p.0[0] < mid.0[0], "{p:?} not in NW");
            assert!(p.0[1] < mid.0[1], "{p:?} not in NW");
        }

        for p in ne {
            assert!(p.0[0] >= mid.0[0], "{p:?} not in NE");
            assert!(p.0[1] < mid.0[1], "{p:?} not in NE");
        }

        for p in sw {
            assert!(p.0[0] < mid.0[0], "{p:?} not in SW");
            assert!(p.0[1] >= mid.0[1], "{p:?} not in SW");
        }

        for p in se {
            assert!(p.0[0] >= mid.0[0], "{p:?} not in SE");
            assert!(p.0[1] >= mid.0[1], "{p:?} not in SE");
        }
    }
}

#[cfg(test)]
mod test_tree_d1 {
    use crate::Tree;
    use crate::point::Point;

    const DEPTH: u32 = 1;

    #[test]
    fn ordered_2d() {
        let points = [[0, 0], [2, 0], [0, 2], [2, 2]].map(Into::into).to_vec();

        let exp_points = &[[0, 0], [2, 0], [0, 2], [2, 2]].map(Into::into);

        let tree = Tree::new(points, DEPTH);

        assert_eq!(tree.points, exp_points);
        assert_eq!(tree.splits, [0, 0, 1, 2, 3]);
    }

    #[test]
    fn unordered_2d() {
        let points = [[0, 2], [2, 2], [2, 0], [0, 0]].map(Into::into).to_vec();

        let exp_points = &[[0, 0], [2, 0], [0, 2], [2, 2]].map(Into::into);

        let tree = Tree::new(points, DEPTH);

        assert_eq!(tree.points, exp_points);
        assert_eq!(tree.splits, [0, 0, 1, 2, 3]);
    }

    #[test]
    fn simple_3d() {
        let points: Vec<Point<i32, 3>> = [(0, 0, 0), (2, 2, 2)].map(Into::into).to_vec();

        let exp_points = [(0, 0, 0), (2, 2, 2)].map(Into::into);

        let tree = Tree::new(points, DEPTH);

        assert_eq!(tree.points, exp_points);
        assert_eq!(tree.splits, [0, 0, 1, 1, 1, 1, 1, 1, 1]);
    }
}

#[cfg(test)]
mod test_tree_d2 {
    use crate::Tree;

    const DEPTH: u32 = 2;

    #[test]
    #[rustfmt::skip]
    fn unordered_2d() {
        let exp_points = [
            // mid: (1, 1)

            (0, 0),

            (1, 0),    (3, 0), (2, 0),                                                      // mid: (2, 0)

            (0, 1),                       (0, 3), (0, 2),                                   // mid: (0, 2)

            (1, 1),    (2, 1), (3, 1),    (1, 2), (1, 3),    (3, 3), (2, 3), (3, 2), (2, 2) // mid: (2, 2)
        ].map(Into::into);

        let points = [
            (3, 0), (3, 3), (2, 3), (0, 0),
            (0, 3), (0, 2), (0, 1), (1, 2),
            (1, 3), (2, 1), (3, 1), (2, 0),
            (1, 0), (1, 1), (3, 2), (2, 2),
        ].map(Into::into).to_vec();

        let tree = Tree::new(points, DEPTH);

        assert_eq!(tree.points, exp_points);
        assert_eq!(
            tree.splits, 
            [
                // whole tree split
                0,

                // depth 2 splits
                0, 1, 4, 7,

                // depth 2 splits
                0, 0, 0, 0,    1, 1, 1, 2,    4, 4, 5, 5,    7, 8, 10, 12
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn sorted_larger_2d() {
        let points = [
            (0, 0), (2, 0), (0, 2), (2, 2),

            (4, 0),

            (0, 4), (2, 4), (0, 6), (2, 6),

            (4, 4), (6, 4), (4, 6), (6, 6),
        ].map(Into::into).to_vec();

        let exp_points = &[
            (0, 0), (2, 0), (0, 2), (2, 2),

            (4, 0),

            (0, 4), (0, 6), (2, 4), (2, 6),

            (4, 4), (6, 4), (4, 6), (6, 6),
        ].map(Into::into);

        let tree: Tree<i32, 2> = Tree::new(points, DEPTH);

        assert_eq!(tree.points, exp_points);
        assert_eq!(
            tree.splits,
            [
                // whole tree split
                0,

                // depth 1 splits
                0, 4, 5, 9,

                // depth 2 splits
                0, 1, 2, 3,    4, 4, 5, 5,    5, 5, 5, 7,    9, 9, 9, 9
            ]
        );
    }
}

#[cfg(test)]
mod test_tree_d3 {
    use std::ops::Range;

    use crate::Tree;
    use crate::point::Point;

    const DEPTH: u32 = 3;

    fn range2(xs: Range<i32>, ys: Range<i32>) -> Vec<Point<i32, 2>> {
        xs.flat_map(|x| ys.clone().map(move |y| [x, y].into()))
            .collect()
    }

    #[test]
    #[rustfmt::skip]
    fn unordered_large() {
        let points = range2(0..5, 0..5);

        let exp_points = &[
            // mid: (2, 2)

            (0, 0),            (1, 0),            (0, 1),            (1, 1),                        // mid: (1, 1)

            (2, 0), (3, 0),    (4, 0),            (2, 1),            (3, 1), (4, 1),                // mid: (3, 1)

            (0, 2),            (1, 2),            (0, 3), (0, 4),    (1, 3), (1, 4),                // mid: (1, 3)

            (2, 2),            (3, 2), (4, 2),    (2, 3), (2, 4),    (3, 3), (3, 4), (4, 3), (4, 4) // mid: (3, 3)
        ].map(Into::into);

        let tree = Tree::new(points, DEPTH);

        assert_eq!(tree.points, exp_points);
        assert_eq!(
            tree.splits,
            [
                // whole tree split
                0, 

                // depth = 1 splits
                0, 4, 10, 16, 

                // depth 2 splits
                0, 1, 2, 3,    4, 5, 7, 8,    10, 11, 12, 14,    16, 17, 19, 21, 

                // depth 3 splits
                0, 0, 0, 0,        1, 1, 1, 1,        2, 2, 2, 2,        3, 3, 3, 3, 
                4, 4, 4, 4,        5, 5, 5, 5,        7, 7, 7, 7,        8, 8, 8, 8,
                10, 10, 10, 10,    11, 11, 11, 11,    12, 12, 12, 12,    14, 14, 14, 14,
                16, 16, 16, 16,    17, 17, 17, 17,    19, 19, 19, 19,    21, 21, 21, 21
            ]
        );
    }
}

#[cfg(test)]
mod test_splits {
    use crate::Tree;

    #[test]
    #[rustfmt::skip]
    fn test_splits_2d() {
        // key split pairs
        let tests: &[(&[usize], &[usize])] = &[
            (&[0],          &[0, 1, 1, 1]),
            (&[0, 0],       &[0, 2, 2, 2]),
            (&[1, 2],       &[0, 0, 1, 2]),
            (&[0, 1, 2, 3], &[0, 1, 2, 3]),
            (&[0, 1, 2, 2], &[0, 1, 2, 4]),
            (&[0, 1, 1, 3], &[0, 1, 3, 3]),
            (&[0, 0, 0, 3], &[0, 3, 3, 3]),
        ];

        let mut splits = [0; 4];

        for (keys, exp) in tests {
            Tree::<i8, 2>::compute_splits(keys, &mut splits);

            assert_eq!(splits, *exp);
            splits.fill(0);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn test_splits_3d() {
        // key split pairs
        let tests: &[(&[usize], &[usize])] = &[
            (&[0, 7],       &[0, 1, 1, 1, 1, 1, 1, 1]),
        ];

        let mut splits = [0; 8];

        for (keys, exp) in tests {
            Tree::<i8, 3>::compute_splits(keys, &mut splits);

            assert_eq!(splits, *exp);
            splits.fill(0);
        }
    }
}

#[cfg(test)]
mod proptests {
    use std::collections::VecDeque;

    use proptest::prelude::*;

    use crate::Tree;
    use crate::bound::Bound;
    use crate::point::Point;
    use crate::util;

    type PointType = i8;
    const N: usize = 2;

    fn assert_point_in_orthant<T: Ord + std::fmt::Debug, const N: usize>(
        p: &Point<T, N>,
        mid: &Point<T, N>,
        mut orth: usize,
    ) {
        for i in 0..N {
            if orth & 1 == 1 {
                assert!(
                    p.0[i] >= mid.0[i],
                    "point {p:?} is < {mid:?} midpoint (index {i})",
                );
            } else {
                assert!(
                    p.0[i] < mid.0[i],
                    "point {p:?} is >= {mid:?} midpoint (index {i})",
                );
            }

            orth >>= 1;
        }
    }

    proptest! {
        #[test]
        fn test_sort_layer_num_splits(
            lo in any::<usize>(),
            points in prop::collection::vec(
                prop::array::uniform(any::<PointType>()),
                1..20
            )
        ) {
            let mut points: Vec<Point<PointType, N>> = points.into_iter()
                .map(Point::from)
                .collect();

            let n = points.len();
            let mut keys = vec![0usize; n];
            let mut split_queue = VecDeque::with_capacity(n);

            let mut swaps = Vec::with_capacity(n);
            let mut splits = vec![0; util::num_divs::<N>()];

            let Some(bound) = Bound::from_points(&points) else {
                unreachable!("We always have at least one point")
            };
            let mid = bound.center();

            Tree::sort_layer(mid.clone(), &mut points, &mut keys, &mut swaps, &mut splits);
            split_queue.extend(splits.iter().copied().map(|s| s + lo));

            // We should always have 2^N splits
            assert_eq!(split_queue.len(), util::num_divs::<N>(), "Expected {} splits, found {}", util::num_divs::<N>(), split_queue.len());
        }

        #[test]
        fn test_sort_layer_sorted(
            points in prop::collection::vec(
                prop::array::uniform(any::<PointType>()),
                1..20
            )
        ) {
            let lo = 0;
            let mut points: Vec<Point<PointType, N>> = points.into_iter()
                .map(Point::from)
                .collect();

            let n = points.len();
            let mut keys = vec![0usize; n];
            let mut split_queue = VecDeque::with_capacity(n);

            let mut swaps = Vec::with_capacity(n);
            let mut splits = vec![0; util::num_divs::<N>()];

            let Some(bound) = Bound::from_points(&points) else {
                unreachable!("We always have at least one point")
            };
            let mid = bound.center();

            Tree::sort_layer(mid.clone(), &mut points, &mut keys, &mut swaps, &mut splits);
            split_queue.extend(splits.iter().copied().map(|s| s + lo));

            // For every orthant
            for (i, &lo) in split_queue.iter().enumerate() {
                let hi = split_queue.get(i + 1).copied().unwrap_or(n);

                // Get all the points in that orthant
                let orth = &points[lo..hi];
                for p in orth {
                    assert_point_in_orthant(p, &mid, i);
                }
            }
        }
    }
}
