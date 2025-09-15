#![deny(unsafe_code)]

mod bound;
mod point;
mod traits;
mod util;

use core::cmp::Ord;
use core::marker::Copy;
use core::ops::Range;
use core::ops::Sub;

use bound::Bound;

use crate::point::Point;
use crate::traits::Epsilon;
use crate::traits::Mean;
use crate::util::num_divs;

pub struct Tree<T, const N: usize> {
    points: Vec<Point<T, N>>,
    splits: Vec<usize>,
    depth: u32,
}

pub type QuadTree<T> = Tree<T, 2>;
pub type OcTree<T> = Tree<T, 3>;

impl<T, const N: usize> Tree<T, N> {
    fn uninit(points: Vec<Point<T, N>>, depth: u32) -> Self
    where
        T: Ord + Copy,
    {
        // n = (2^N)^d
        let n = util::num_divs::<N>().pow(depth);
        let splits = vec![0; n];

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

        // Build the tree if more than one point is provided
        if let Some(bound) = Bound::from_points(&tree.points) {
            tree.build(bound);
        }

        tree
    }

    fn build(&mut self, bound: Bound<T, N>)
    where
        T: Mean + Epsilon + Sub<Output = T> + Ord,
    {
        let point_range = 0..self.points.len();
        let split_range = 0..self.splits.len();

        if point_range.len() < 2 {
            return;
        }

        self.partition(bound, self.depth, point_range, split_range);
    }

    fn partition(
        &mut self,
        bound: Bound<T, N>,
        d: u32,
        point_range: Range<usize>,
        split_range: Range<usize>,
    ) where
        T: Mean + Epsilon + Sub<Output = T> + Ord,
    {
        if d == 0 {
            return;
        }

        let lo = split_range.start;

        let mid = bound.center();
        self.partition_level(mid, 0, point_range, split_range);

        let Some(orthants) = bound.split() else {
            return;
        };

        // Stride into the splits array.
        //
        // If the depth is 1, splits of interest are next to each other
        // If the depth is 2, between each split of interest is 2^N splits for the layer below us
        // If the depth is 3, each of *those* splits contains 2^N splits between *them*.
        // etc...
        let stride = num_divs::<N>().pow(d - 1);

        for (i, ort) in orthants.enumerate() {
            let s_lo = lo + stride * i;
            let s_hi = lo + stride * (i + 1);

            let p_lo = self.splits[s_lo];
            let p_hi = self.splits.get(s_hi).copied().unwrap_or(self.points.len());

            self.partition(ort, d - 1, p_lo..p_hi, s_lo..s_hi);
        }
    }

    fn partition_level(
        &mut self,
        mid: Point<T, N>,
        i: usize,
        point_range: Range<usize>,
        split_range: Range<usize>,
    ) where
        T: Clone + Ord,
    {
        let (p_lo, p_hi) = (point_range.start, point_range.end);
        let (s_lo, s_hi) = (split_range.start, split_range.end);

        if p_hi - p_lo < 2 || i == N {
            return;
        }

        let p_mid =
            util::partition_in_place(&mut self.points[p_lo..p_hi], |p| p.0[i] < mid.0[i]) + p_lo;

        let s_mid = s_lo + ((s_hi - s_lo) / 2);
        self.splits[s_mid..s_hi].fill(p_mid);

        self.partition_level(mid.clone(), i + 1, p_lo..p_mid, s_lo..s_mid);
        self.partition_level(mid, i + 1, p_mid..p_hi, s_mid..s_hi);
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::point::Point;
    use crate::Tree;

    fn range2(xs: Range<i32>, ys: Range<i32>) -> Vec<Point<i32, 2>> {
        xs.flat_map(|x| ys.clone().map(move |y| [x, y].into()))
            .collect()
    }

    #[test]
    fn ordered_d1() {
        let points = [[0, 0], [0, 2], [2, 0], [2, 2]].map(Into::into).to_vec();
        let depth = 1;

        let exp_points = &[[0, 0], [0, 2], [2, 0], [2, 2]].map(Into::into);

        let tree = Tree::new(points, depth);

        assert_eq!(tree.points, exp_points);
        assert_eq!(tree.splits, [0, 1, 2, 3]);
    }

    #[test]
    fn unordered_d1() {
        let points = [[0, 2], [2, 2], [2, 0], [0, 0]].map(Into::into).to_vec();
        let depth = 1;

        let exp_points = &[[0, 0], [0, 2], [2, 0], [2, 2]].map(Into::into);

        let tree = Tree::new(points, depth);

        assert_eq!(tree.points, exp_points);
        assert_eq!(tree.splits, [0, 1, 2, 3]);
    }

    #[test]
    #[rustfmt::skip]
    fn unordered_d2() {
        let depth = 2;
        let points = [
            (2, 0), (1, 2), (2, 3), (0, 1),
            (3, 3), (3, 1), (0, 0), (0, 3),
            (1, 0), (3, 2), (2, 2), (1, 3),
            (0, 2), (2, 1), (3, 0), (1, 1)
        ].map(Into::into).to_vec();

        let exp_points = &[
            (0, 0), (0, 3), (0, 2), (0, 1),
            (3, 0), (2, 0), (1, 0), (1, 1),
            (1, 2), (1, 3), (2, 1), (3, 1),
            (3, 2), (2, 2), (3, 3), (2, 3)
        ].map(Into::into);

        let tree = Tree::new(points, depth);

        assert_eq!(tree.points, exp_points);
    }

    #[test]
    #[rustfmt::skip]
    fn unordered_d3() {
        let depth = 3;
        let points = range2(0..5, 0..5);

        let exp_points = &[
            (0, 0), (0, 1), (1, 0), (1, 1), (0, 4),
            (0, 3), (0, 2), (1, 2), (1, 3), (1, 4),
            (2, 0), (2, 1), (4, 1), (4, 0), (3, 1),
            (3, 0), (2, 2), (3, 2), (3, 3), (2, 3),
            (3, 4), (2, 4), (4, 2), (4, 3), (4, 4)
        ].map(Into::into);

        let tree = Tree::new(points, depth);

        assert_eq!(tree.points, exp_points);
    }
}
