#![deny(unsafe_code)]

mod traits;
mod util;

#[derive(Clone)]
pub struct Point<T, const N: usize>(pub [T; N]);

#[derive(Clone)]
struct Bound<T, const N: usize> {
    pub min: Point<T, N>,
    pub max: Point<T, N>,
}

impl<'a, T, const N: usize> ::core::ops::BitOrAssign<&'a Point<T, N>> for Bound<T, N>
where
    T: ::core::cmp::Ord + ::core::marker::Copy,
{
    fn bitor_assign(&mut self, rhs: &'a Point<T, N>) {
        for (i, &t) in rhs.0.iter().enumerate() {
            self.min.0[i] = self.min.0[i].min(t);
            self.max.0[i] = self.max.0[i].max(t);
        }
    }
}

struct BoundIterator<T, const N: usize> {
    min: Point<T, N>,
    max: Point<T, N>,
    mid: Point<T, N>,
    i: usize,
}

impl<T, const N: usize> BoundIterator<T, N> {
    pub fn new(min: Point<T, N>, mid: Point<T, N>, max: Point<T, N>) -> Self {
        Self {
            min,
            mid,
            max,
            i: 0,
        }
    }
}

impl<T, const N: usize> Iterator for BoundIterator<T, N>
where
    T: ::core::marker::Copy + ::core::cmp::Ord,
{
    type Item = Bound<T, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if util::num_divs::<N>() > self.i {
            return None;
        }

        let mut p1 = self.min.clone();

        let mut j = self.i;
        for k in 0..N {
            p1.0[k] = if j & 1 == 1 {
                self.min.0[k]
            } else {
                self.max.0[k]
            };

            j >>= 1;
        }

        self.i += 1;

        Some(Bound::new(p1, self.mid.clone()))
    }
}

impl<T, const N: usize> Bound<T, N> {
    pub fn new(p1: Point<T, N>, p2: Point<T, N>) -> Self
    where
        T: ::core::cmp::Ord + ::core::marker::Copy,
    {
        Self::from_points(&[p1, p2]).unwrap() // won't fail
    }

    pub fn from_points(points: &[Point<T, N>]) -> Option<Self>
    where
        T: ::core::cmp::Ord + ::core::marker::Copy,
    {
        if let [first, points @ ..] = points {
            let mut bound = Bound {
                min: first.clone(),
                max: first.clone(),
            };

            for point in points {
                bound |= point;
            }

            Some(bound)
        } else {
            None
        }
    }

    pub fn center(&self) -> Point<T, N>
    where
        T: traits::Mean + ::core::fmt::Debug,
    {
        // Note: `Debug` is necessary because of `unwrap` (which will never panic). We could relax
        // this bound by using `unwrap_unchecked` but I deemed it more important to not use unsafe
        // code, since all reasonable `T` is `Debug` anyway.
        Point(
            self.min
                .0
                .iter()
                .zip(&self.max.0)
                .map(|(&t1, &t2)| t1.mean(t2))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }

    // TODO: Instead of `Option`, return empty iter?
    pub fn split(&self) -> Option<impl Iterator<Item = Self>>
    where
        T: traits::Mean
            + traits::Epsilon
            + ::core::ops::Sub<Output = T>
            + ::core::cmp::Ord
            + ::core::fmt::Debug,
    {
        if self.minimal() {
            None
        } else {
            let mid = self.center();

            Some(BoundIterator::new(self.min.clone(), mid, self.max.clone()))
        }
    }

    fn minimal(&self) -> bool
    where
        T: traits::Epsilon + ::core::ops::Sub<Output = T> + ::core::cmp::Ord + Copy,
    {
        self.max
            .0
            .iter()
            .zip(&self.min.0)
            .any(|(&t1, &t2)| t1.sub(t2) <= T::EPSILON)
    }
}

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
