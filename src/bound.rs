use crate::traits;
use crate::util;
use crate::Point;

#[derive(Clone)]
pub struct Bound<T, const N: usize> {
    pub min: Point<T, N>,
    pub max: Point<T, N>,
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
