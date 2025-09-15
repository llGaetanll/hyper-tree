use crate::traits;
use crate::util;
use crate::Point;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bound<T, const N: usize> {
    pub min: Point<T, N>,
    pub max: Point<T, N>,
}

impl<T, const N: usize> Bound<T, N> {
    pub fn new(p1: Point<T, N>, p2: Point<T, N>) -> Self
    where
        T: ::core::cmp::Ord + ::core::marker::Copy,
    {
        let Some(bound) = Self::from_points(&[p1, p2]) else {
            unreachable!()
        };

        bound
    }

    pub fn from_points(points: &[Point<T, N>]) -> Option<Self>
    where
        T: ::core::cmp::Ord + ::core::marker::Copy,
    {
        match points {
            [] | [_] => None,
            [point, rest @ ..] => {
                let mut bound = Bound {
                    min: point.clone(),
                    max: point.clone(),
                };

                for point in rest {
                    bound |= point;
                }

                Some(bound)
            }
        }
    }

    pub fn center(&self) -> Point<T, N>
    where
        T: traits::Mean,
    {
        let Ok(point) = self
            .min
            .0
            .iter()
            .zip(&self.max.0)
            .map(|(&t1, &t2)| t1.mean(t2))
            .collect::<Vec<_>>()
            .try_into()
        else {
            unreachable!()
        };

        Point(point)
    }

    pub fn split(&self) -> Option<impl Iterator<Item = Self>>
    where
        T: traits::Mean + traits::Epsilon + ::core::ops::Sub<Output = T> + ::core::cmp::Ord,
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
    mid: Point<T, N>,
    max: Point<T, N>,
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
        if self.i == util::num_divs::<N>() {
            return None;
        }

        let mut p1 = self.min.clone();

        let mut j = self.i;
        for k in 0..N {
            p1.0[k] = if j & 1 == 1 {
                self.max.0[k]
            } else {
                self.min.0[k]
            };

            j >>= 1;
        }

        self.i += 1;

        Some(Bound::new(p1, self.mid.clone()))
    }
}

#[cfg(test)]
mod tests {
    use crate::bound::Bound;
    use crate::bound::BoundIterator;

    #[test]
    fn test_bound_iterator() {
        let min = (0, 0).into();
        let mid = (1, 1).into();
        let max = (2, 2).into();

        let mut iter = BoundIterator::new(min, mid, max);

        assert_eq!(iter.next(), Some(Bound::new((0, 0).into(), (1, 1).into())));
        assert_eq!(iter.next(), Some(Bound::new((2, 0).into(), (1, 1).into())));
        assert_eq!(iter.next(), Some(Bound::new((0, 2).into(), (1, 1).into())));
        assert_eq!(iter.next(), Some(Bound::new((2, 2).into(), (1, 1).into())));

        assert_eq!(iter.next(), None);
    }
}
