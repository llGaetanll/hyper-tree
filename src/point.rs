use std::fmt::Debug;

#[derive(Clone, PartialEq, Eq)]
pub struct Point<T, const N: usize>(pub [T; N]);

impl<T: Debug, const N: usize> Debug for Point<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tuple = f.debug_tuple("");

        for element in &self.0 {
            tuple.field(element);
        }

        tuple.finish()
    }
}

impl<T, const N: usize> From<[T; N]> for Point<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T> From<(T, T)> for Point<T, 2> {
    fn from(value: (T, T)) -> Self {
        Self([value.0, value.1])
    }
}

impl<T> From<(T, T, T)> for Point<T, 3> {
    fn from(value: (T, T, T)) -> Self {
        Self([value.0, value.1, value.2])
    }
}

impl<T> From<(T, T, T, T)> for Point<T, 4> {
    fn from(value: (T, T, T, T)) -> Self {
        Self([value.0, value.1, value.2, value.3])
    }
}

impl<T> From<(T, T, T, T, T)> for Point<T, 5> {
    fn from(value: (T, T, T, T, T)) -> Self {
        Self([value.0, value.1, value.2, value.3, value.4])
    }
}

impl<T> From<(T, T, T, T, T, T)> for Point<T, 6> {
    fn from(value: (T, T, T, T, T, T)) -> Self {
        Self([value.0, value.1, value.2, value.3, value.4, value.5])
    }
}

impl<T> From<(T, T, T, T, T, T, T)> for Point<T, 7> {
    fn from(value: (T, T, T, T, T, T, T)) -> Self {
        Self([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6,
        ])
    }
}

impl<T> From<(T, T, T, T, T, T, T, T)> for Point<T, 8> {
    fn from(value: (T, T, T, T, T, T, T, T)) -> Self {
        Self([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6, value.7,
        ])
    }
}

impl<T> From<(T, T, T, T, T, T, T, T, T)> for Point<T, 9> {
    fn from(value: (T, T, T, T, T, T, T, T, T)) -> Self {
        Self([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6, value.7, value.8,
        ])
    }
}

impl<T> From<(T, T, T, T, T, T, T, T, T, T)> for Point<T, 10> {
    fn from(value: (T, T, T, T, T, T, T, T, T, T)) -> Self {
        Self([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6, value.7, value.8,
            value.9,
        ])
    }
}

impl<T> From<(T, T, T, T, T, T, T, T, T, T, T)> for Point<T, 11> {
    fn from(value: (T, T, T, T, T, T, T, T, T, T, T)) -> Self {
        Self([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6, value.7, value.8,
            value.9, value.10,
        ])
    }
}

impl<T> From<(T, T, T, T, T, T, T, T, T, T, T, T)> for Point<T, 12> {
    fn from(value: (T, T, T, T, T, T, T, T, T, T, T, T)) -> Self {
        Self([
            value.0, value.1, value.2, value.3, value.4, value.5, value.6, value.7, value.8,
            value.9, value.10, value.11,
        ])
    }
}
