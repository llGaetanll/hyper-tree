pub trait Mean: Copy {
    fn mean(self, other: Self) -> Self;
}

pub trait Epsilon {
    /// The smallest non-zero value of the type
    const EPSILON: Self;
}

impl Mean for u8 {
    fn mean(self, other: Self) -> Self {
        (((self as u32) + (other as u32)) / 2u32) as Self
    }
}

impl Mean for u16 {
    fn mean(self, other: Self) -> Self {
        (((self as u32) + (other as u32)) / 2u32) as Self
    }
}

impl Mean for u32 {
    fn mean(self, other: Self) -> Self {
        (((self as u64) + (other as u64)) / 2u64) as Self
    }
}

impl Mean for u64 {
    fn mean(self, other: Self) -> Self {
        (((self as u128) + (other as u128)) / 2u128) as Self
    }
}

impl Mean for i8 {
    fn mean(self, other: Self) -> Self {
        (((self as i32) + (other as i32)) / 2i32) as Self
    }
}

impl Mean for i16 {
    fn mean(self, other: Self) -> Self {
        (((self as i32) + (other as i32)) / 2i32) as Self
    }
}

impl Mean for i32 {
    fn mean(self, other: Self) -> Self {
        (((self as i64) + (other as i64)) / 2i64) as Self
    }
}

impl Mean for i64 {
    fn mean(self, other: Self) -> Self {
        (((self as i128) + (other as i128)) / 2i128) as Self
    }
}

macro_rules! impl_epsilon_int {
    ($t:ty) => {
        impl Epsilon for $t {
            const EPSILON: Self = 1 as $t;
        }
    };

    ($t:ty, $($ts:ty),+) => {
        impl_epsilon_int!($t);

        impl_epsilon_int!($($ts),+);
    };
}

#[rustfmt::skip]
impl_epsilon_int!(
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128
);

impl Epsilon for f32 {
    const EPSILON: Self = Self::EPSILON;
}

impl Epsilon for f64 {
    const EPSILON: Self = Self::EPSILON;
}
