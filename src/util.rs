// TODO: while this could panic in theory, `N` is
// almost always `2` or `3`, very rarely `4`.
pub const fn num_divs<const N: usize>() -> usize {
    2u32.pow(N as u32) as usize
}

/// Assumes that `swaps = [0, 1, .., n]` is already initialized
pub fn argsort<T: Ord>(values: &[T], swaps: &mut [usize]) {
    assert_eq!(values.len(), swaps.len());

    swaps.sort_by_key(|&i| &values[i]);
}

const FLAG: usize = isize::MIN as usize;

pub fn sort_by_argsort<T>(data: &mut [T], swaps: &mut [usize]) {
    assert_eq!(data.len(), swaps.len());
    assert!(data.len() <= isize::MAX as usize);

    for swap in swaps.iter() {
        debug_assert!(!is_tagged(*swap));
    }

    let n = data.len();
    for i in 0..n {
        if is_tagged(swaps[i]) {
            continue;
        }

        let mut cursor = i;
        loop {
            let next = untag(swaps[cursor]);
            tag(&mut swaps[cursor]);
            if next == i {
                break;
            } else {
                data.swap(cursor, next);
                cursor = next;
            }
        }
    }

    for swap in swaps {
        *swap &= !FLAG;
    }
}

fn tag(swap: &mut usize) {
    *swap |= FLAG
}

fn untag(swap: usize) -> usize {
    swap & !FLAG
}

fn is_tagged(swap: usize) -> bool {
    swap & FLAG == FLAG
}

#[cfg(test)]
mod sort_tests {
    #[test]
    fn test_argsort_identity() {
        let lst = [0, 1, 2, 3];

        let mut swaps: Vec<usize> = (0..4).collect();
        super::argsort(&lst, &mut swaps);

        assert_eq!(swaps, lst);
    }

    #[test]
    fn sort_1() {
        let mut lst = [1, 2, 4, 3, 5];

        let mut swaps: Vec<usize> = (0..5).collect();
        super::argsort(&lst, &mut swaps);
        super::sort_by_argsort(&mut lst, &mut swaps);

        assert_eq!(swaps, [0, 1, 3, 2, 4]);
        assert_eq!(lst, [1, 2, 3, 4, 5])
    }

    #[test]
    fn sort_2() {
        let mut lst = [1, 3, 2, 0];
        let mut swaps: Vec<usize> = (0..4).collect();

        super::argsort(&lst, &mut swaps);
        super::sort_by_argsort(&mut lst, &mut swaps);

        assert_eq!(lst, [0, 1, 2, 3])
    }
}
