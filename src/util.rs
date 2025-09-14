// TODO: while this could panic in theory, `N` is
// almost always `2` or `3`, very rarely `4`.
pub const fn num_divs<const N: usize>() -> usize {
    2u32.pow(N as u32) as usize
}

/// Takes `&mut [T]` and a predicate `P: FnMut(&T) -> bool` and partitions the list according to
/// the predicate. Swaps elements of the list such that all elements satisfying `P` appear before
/// any element not satisfying `P`.
///
/// The index `i` returned by the function always points at the first element for which `P` is false.
/// Note that if `P` is trivial, then `i = |list|` points outside the list.
pub fn partition_in_place<T, P>(list: &mut [T], mut predicate: P) -> usize
where
    P: FnMut(&T) -> bool,
{
    if list.is_empty() {
        return 0;
    }

    let (mut lo, mut hi) = (0, list.len() - 1);

    while lo < hi {
        if predicate(&list[lo]) {
            lo += 1;
            continue;
        }

        if !predicate(&list[hi]) {
            hi -= 1;
            continue;
        }

        list.swap(lo, hi);
        lo += 1;
        hi -= 1;
    }

    if predicate(&list[lo]) {
        lo + 1
    } else {
        lo
    }
}

#[cfg(test)]
mod partition_tests {
    fn test_list(input: &mut [i32], output: &[i32], exp: usize) {
        let idx = super::partition_in_place(input, |&x| x < 10);

        assert_eq!(input, output);
        assert_eq!(idx, exp);
    }

    #[test]
    fn empty() {
        test_list(&mut [], &[], 0);
    }

    #[test]
    #[rustfmt::skip]
    fn one() {
        test_list(&mut [0],  &[0],  1);
        test_list(&mut [10], &[10], 0);
    }

    #[test]
    #[rustfmt::skip]
    fn two() {
        test_list(&mut [0, 10],  &[0, 10],  1);
        test_list(&mut [0, 1],   &[0, 1],   2);
        test_list(&mut [10, 11], &[10, 11], 0);
        test_list(&mut [10, 0],  &[0, 10],  1);
        test_list(&mut [1, 0],   &[1, 0],   2);
        test_list(&mut [11, 10], &[11, 10], 0);
    }

    #[test]
    #[rustfmt::skip]
    fn three() {
        test_list(&mut [0, 1, 2],   &[0, 1, 2],   3);
        test_list(&mut [0, 1, 10],  &[0, 1, 10],  2);
        test_list(&mut [0, 10, 11], &[0, 10, 11], 1);
        test_list(&mut [10, 11, 12], &[10, 11, 12], 0);
    }

    #[test]
    #[rustfmt::skip]
    fn misc() {
        test_list(
            &mut [0, 1, 10, 11],
            &    [0, 1, 10, 11], 
            2,
        );
        test_list(
            &mut [11, 12, 13, 14, 1, 2, 3, 4],
            &    [4, 3, 2, 1, 14, 13, 12, 11],
            4,
        );
    }
}
