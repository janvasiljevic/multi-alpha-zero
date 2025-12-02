use std::cmp::Ordering;
use std::iter::Zip;

pub fn zip_eq_exact<L, R, LI, RI>(left: L, right: R) -> Zip<LI, RI>
where
    L: IntoIterator<IntoIter = LI>,
    R: IntoIterator<IntoIter = RI>,
    LI: ExactSizeIterator,
    RI: ExactSizeIterator,
{
    let left = left.into_iter();
    let right = right.into_iter();
    assert_eq!(
        left.len(),
        right.len(),
        "Both iterators must have the same length",
    );
    left.zip(right)
}

/// Similar to [rand::seq::IteratorRandom::choose] but will only pick items with the maximum key.
/// Equivalent to first finding the max key, then filtering items matching that key and then choosing a random element,
/// but implemented in a single pass over the iterator.
pub fn choose_max_by_key<T, I: IntoIterator<Item = T>, K: Ord, F: FnMut(&T) -> K>(
    iter: I,
    mut key: F,
    rng: &mut impl rand::Rng,
) -> Option<T> {
    let mut iter = iter.into_iter();

    let mut curr = iter.next()?;
    let mut max_key = key(&curr);
    let mut i = 1;

    for next in iter {
        let next_key = key(&next);
        match next_key.cmp(&max_key) {
            Ordering::Less => continue,
            Ordering::Equal => {
                i += 1;
                if rng.random_range(0..i) == 0 {
                    curr = next;
                }
            }
            Ordering::Greater => {
                i = 1;
                curr = next;
                max_key = next_key;
            }
        }
    }

    Some(curr)
}
