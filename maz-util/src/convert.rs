use std::collections::HashMap;

pub fn hashmap_to_histogram<V>(map: &HashMap<u32, V>) -> Vec<f32>
where
    V: Into<f32> + Copy,
{
    let max = map.keys().max().copied().unwrap_or(0);

    let mut histogram = vec![0.0; (max + 1) as usize];

    for (&key, &value) in map.iter() {
        histogram[key as usize] = value.into();
    }

    histogram
}
