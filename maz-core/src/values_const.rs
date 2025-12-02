use std::fmt::{Debug, Display, Formatter};
use crate::mapping::{BoardPlayer, Outcome};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValuesAbs<const N: usize> {
    pub value_abs: [f32; N],
    pub moves_left: f32,
}

impl<const N: usize> Default for ValuesAbs<N> {
    fn default() -> Self {
        ValuesAbs {
            value_abs: [0.0; N],
            moves_left: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValuesPov<const N: usize> {
    pub value_pov: [f32; N],
    pub moves_left: f32,
    pub pov: usize, // Player POV as usize
}

impl<const N: usize> ValuesAbs<N> {
    pub fn pov(self, pov: usize) -> ValuesPov<N> {
        let mut pov_values: [f32; N] = self.value_abs;

        pov_values.rotate_left(pov);

        ValuesPov {
            value_pov: pov_values,
            moves_left: self.moves_left,
            pov,
        }
    }

    pub fn val_for_player(&self, player: impl BoardPlayer) -> f32 {
        self.value_abs[player.into()]
    }

    pub fn from_slice(raw: &[f32], moves_left: f32) -> Self {
        debug_assert_eq!(
            raw.len(),
            N,
            "Raw values length {} does not match expected size {}",
            raw.len(),
            N
        );

        let mut value_abs = [0.0; N];
        value_abs.copy_from_slice(raw);

        ValuesAbs {
            value_abs,
            moves_left,
        }
    }
    
    pub fn to_values_vec(&self, is_absolute: bool, pov: usize) -> Vec<f32> {
        if is_absolute {
            self.value_abs.to_vec()
        } else {
            let pov_values = self.pov(pov);
            pov_values.value_pov.to_vec()
        }
    }
}

impl<const N: usize> ValuesPov<N> {
    pub fn abs(self) -> ValuesAbs<N> {
        let mut value_abs = self.value_pov;

        value_abs.rotate_right(self.pov.into());

        ValuesAbs {
            value_abs,
            moves_left: self.moves_left,
        }
    }

    pub fn from_slice(raw: &[f32], moves_left: f32, pov: usize) -> Self {
        debug_assert_eq!(
            raw.len(),
            N,
            "Raw values length {} does not match expected size {}",
            raw.len(),
            N
        );

        let mut value_pov = [0.0; N];
        value_pov.copy_from_slice(raw);

        ValuesPov {
            value_pov,
            moves_left,
            pov,
        }
    }
}

impl<const N: usize> ValuesAbs<N> {
    /// The most important function here: Converts the abstract outcome into absolute values.
    pub fn from_outcome(outcome: Outcome, moves_left: f32, contempt: f32) -> Self {
        match outcome {
            Outcome::WonBy(player) => {
                let mut value_abs = [-1.0; N];
                value_abs[player as usize] = 1.0;
                ValuesAbs {
                    value_abs,
                    moves_left,
                }
            }
            Outcome::AllDraw => ValuesAbs {
                value_abs: [contempt; N],
                moves_left,
            },
            Outcome::PartialDraw(players) => {
                let mut value_abs = [contempt; N];
                for i in 0..N {
                    if (players >> i) & 1 == 0 {
                        value_abs[i] = -1.0;
                    }
                }
                ValuesAbs {
                    value_abs,
                    moves_left,
                }
            }
        }
    }
}

impl<const N: usize> std::ops::Add for ValuesAbs<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Use std::array::from_fn for a clean, idiomatic way to build the new array.
        let value_abs = std::array::from_fn(|i| self.value_abs[i] + rhs.value_abs[i]);
        ValuesAbs {
            value_abs,
            moves_left: self.moves_left + rhs.moves_left,
        }
    }
}

impl<const N: usize> std::ops::Add for ValuesPov<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.pov == rhs.pov,
            "Cannot add ValuesPov with different POVs"
        );
        let value = std::array::from_fn(|i| self.value_pov[i] + rhs.value_pov[i]);
        ValuesPov {
            value_pov: value,
            moves_left: self.moves_left + rhs.moves_left,
            pov: self.pov,
        }
    }
}

impl<const N: usize> std::ops::AddAssign<&Self> for ValuesAbs<N> {
    fn add_assign(&mut self, rhs: &Self) {
        for i in 0..N {
            self.value_abs[i] += rhs.value_abs[i];
        }
        self.moves_left += rhs.moves_left;
    }
}

impl<const N: usize> std::ops::AddAssign<&Self> for ValuesPov<N> {
    fn add_assign(&mut self, rhs: &Self) {
        debug_assert_eq!(
            self.pov, rhs.pov,
            "Cannot add ValuesPov with different POVs"
        );
        for i in 0..N {
            self.value_pov[i] += rhs.value_pov[i];
        }
        self.moves_left += rhs.moves_left;
    }
}

impl<const N: usize> std::ops::Div<f32> for ValuesAbs<N> {
    type Output = ValuesAbs<N>;

    fn div(self, rhs: f32) -> Self::Output {
        let value_abs = std::array::from_fn(|i| self.value_abs[i] / rhs);
        ValuesAbs {
            value_abs,
            moves_left: self.moves_left / rhs,
        }
    }
}

impl<const N: usize> Display for ValuesAbs<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "v:[{}] ml:{}",
            self.value_abs
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join("/"),
            self.moves_left
        )
    }
}

impl<const N: usize> Display for ValuesPov<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "v:[{}] ml:{} pov:{}",
            self.value_pov
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join("/"),
            self.moves_left,
            self.pov,
        )
    }
}

pub trait ValuesAbsLikeImpl {
    fn value_abs(&self) -> &[f32];
}

impl<const N: usize> ValuesAbsLikeImpl for ValuesAbs<N> {
    fn value_abs(&self) -> &[f32] {
        &self.value_abs
    }
}

pub trait ValuesAbsLike: Send + Debug {
    fn value_abs(&self) -> &[f32];

    fn clone_box(&self) -> Box<dyn ValuesAbsLike>;
}

impl<T> ValuesAbsLike for T
where
    T: 'static + Send + Debug + Clone + ValuesAbsLikeImpl,
{
    fn value_abs(&self) -> &[f32] {
        ValuesAbsLikeImpl::value_abs(self)
    }
    fn clone_box(&self) -> Box<dyn ValuesAbsLike> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ValuesAbsLike> {
    fn clone(&self) -> Box<dyn ValuesAbsLike> {
        self.clone_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use game_hex::game_hex::HexPlayer;

    #[test]
    fn test_values_2_abs_from_outcome() {
        let outcome = Outcome::WonBy(0);
        let values = ValuesAbs::<2>::from_outcome(outcome, 5.0, 0.0);
        assert_eq!(values.value_abs, [1.0, -1.0]);
        assert_eq!(values.moves_left, 5.0);

        println!("ValuesAbs from outcome: {values}");
    }

    #[test]
    fn test_values_2_pov() {
        let values = ValuesAbs::<2> {
            value_abs: [1.0, -1.0],
            moves_left: 5.0,
        };

        let player: usize = HexPlayer::P1.into();

        let pov_values = values.pov(player);
        assert_eq!(pov_values.value_pov, [1.0, -1.0]);
        assert_eq!(pov_values.moves_left, 5.0);

        println!("ValuesPov: {pov_values} from ValuesAbs: {values}");

        let un_pov_values = pov_values.abs();
        assert_eq!(un_pov_values.value_abs, [1.0, -1.0]);
        assert_eq!(un_pov_values.moves_left, 5.0);

        let pov_values = values.pov(HexPlayer::P2.into());
        assert_eq!(pov_values.value_pov, [-1.0, 1.0]);
        assert_eq!(pov_values.moves_left, 5.0);

        println!("ValuesPov {pov_values} from ValuesAbs: {values}");

        let un_pov_values = pov_values.abs();
        assert_eq!(un_pov_values.value_abs, [1.0, -1.0]);
        assert_eq!(un_pov_values.moves_left, 5.0);
    }

    #[test]
    fn test_value_3_abs_from_outcome() {
        let outcome = Outcome::PartialDraw(0b011);
        let values = ValuesAbs::<3>::from_outcome(outcome, 4.0, 0.0);
        assert_eq!(values.value_abs, [0.0, 0.0, -1.0]);
        assert_eq!(values.moves_left, 4.0);

        println!("ValuesAbs from outcome: {values}");

        let pov_values = values.pov(2);
        assert_eq!(pov_values.value_pov, [-1.0, 0.0, 0.0]);
        assert_eq!(pov_values.moves_left, 4.0);
        assert_eq!(pov_values.abs(), values);

        println!("ValuesPov: {pov_values} from ValuesAbs: {values}");

        let pov_values = values.pov(1);
        assert_eq!(pov_values.value_pov, [0.0, -1.0, 0.0]);
        assert_eq!(pov_values.moves_left, 4.0);
        assert_eq!(pov_values.abs(), values);

        println!("ValuesPov: {pov_values} from ValuesAbs: {values}");

        let pov_values = values.pov(0);
        assert_eq!(pov_values.value_pov, [0.0, 0.0, -1.0]);
        assert_eq!(pov_values.moves_left, 4.0);
        assert_eq!(pov_values.abs(), values);

        println!("ValuesPov: {pov_values} from ValuesAbs: {values}");
    }

    #[test]
    fn test_outcome_partial_draw() {
        let partial_drawn_indices = [0, 2];

        let mut stalemate_mask = 0u8;

        for &index in &partial_drawn_indices {
            stalemate_mask |= 1 << index as u8;
        }
        assert_eq!(stalemate_mask, 0b_0000_0101);

        let outcome = Outcome::PartialDraw(stalemate_mask);
        let values = ValuesAbs::<3>::from_outcome(outcome, 4.0, 0.0);
        assert_eq!(values.value_abs, [0.25, -1.0, 0.25]);
        assert_eq!(values.moves_left, 4.0);

        println!("ValuesAbs from outcome: {values}");
    }

    #[test]
    fn do_i_understand_rotate () {
        let mut arr = [1, 2, 0];

        arr.rotate_right(1);

        println!("arr: {:?}", arr);
    }
}