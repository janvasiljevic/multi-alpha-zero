use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AxialCoord {
    pub q: i32,
    pub r: i32,
}

impl Display for AxialCoord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(q: {}, r: {})", self.q, self.r)
    }
}

impl AxialCoord {
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Returns the S coordinate in cube coordinates (q + r + s = 0)
    /// Not the same system as in Chess
    pub fn s(&self) -> i32 {
        -self.q - self.r
    }

    pub fn neighbors(&self) -> [AxialCoord; 6] {
        [
            AxialCoord::new(self.q + 1, self.r),
            AxialCoord::new(self.q - 1, self.r),
            AxialCoord::new(self.q, self.r + 1),
            AxialCoord::new(self.q, self.r - 1),
            AxialCoord::new(self.q + 1, self.r - 1),
            AxialCoord::new(self.q - 1, self.r + 1),
        ]
    }
}
