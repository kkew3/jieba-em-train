//! F64 buffer.

use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

const S: usize = 128;

pub enum F64Buf {
    Stack([f64; S]),
    Heap(Vec<f64>),
}

impl F64Buf {
    pub fn zeros(capacity: usize) -> Self {
        if capacity <= S {
            Self::Stack([0f64; S])
        } else {
            Self::Heap(vec![0f64; capacity])
        }
    }
}

impl AsRef<[f64]> for F64Buf {
    fn as_ref(&self) -> &[f64] {
        match self {
            Self::Stack(v) => v,
            Self::Heap(v) => v,
        }
    }
}

impl AsMut<[f64]> for F64Buf {
    fn as_mut(&mut self) -> &mut [f64] {
        match self {
            Self::Stack(v) => v,
            Self::Heap(v) => v,
        }
    }
}

impl Index<usize> for F64Buf {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl IndexMut<usize> for F64Buf {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

impl Index<RangeFull> for F64Buf {
    type Output = [f64];

    fn index(&self, index: RangeFull) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl IndexMut<RangeFull> for F64Buf {
    fn index_mut(&mut self, index: RangeFull) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

macro_rules! impl_index {
    ($range:tt) => {
        impl Index<$range<usize>> for F64Buf {
            type Output = [f64];

            fn index(&self, index: $range<usize>) -> &Self::Output {
                &self.as_ref()[index]
            }
        }

        impl IndexMut<$range<usize>> for F64Buf {
            fn index_mut(&mut self, index: $range<usize>) -> &mut Self::Output {
                &mut self.as_mut()[index]
            }
        }
    };
}

impl_index!(Range);
impl_index!(RangeInclusive);
impl_index!(RangeFrom);
impl_index!(RangeTo);
impl_index!(RangeToInclusive);
