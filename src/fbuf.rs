//! F64 buffer.

use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

pub enum F64Buf<'a> {
    Stack(&'a mut [f64]),
    Heap(Vec<f64>),
}

impl<'a> F64Buf<'a> {
    pub fn zeros(buf: &'a mut [f64], capacity: usize) -> Self {
        if capacity <= buf.len() {
            buf[..capacity].fill(0.0);
            Self::Stack(&mut buf[..capacity])
        } else {
            Self::Heap(vec![0f64; capacity])
        }
    }
}

impl<'a> AsRef<[f64]> for F64Buf<'a> {
    fn as_ref(&self) -> &[f64] {
        match self {
            Self::Stack(v) => v,
            Self::Heap(v) => v,
        }
    }
}

impl<'a> AsMut<[f64]> for F64Buf<'a> {
    fn as_mut(&mut self) -> &mut [f64] {
        match self {
            Self::Stack(v) => v,
            Self::Heap(v) => v,
        }
    }
}

impl<'a> Index<usize> for F64Buf<'a> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<'a> IndexMut<usize> for F64Buf<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

impl<'a> Index<RangeFull> for F64Buf<'a> {
    type Output = [f64];

    fn index(&self, index: RangeFull) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<'a> IndexMut<RangeFull> for F64Buf<'a> {
    fn index_mut(&mut self, index: RangeFull) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

macro_rules! impl_index {
    ($range:tt) => {
        impl<'a> Index<$range<usize>> for F64Buf<'a> {
            type Output = [f64];

            fn index(&self, index: $range<usize>) -> &Self::Output {
                &self.as_ref()[index]
            }
        }

        impl<'a> IndexMut<$range<usize>> for F64Buf<'a> {
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
