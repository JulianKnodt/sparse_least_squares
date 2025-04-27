#[cfg(not(feature = "f64"))]
pub type F = f32;

#[cfg(feature = "f64")]
pub type F = f64;

/// Constructor for a given sparsity pattern
mod builder;
use builder::{BuilderInsertError, SparsityPattern, SparsityPatternBuilder};

/// Compressed Sparse Matrix.
mod cs;

/// Compressed Sparse Column Matrix
pub mod csc;

//// Sparse LU algorithm
pub mod sparse_lu;

/*
#[derive(Debug, PartialEq, Eq)]
pub enum SparseEntryMut<'a, T> {
    /// The entry is a mutable reference to an explicitly stored element.
    ///
    /// Note that the naming here is a misnomer: The element can still be zero, even though it
    /// is explicitly stored (a so-called "explicit zero").
    NonZero(&'a mut T),
    /// The entry is implicitly zero i.e. it is not explicitly stored.
    Zero,
}
*/
