#[cfg(not(feature = "f64"))]
pub type F = f32;

#[cfg(feature = "f64")]
pub type F = f64;

/// Constructor for a given sparsity pattern
mod builder;
pub use builder::{BuilderInsertError, SparsityPattern, SparsityPatternBuilder};

/// Compressed Sparse Matrix.
mod cs;

/// Compressed Sparse Column Matrix
pub mod csc;
pub use csc::Csc;

/// Sparse LU algorithm
mod sparse_lu;
pub use sparse_lu::LeftLookingLUFactorization;

// TODO implement gauss seidel?
