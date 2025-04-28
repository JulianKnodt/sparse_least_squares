use super::F;

use super::cs::{CsBuilder, CsMatrix};
use super::{BuilderInsertError, SparsityPattern};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Csc<T>(CsMatrix<T>);

impl<T> Csc<T> {
    pub fn ncols(&self) -> usize {
        self.0.pattern.major_dim()
    }
    pub fn nrows(&self) -> usize {
        self.0.pattern.minor_dim
    }
    pub fn col_iter(
        &self,
        i: usize,
    ) -> impl Iterator<Item = (usize, &T)> + DoubleEndedIterator + '_ {
        self.0.lane_iter(i)
    }
    pub fn pattern(&self) -> &SparsityPattern {
        &self.0.pattern
    }

    pub fn col(&self, i: usize) -> (&[T], &[usize]) {
        self.0.lane(i)
    }

    pub fn values(&self) -> &[T] {
        self.0.values()
    }

    pub fn values_mut(&mut self) -> &mut [T] {
        self.0.values_mut()
    }
    /// Constructs a CSC matrix from a set of triples. Fails if there are duplicate entries.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        t: &mut [([usize; 2], T)],
    ) -> Result<Self, BuilderInsertError>
    where
        T: Copy,
    {
        let mut builder = CscBuilder::new(rows, cols);
        t.sort_unstable_by_key(|a| a.0);
        for &([x, y], v) in t.iter() {
            builder.insert(y, x, v)?;
        }
        Ok(builder.build())
    }

    pub fn from_btreemap(
        rows: usize,
        cols: usize,
        map: &std::collections::BTreeMap<[usize; 2], T>,
    ) -> Result<Self, BuilderInsertError>
    where
        T: Copy,
    {
        let mut builder = CscBuilder::new(rows, cols);
        for (&[x, y], &v) in map.iter() {
            builder.insert(y, x, v)?;
        }
        Ok(builder.build())
    }
}

impl Csc<F> {
    pub fn identity(n: usize) -> Self {
        Csc(CsMatrix::identity(n))
    }
    /// Solves a lower triangular system, `self` is a matrix of NxN, and `b` is a column vector of size N
    /// Assuming that b is dense.
    pub fn dense_lower_triangular_solve(&self, b: &[F], out: &mut [F], unit_diagonal: bool) {
        self.dense_lower_triangular_solve_arr(
            unsafe { std::mem::transmute::<_, &[[F; 1]]>(b) },
            unsafe { std::mem::transmute(out) },
            unit_diagonal,
        );
    }
    /// Solves a lower triangular system, `self` is a matrix of NxN, and `b` is a column vector of size N
    /// Assuming that b is dense.
    pub fn dense_lower_triangular_solve_arr<const N: usize>(
        &self,
        b: &[[F; N]],
        out: &mut [[F; N]],
        unit_diagonal: bool,
    ) {
        assert_eq!(self.nrows(), self.ncols());
        assert_eq!(self.ncols(), b.len());
        assert_eq!(out.len(), b.len());
        out.copy_from_slice(b);
        let n = b.len();

        for i in 0..n {
            for d in 0..N {
                let mut iter = self.col_iter(i).peekable();
                while iter.next_if(|n| n.0 < i).is_some() {}
                if let Some(n) = iter.peek() {
                    if n.0 == i && !unit_diagonal {
                        assert!(n.0 <= i);
                        out[i][d] /= n.1;
                        iter.next();
                    }
                }
                let mul = out[i][d];
                for (ri, v) in self.col_iter(i) {
                    use std::cmp::Ordering::*;
                    // ensure that only using the lower part
                    match ri.cmp(&i) {
                        Greater => out[ri][d] -= v * mul,
                        Equal | Less => {}
                    }
                }
            }
        }
    }

    /// Solves an upper triangular system, `self` is a matrix of NxN, and `b` is a column vector of size N
    /// Assuming that b is dense.
    pub fn dense_upper_triangular_solve(&self, b: &[F], out: &mut [F]) {
        self.dense_upper_triangular_solve_arr(
            unsafe { std::mem::transmute::<_, &[[F; 1]]>(b) },
            unsafe { std::mem::transmute(out) },
        );
    }

    /// Solves an upper triangular system, `self` is a matrix of NxN, and `b` is a column vector of size N
    /// Assuming that b is dense.
    pub fn dense_upper_triangular_solve_arr<const N: usize>(
        &self,
        b: &[[F; N]],
        out: &mut [[F; N]],
    ) {
        assert_eq!(self.nrows(), self.ncols());
        assert_eq!(self.ncols(), b.len());
        assert_eq!(out.len(), b.len());
        out.copy_from_slice(b);
        let n = b.len();

        for i in (0..n).rev() {
            for d in 0..N {
                let mut iter = self.col_iter(i).rev().peekable();
                while iter.next_if(|n| n.0 > i).is_some() {}
                if let Some(n) = iter.peek() {
                    if n.0 == i {
                        out[i][d] /= *n.1;
                        iter.next();
                    }
                }
                // introduce a NaN, intentionally, if the diagonal doesn't have a value.
                let mul = out[i][d];
                for (row, &v) in iter {
                    use std::cmp::Ordering::*;
                    match row.cmp(&i) {
                        Less => out[row][d] -= v * mul,
                        Equal | Greater => {}
                    }
                }
            }
        }
    }

    /// Solves a sparse lower triangular system `Ax = b`, with both the matrix and vector
    /// sparse.
    /// sparsity_idxs should be precomputed using the sparse_lower_triangle pattern.
    ///
    /// `out_sparsity_pattern` must also be pre-sorted.
    ///
    /// Assumes that the diagonal of the sparse matrix is all 1 if `assume_unit` is true.
    pub(crate) fn sparse_lower_triangular_solve_sorted(
        &self,
        // input vector idxs & values
        b_idxs: &[usize],
        b: &[F],
        // idx -> row
        // for now, is permitted to be unsorted
        // TODO maybe would be better to enforce sorted, but would have to sort internally.
        out_sparsity_pattern: &[usize],
        out: &mut [F],
        assume_unit: bool,
    ) {
        debug_assert_eq!(self.nrows(), self.ncols());
        debug_assert_eq!(b.len(), b_idxs.len());
        debug_assert!(b_idxs.iter().all(|&bi| bi < self.ncols()));

        debug_assert_eq!(out_sparsity_pattern.len(), out.len());
        debug_assert!(out_sparsity_pattern.iter().all(|&i| i < self.ncols()));

        // initialize out with b
        // TODO can make this more efficient by keeping two iterators in sorted order
        out.fill(0.);
        for i in 0..b.len() {
            let bv = unsafe { *b.get_unchecked(i) };
            let bi = unsafe { *b_idxs.get_unchecked(i) };
            let Some(out_pos) = out_sparsity_pattern.iter().position(|&p| p == bi) else {
                continue;
            };
            *unsafe { out.get_unchecked_mut(out_pos) } = bv;
        }
        // end init

        // assuming that the output sparsity pattern is sorted
        // iterate thru
        for (i, &row) in out_sparsity_pattern.iter().enumerate() {
            let mut iter = self.col_iter(row).peekable();
            if !assume_unit {
                while iter.next_if(|n| n.0 < row).is_some() {}
                match iter.peek() {
                    Some((r, l_val)) if *r == row => {
                        *unsafe { out.get_unchecked_mut(i) } /= **l_val
                    }
                    // here it now becomes implicitly 0,
                    // likely this should introduce NaN or some other behavior.
                    _ => {}
                }
            }
            let mul = unsafe { *out.get_unchecked(i) };
            for (ni, &nrow) in out_sparsity_pattern.iter().enumerate().skip(i + 1) {
                debug_assert!(nrow > row);
                while iter.next_if(|n| n.0 < nrow).is_some() {}
                let l_val = match iter.peek() {
                    Some((r, l_val)) if *r == nrow => l_val,
                    _ => continue,
                };
                *unsafe { out.get_unchecked_mut(ni) } -= *l_val * mul;
            }
        }
    }
}

/// An incremental builder for a Csc matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscBuilder<T>(CsBuilder<T>);

impl<T> CscBuilder<T> {
    /// Constructs a new instance of a Csc Builder.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self(CsBuilder::new(cols, rows))
    }
    /// Convert back from a matrix to a CscBuilder.
    #[inline]
    pub fn from_mat(mat: Csc<T>) -> Self {
        Self(CsBuilder::from_mat(mat.0))
    }
    /// Backtracks back to column `col`, deleting all entries ahead of it.
    pub fn revert_to_col(&mut self, col: usize) -> bool {
        self.0.revert_to_major(col)
    }
    /// Inserts a value into the builder. Must be called in ascending col, row order.
    pub fn insert(&mut self, row: usize, col: usize, val: T) -> Result<(), BuilderInsertError> {
        self.0.insert(col, row, val)
    }
    /// Converts this builder into a valid Csc.
    pub fn build(self) -> Csc<T> {
        Csc(self.0.build())
    }
}
