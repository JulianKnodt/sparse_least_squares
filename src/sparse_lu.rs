//use crate::SparseEntryMut;
use super::F;
use crate::csc::{Csc, CscBuilder};

/// Constructs an LU Factorization using a left-looking approach.
/// This means it will construct each column, starting from the leftmost one.
pub struct LeftLookingLUFactorization<T> {
    /// A single matrix stores both the lower and upper triangular components
    l_u: Csc<T>,
}

impl LeftLookingLUFactorization<F> {
    /// Returns the joint L\U matrix. Here, `L` implicitly has 1 along the diagonal.
    pub fn lu(&self) -> &Csc<F> {
        &self.l_u
    }

    /*
    /// Returns the upper triangular part of this matrix.
    pub fn u(&self) -> Csc<T> {
        self.l_u.upper_triangle()
    }

    /// Returns the lower triangular part of this matrix.
    pub fn l(&self) -> Csc<T> {
        let mut l = self.l_u.lower_triangle();
        let n = self.l_u.nrows();
        for i in 0..n {
            if let SparseEntryMut::NonZero(v) = l.index_entry_mut(i, i) {
                *v = T::one();
            } else {
                unreachable!();
            }
        }
        l
    }
    */

    /// Computes `x` in `LUx = b`, where `b` is a dense vector.
    pub fn solve(&self, b: &[F], dst: &mut [F]) {
        assert_eq!(b.len(), dst.len());
        let mut y = vec![0.; b.len()];
        // Implementation: Solve two systems: Ly = b, then Ux = y.
        self.l_u.dense_lower_triangular_solve(b, &mut y, true);
        self.l_u.dense_upper_triangular_solve(&y, dst);
    }

    /// Construct a new sparse LU factorization
    /// from a given CSC matrix.
    pub fn new(a: &Csc<F>) -> Self {
        assert_eq!(a.nrows(), a.ncols());
        let n = a.nrows();

        // this initially starts as an identity  matrix.
        // but the ones are all implicit.
        let mut csc_builder = CscBuilder::new(n, n);

        let mut val_buf = vec![];
        let mut pat_buf = vec![];

        for ci in 0..a.ncols() {
            let curr_mat = csc_builder.build();

            let (col_vals, col_ris) = a.col(ci);
            curr_mat
                .pattern()
                .sparse_lower_triangular_solve(col_ris, &mut pat_buf);
            pat_buf.sort_unstable();
            val_buf.resize(pat_buf.len(), 0.);

            // Solve the current column, assuming that it is lower triangular
            curr_mat.sparse_lower_triangular_solve_sorted(
                col_ris,
                col_vals,
                &pat_buf,
                &mut val_buf,
                true,
            );

            // convert builder back to matrix
            csc_builder = CscBuilder::from_mat(curr_mat);
            assert!(csc_builder.revert_to_col(ci));
            let mut ukk = 0.;
            for (row, val) in pat_buf.drain(..).zip(val_buf.drain(..)) {
                use std::cmp::Ordering;
                let val = match row.cmp(&ci) {
                    Ordering::Less => val,
                    Ordering::Equal => {
                        ukk = val;
                        val
                    }
                    Ordering::Greater => {
                        assert_ne!(ukk, 0.);
                        val / ukk
                    }
                };
                assert_eq!(csc_builder.insert(row, ci, val), Ok(()));
            }
        }

        let l_u = csc_builder.build();
        Self { l_u }
    }
}
