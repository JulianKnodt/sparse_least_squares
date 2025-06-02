use super::F;
use crate::csc::{Csc, CscBuilder};

/// Constructs an LU Factorization using a left-looking approach.
/// This means it will construct each column, starting from the leftmost one.
pub struct LeftLookingLUFactorization<T> {
    /// A single matrix stores both the lower and upper triangular components
    l_u: Csc<T>,

    pivot: Vec<usize>,
}

impl LeftLookingLUFactorization<F> {
    /// Returns the joint L\U matrix. Here, `L` implicitly has 1 along the diagonal.
    /// Do not forget that there is a pivot if reading from this matrix
    #[inline]
    pub fn lu(&self) -> &Csc<F> {
        &self.l_u
    }

    /// Pivot vector for the LU matrix
    #[inline]
    pub fn pivot(&self) -> &[usize] {
        &self.pivot
    }
    //fn apply_pivot<T>(&self, v: &mut [T]) {}

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
    /// The output will be stored in b, and buf is used as a temporary buffer.
    pub fn solve(&self, b: &mut [F], buf: &mut [F]) {
        assert_eq!(b.len(), buf.len());
        // Implementation: Solve two systems: Ly = b, then Ux = y.
        self.l_u.dense_lower_triangular_solve(b, buf, true);
        self.l_u.dense_upper_triangular_solve(buf, b);
    }

    /// Computes `x` in `LUx = b`, where `b` is a dense vector.
    /// The output will be stored in b, and buf is used as a temporary buffer.
    pub fn solve_arr<const N: usize>(&self, b: &mut [[F; N]], buf: &mut [[F; N]]) {
        assert_eq!(b.len(), buf.len());
        // Implementation: Solve two systems: Ly = b, then Ux = y.
        self.l_u.dense_lower_triangular_solve_arr(b, buf, true);
        self.l_u.dense_upper_triangular_solve_arr(buf, b);
    }

    /// Construct a new sparse LU factorization
    /// from a given CSC matrix.
    pub fn new(a: &Csc<F>) -> Self {
        assert_eq!(a.nrows(), a.ncols());
        let n = a.nrows();

        let mut pivot = vec![0; n];
        for i in 0..n {
            pivot[i] = i;
        }

        // this initially starts as an identity  matrix.
        // but the ones are all implicit.
        let mut csc_builder: CscBuilder<F> = CscBuilder::new(n, n);

        let mut val_buf = vec![];
        let mut pat_contains = vec![false; n];
        let mut pat_buf = vec![];
        let mut stack = vec![];

        for ci in 0..n {
            let curr_mat = csc_builder.build();
            /*
            let col_iter = curr_mat.col_iter(ci);
            let opt = col_iter
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("Non-full rank input").0;
            println!("{}", opt);
            */

            let (col_vals, col_ris) = a.col(ci);
            curr_mat.pattern().sparse_lower_triangular_solve_bool(
                col_ris,
                &mut pat_contains,
                &mut stack,
            );
            pat_buf.clear();
            pat_buf.extend(
                pat_contains
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| c.then_some(i)),
            );

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
            let v = csc_builder.revert_to_col(ci);
            debug_assert!(v);
            let mut ukk = 0.;
            debug_assert_eq!(pat_buf.len(), val_buf.len());
            for i in 0..pat_buf.len() {
                let row = unsafe { *pat_buf.get_unchecked(i) };
                let val = unsafe { *val_buf.get_unchecked(i) };
                use std::cmp::Ordering;
                let val = match row.cmp(&ci) {
                    Ordering::Less => val,
                    Ordering::Equal => {
                        ukk = val;
                        val
                    }
                    Ordering::Greater => {
                        assert_ne!(ukk, 0., "{val} {i} row={row} col={ci}");
                        val / ukk
                    }
                };
                assert!(val.is_finite());
                let ins = csc_builder.insert(row, ci, val);
                debug_assert_eq!(ins, Ok(()));
            }
        }

        let l_u = csc_builder.build();
        assert!(l_u.values().iter().copied().all(F::is_finite));
        Self { l_u, pivot }
    }
}
