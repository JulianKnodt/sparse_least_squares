/// An error when adding into the SparsityPatternBuilder
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BuilderInsertError {
    ///
    MajorTooLow(usize),
    /// Provided Minor, Current Minor
    MinorTooLow(usize, usize),
}

/// How the sparsity for a matrix is laid out
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SparsityPattern {
    pub(crate) major_offsets: Vec<usize>,
    pub(crate) minor_indices: Vec<usize>,
    pub minor_dim: usize,
}

impl SparsityPattern {
    pub fn major_dim(&self) -> usize {
        self.major_offsets.len() - 1
    }

    pub fn lane(&self, i: usize) -> &[usize] {
        let s = self.major_offsets[i];
        let e = self.major_offsets[i + 1];
        &self.minor_indices[s..e]
    }

    pub fn nnz(&self) -> usize {
        self.minor_indices.len()
    }

    pub fn entries(&self) -> impl Iterator<Item = [usize; 2]> + '_ {
        (0..self.major_dim()).flat_map(move |i| {
            let s = self.major_offsets[i];
            let e = self.major_offsets[i + 1];
            (s..e).map(move |j| [i, self.minor_indices[j]])
        })
    }
    /// Creates the sparsity pattern of an identity matrix of size `n`.
    pub fn identity(n: usize) -> Self {
        Self {
            major_offsets: (0..=n).collect(),
            minor_indices: (0..n).collect(),
            minor_dim: n,
        }
    }

    /// Computes the output sparsity pattern of `x` in `Ax = b`.
    /// where A's nonzero pattern is given by `self` and the non-zero indices
    /// of vector `b` are specified as a slice.
    /// The output is not necessarily in sorted order, but is topological sort order.
    /// Treats `self` as lower triangular, even if there are elements in the upper triangle.
    /// Acts as if b is one major lane (i.e. CSC matrix and one column)
    pub fn sparse_lower_triangular_solve(&self, b: &[usize], out: &mut Vec<usize>) {
        debug_assert!(b.iter().all(|&i| i < self.major_dim()));
        out.clear();

        // From a given starting column, traverses and finds all reachable indices.
        fn reach(sp: &SparsityPattern, j: usize, out: &mut Vec<usize>) {
            // TODO this may be slow?

            // already traversed
            if out.contains(&j) {
                return;
            }

            out.push(j);
            for &i in sp.lane(j) {
                if i < j {
                    continue;
                }
                reach(sp, i, out);
            }
        }

        for &i in b {
            reach(&self, i, out);
        }
    }

    /// Computes the output sparsity pattern of `x` in `Ax = b`.
    /// where A's nonzero pattern is given by `self` and the non-zero indices
    /// of vector `b` are specified as a slice.
    /// The output is not necessarily in sorted order, but is topological sort order.
    /// Treats `self` as lower triangular, even if there are elements in the upper triangle.
    /// Acts as if b is one major lane (i.e. CSC matrix and one column)
    pub(crate) fn sparse_lower_triangular_solve_bool(
        &self,
        b: &[usize],
        out: &mut Vec<bool>,
        stack: &mut Vec<u32>,
    ) {
        out.fill(false);

        // From a given starting column, traverses and finds all reachable indices.
        for &i in b {
            stack.push(i as u32);
            while let Some(j) = stack.pop() {
                // already traversed
                if out[j as usize] {
                    return;
                }

                out[j as usize] = true;
                for &i in self.lane(j as usize) {
                    if (i as u32) < j {
                        continue;
                    }
                    stack.push(j as u32);
                }
            }
        }
    }

    /// Computes the output sparsity pattern of `x` in `Ax = b`.
    /// where A's nonzero pattern is given by `self` and the non-zero indices
    /// of vector `b` are specified as a slice.
    /// The output is not necessarily in sorted order, but is topological sort order.
    /// Treats `self` as upper triangular, even if there are elements in the lower triangle.
    /// Acts as if b is one major lane (i.e. CSC matrix and one column)
    pub fn sparse_upper_triangular_solve(&self, b: &[usize], out: &mut Vec<usize>) {
        assert!(b.iter().all(|&i| i < self.major_dim()));
        out.clear();

        // From a given starting column, traverses and finds all reachable indices.
        fn reach(sp: &SparsityPattern, j: usize, out: &mut Vec<usize>) {
            // already traversed
            if out.contains(&j) {
                return;
            }

            out.push(j);
            // iteration order here does not matter, but technically it should be rev?
            for &i in sp.lane(j).iter().rev() {
                if i > j {
                    continue;
                }
                reach(sp, i, out);
            }
        }

        for &i in b {
            reach(&self, i, out);
        }
    }
}

/// A builder that allows for constructing a sparsity pattern.
/// It requires elements to be added in sorted order. Specifically,
/// For each element the major must be >= the previous element's major.
/// If the major is the same, the minor must be in ascending order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsityPatternBuilder {
    buf: SparsityPattern,
    major_dim: usize,
}

impl SparsityPatternBuilder {
    /// Constructs a new empty builder.
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            buf: SparsityPattern {
                major_offsets: vec![0],
                minor_indices: vec![],
                minor_dim,
            },
            major_dim,
        }
    }
    /// The number of non-zero entries inserted into `self`.
    pub fn num_entries(&self) -> usize {
        self.buf.minor_indices.len()
    }

    /// Allows for general assignment of indices
    pub fn insert(&mut self, maj: usize, min: usize) -> Result<(), BuilderInsertError> {
        assert!(maj < self.major_dim);
        assert!(min < self.buf.minor_dim);

        let curr_major = self.buf.major_dim();

        // cannot go backwards in major
        if maj < curr_major {
            return Err(BuilderInsertError::MajorTooLow(curr_major));
        }
        // cannot go backwards in minor
        if maj == curr_major
            && *self.buf.major_offsets.last().unwrap() < self.buf.minor_indices.len()
            && !self.buf.minor_indices.is_empty()
            && min <= *self.buf.minor_indices.last().unwrap()
        {
            return Err(BuilderInsertError::MinorTooLow(
                min,
                *self.buf.minor_indices.last().unwrap(),
            ));
        }
        // add any advances in row.
        for _ in curr_major..maj {
            self.buf.major_offsets.push(self.buf.minor_indices.len());
        }
        self.buf.minor_indices.push(min);
        Ok(())
    }
    /*
    /// Returns a valid partial sparsity pattern.
    /// All the major lanes up to the current insertion will be completed.
    pub(crate) fn valid_partial(&mut self) -> &SparsityPattern {
        if *self.buf.major_offsets.last().unwrap() != self.buf.minor_indices.len() {
            self.buf.major_offsets.push(self.buf.minor_indices.len());
        }
        &self.buf
    }
    */
    /// Consumes self and outputs the constructed `SparsityPattern`.
    /// If elements were added to the last major, but `advance_major`
    /// was not called, will implicitly call `advance_major` then
    /// output the values.
    #[inline]
    pub fn build(mut self) -> SparsityPattern {
        self.buf
            .major_offsets
            .resize(self.major_dim + 1, self.buf.minor_indices.len());
        debug_assert_eq!(self.buf.major_dim(), self.major_dim);
        self.buf
    }

    /// Reverts the major index of `self` back to `maj`, deleting any entries ahead of it.
    /// Preserves entries in `maj`.
    pub fn revert_to_major(&mut self, maj: usize) -> bool {
        // preserve maj + 1 elements in self
        if self.buf.major_offsets.len() + 1 <= maj {
            return false;
        }
        let last = self.buf.major_offsets[maj + 1];
        self.buf.major_offsets.truncate(maj + 1);
        self.buf.minor_indices.truncate(last + 1);
        true
    }

    /// Allows for rebuilding part of a sparsity pattern, assuming that
    /// items after maj_start have not been filled in.
    pub fn from(sp: SparsityPattern) -> Self {
        SparsityPatternBuilder {
            major_dim: sp.major_dim(),
            buf: sp,
        }
    }

    /// Returns the current major being modified by `self`.
    pub fn current_major(&self) -> usize {
        debug_assert!(!self.buf.major_offsets.is_empty());
        self.buf.major_offsets.len() - 1
    }
}
