use super::{BuilderInsertError, SparsityPattern, SparsityPatternBuilder};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsMatrix<T> {
    pub(crate) pattern: SparsityPattern,
    values: Vec<T>,
}

impl<T> CsMatrix<T> {
    pub fn lane(&self, i: usize) -> (&[T], &[usize]) {
        let s = self.pattern.major_offsets[i];
        let e = self.pattern.major_offsets[i + 1];
        (&self.values[s..e], &self.pattern.minor_indices[s..e])
    }
    /*
    pub(crate) fn lane_mut(&mut self, i: usize) -> (&mut [T], &mut [usize]) {
        let s = self.pattern.major_offsets[i];
        let e = self.pattern.major_offsets[i + 1];
        (
            &mut self.values[s..e],
            &mut self.pattern.minor_indices[s..e],
        )
    }
    */

    pub(crate) fn swap_minor(&mut self, a: usize, b: usize) {
        for i in self.pattern.minor_indices.iter_mut() {
            if *i == a {
                *i = b;
            } else if *i == b {
                *i = a;
            }
        }
        // for each major, resort the indices
        for i in 0..self.pattern.major_dim() {
            let s = self.pattern.major_offsets[i];
            let e = self.pattern.major_offsets[i + 1];
            for idx in s..e.saturating_sub(1) {
                let mi = &self.pattern.minor_indices;
                if mi[idx] > mi[idx + 1] {
                    self.values.swap(idx, idx + 1);
                    self.pattern.minor_indices.swap(idx, idx + 1);
                }
            }

            for idx in (s + 1..e).rev() {
                let mi = &self.pattern.minor_indices;
                if mi[idx - 1] > mi[idx] {
                    self.values.swap(idx - 1, idx);
                    self.pattern.minor_indices.swap(idx - 1, idx);
                }
            }
        }
    }
    pub fn lane_iter(
        &self,
        i: usize,
    ) -> impl Iterator<Item = (usize, &T)> + DoubleEndedIterator + '_ {
        let s = self.pattern.major_offsets[i];
        let e = self.pattern.major_offsets[i + 1];
        (s..e).map(|i| (self.pattern.minor_indices[i], &self.values[i]))
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

impl CsMatrix<super::F> {
    pub fn identity(n: usize) -> Self {
        Self {
            pattern: SparsityPattern::identity(n),
            values: vec![1.; n],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsBuilder<T> {
    sparsity_builder: SparsityPatternBuilder,
    values: Vec<T>,
}

impl<T> CsBuilder<T> {
    /// Constructs a new CsBuilder of the given size.
    pub fn new(major_dim: usize, minor_dim: usize) -> Self {
        Self {
            sparsity_builder: SparsityPatternBuilder::new(major_dim, minor_dim),
            values: vec![],
        }
    }
    /// Given an existing CsMatrix, allows for modification by converting it into a builder.
    pub fn from_mat(mat: CsMatrix<T>) -> Self {
        let CsMatrix { pattern, values } = mat;

        CsBuilder {
            sparsity_builder: SparsityPatternBuilder::from(pattern),
            values,
        }
    }
    /// Backtracks to a given major index
    pub fn revert_to_major(&mut self, maj: usize) -> bool {
        if !self.sparsity_builder.revert_to_major(maj) {
            return false;
        }

        self.values.truncate(self.sparsity_builder.num_entries());
        true
    }
    pub(crate) fn insert(
        &mut self,
        maj: usize,
        min: usize,
        val: T,
    ) -> Result<(), BuilderInsertError> {
        self.sparsity_builder.insert(maj, min)?;
        self.values.push(val);
        Ok(())
    }
    pub(crate) fn insert_sum(
        &mut self,
        maj: usize,
        min: usize,
        val: T,
        add: impl Fn(T, T) -> T + Copy,
    ) -> Result<bool, BuilderInsertError>
    where
        T: Copy,
    {
        if self.sparsity_builder.insert_sum(maj, min)? {
            let dst = self.values.last_mut().unwrap();
            *dst = add(*dst, val);
            Ok(true)
        } else {
            self.values.push(val);
            Ok(false)
        }
    }
    pub fn build(self) -> CsMatrix<T> {
        let CsBuilder {
            sparsity_builder,
            values,
        } = self;
        let pattern = sparsity_builder.build();
        CsMatrix { pattern, values }
    }
}
