#![feature(assert_matches)]
use std::assert_matches::assert_matches;

use sparse_lu::LeftLookingLUFactorization;
use sparse_lu::csc::{Csc, CscBuilder};

#[test]
fn test_basic_lu_factorization() {
    let n = 5;
    let mut a = CscBuilder::new(n, n);
    for i in 0..n {
        assert_matches!(a.insert(i, i, 1.), Ok(_));
    }
    // construct an identity matrix as a basic test
    let a = a.build();

    let lu_fact = LeftLookingLUFactorization::new(&a);

    assert_eq!(lu_fact.lu(), &a);
}

#[test]
fn test_basic_lu_factorization_with_one_more_entry() {
    let n = 3;
    let mut a = CscBuilder::new(n, n);
    for i in 0..n {
        assert!(a.insert(i, i, if i == 0 { 0.5 } else { 1. }).is_ok());
        if i == 0 {
            assert_matches!(a.insert(1, 0, 1.), Ok(_));
        }
    }
    // construct an identity matrix as a basic test
    let a = a.build();

    let lu_fact = LeftLookingLUFactorization::new(&a);

    let mut ground_truth = CscBuilder::new(n, n);
    for i in 0..n {
        assert!(
            ground_truth
                .insert(i, i, if i == 0 { 0.5 } else { 1. })
                .is_ok()
        );
        if i == 0 {
            assert!(ground_truth.insert(1, 0, 2.).is_ok());
        }
    }
    let gt = ground_truth.build();

    assert_eq!(lu_fact.lu(), &gt);
}

#[test]
pub fn test_lu_fact_sparse() {
    let a = Csc::from_triplets(
        3,
        3,
        &mut [
            ([0, 0], 47.),
            ([0, 1], 91.),
            ([1, 0], 92.),
            ([1, 1], 12.),
            ([1, 2], 31.),
            ([2, 0], 16.),
            ([2, 2], 87.),
        ],
    )
    .unwrap();
    let lu_fact = LeftLookingLUFactorization::new(&a);
    let mut buf = [0.; 3];
    let mut out = [1.; 3];
    lu_fact.solve(&mut out, &mut buf);
    let solved = a.vecmul(&out);
    for i in 0..3 {
      assert!((solved[i] - 1.).abs() < 1e-5);
    }
}

#[test]
pub fn test_lu_fact_dense() {
    let a = Csc::from_triplets(
        3,
        3,
        &mut [
            ([0, 0], 47.),
            ([0, 1], 91.),
            ([0, 2], 0.),
            ([1, 0], 92.),
            ([1, 1], 12.),
            ([1, 2], 31.),
            ([2, 0], 16.),
            ([2, 1], 0.),
            ([2, 2], 87.),
        ],
    )
    .unwrap();
    let lu_fact = LeftLookingLUFactorization::new(&a);
    let mut buf = [0.; 3];
    let mut out = [1.; 3];
    lu_fact.solve(&mut out, &mut buf);
    let solved = a.vecmul(&out);
    for i in 0..3 {
      assert!((solved[i] - 1.).abs() < 1e-5);
    }
}
