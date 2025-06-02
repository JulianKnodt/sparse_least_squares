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
    let n = 2;
    let mut a = CscBuilder::new(n, n);
    for i in 0..n {
        assert!(a.insert(i, i, if i == 0 { 1. } else { 0.5 }).is_ok());
        if i == 0 {
            assert_matches!(a.insert(1, 0, 2.), Ok(_));
        }
    }
    // construct an identity matrix as a basic test
    let a = a.build();

    let lu_fact = LeftLookingLUFactorization::new(&a);

    let gt = Csc::from_triplets(
        2,
        2,
        &mut [
            ([0, 0], 2.), //
            ([0, 1], 0.5),
            ([1, 0], 0.5),
            ([1, 1], -0.25),
        ],
    )
    .unwrap();

    assert_eq!(lu_fact.pivot(), &[1, 0]);
    let new_lu = lu_fact.lu();
    assert_eq!(lu_fact.lu(), &gt, "{:?} {:?}", new_lu.col(0), new_lu.col(1));
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
    let mut out = [1., 2., 4.];
    let og = out;
    lu_fact.solve(&mut out, &mut buf);
    let solved = a.vecmul(&out);
    for i in 0..3 {
        assert!((solved[i] - og[i]).abs() < 1e-5);
    }
}

#[test]
pub fn test_lu_fact_sparse_pivot() {
    let a = Csc::from_triplets(
        3,
        3,
        &mut [
            ([0, 1], 50.),
            ([0, 2], 238.28),
            ([1, 1], 1000.),
            ([2, 0], 87.),
        ],
    )
    .unwrap();
    let lu_fact = LeftLookingLUFactorization::new(&a);
    let mut buf = [0.; 3];
    let mut out = [100., 0.02, 10.];
    let og = out;
    lu_fact.solve(&mut out, &mut buf);
    let solved = a.vecmul(&out);
    for i in 0..3 {
        assert!(
            (solved[i] - og[i]).abs() < 1e-3,
            "i={i}: {} {} (whole = {solved:?})",
            solved[i],
            og[i]
        );
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
            ([1, 0], -92.),
            ([1, 1], 12.),
            ([1, 2], 31.),
            ([2, 0], -16.),
            ([2, 1], 0.),
            ([2, 2], 87.),
        ],
    )
    .unwrap();
    let lu_fact = LeftLookingLUFactorization::new(&a);
    assert_eq!(lu_fact.pivot(), &[1, 0, 2]);
    let mut buf = [0.; 3];
    let mut out = [1., 2., 4.];
    let og = out;
    lu_fact.solve(&mut out, &mut buf);
    let solved = a.vecmul(&out);
    for i in 0..3 {
        assert!((solved[i] - og[i]).abs() < 1e-5, "{:?}", solved);
    }
}
