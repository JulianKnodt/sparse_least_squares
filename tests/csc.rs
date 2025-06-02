use sparse_lu::Csc;

#[test]
fn test_dense_lower_triangular_solve() {
    let mut a = Csc::identity(3);
    let v = [1., 2., 3.];
    let mut out = [0.; 3];
    a.dense_lower_triangular_solve(&v, &mut out, true);
    assert_eq!(out, v);
    a.dense_lower_triangular_solve(&v, &mut out, false);
    assert_eq!(out, v);

    a.values_mut()[0] = 2.;

    a.dense_lower_triangular_solve(&v, &mut out, false);
    assert_eq!(out, [0.5, 2., 3.]);
    a.dense_lower_triangular_solve(&v, &mut out, true);
    assert_eq!(out, v);
}

#[test]
fn test_permuting() {
    let mut a = Csc::identity(3);
    a.swap_rows(1, 2);
    assert_eq!(a.col(0).1, &[0]);
    assert_eq!(a.col(1).1, &[2]);
    assert_eq!(a.col(2).1, &[1]);

    let mut dense = Csc::from_triplets(
        3,
        3,
        &mut [
            ([0, 0], 0.),
            ([0, 1], 1.),
            ([0, 2], 2.),
            ([1, 0], 3.),
            ([1, 1], 4.),
            ([1, 2], 5.),
            ([2, 0], 6.),
            ([2, 1], 7.),
            ([2, 2], 8.),
        ],
    )
    .unwrap();
    dense.swap_rows(1, 2);
    assert_eq!(
        ([0., 2., 1.].as_slice(), [0, 1, 2].as_slice()),
        dense.col(0)
    );
    assert_eq!(
        ([3., 5., 4.].as_slice(), [0, 1, 2].as_slice()),
        dense.col(1)
    );
    assert_eq!(
        ([6., 8., 7.].as_slice(), [0, 1, 2].as_slice()),
        dense.col(2)
    );
}
