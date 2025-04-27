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
    if let SparseEntryMut::NonZero(e) = a.index_entry_mut(0, 0) {
        *e = 2.;
    };
    a.dense_lower_triangular_solve(&v, &mut out, false);
    assert_eq!(out, [0.5, 2., 3.]);
    a.dense_lower_triangular_solve(&v, &mut out, true);
    assert_eq!(out, v);
}
