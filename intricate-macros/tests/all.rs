#[test]
fn tests() {
    let t = trybuild::TestCases::new();
    // really have to take a look at what compile error happned here to be sure it is working
    t.compile_fail("tests/activation.rs");
    t.pass("tests/layer_enum.rs");
}
