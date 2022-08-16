use intricate_macros::EnumLayer;
use intricate::layers::{
    Dense,
    Layer,
    activations::TanH
};

#[derive(Debug, EnumLayer)]
enum MyLayerEnum<'a> {
    MyDense(Dense<'a>),
    MyTanH(TanH<'a>),
}

fn main() {
    // Should have implemented From<LayerStruct> for every Layer variant of the enum
    let dense: MyLayerEnum = Dense::new_raw(0, 0).into();
    let tanh: MyLayerEnum = TanH::new_raw(0).into();

    // Should have implemented intricate::layers::Layer for the enum and should work for every
    // variant
    let _: Box<dyn Layer> = Box::new(dense);
    let _: Box<dyn Layer> = Box::new(tanh);
}