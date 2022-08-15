//! A library that implements some very useful macros that help writing code for the
//! [Intricate](https://github.com/gabrielmfern/intricate) library.
//!
//! Intricate is GPU accelerated machine learning Rust library that uses OpenCL under the hood to
//! do computations on any device that has OpenCL implementations.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(EnumLayer)]
/// Derives the implementation of intricate::layers::Layer for
/// a enum containing layers, this is used as to not have to write
/// this implementation and change it everytime we need to add a new layer.
///
/// This will also derive `From<...>` for every layer variant of the enum.
pub fn enum_layer(_input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(_input as DeriveInput);
    let enum_name = &input.ident;

    let layer_variants = if let Data::Enum(enm) = input.data {
        enm.variants
    } else {
        panic!("The 'EnumLayer' derive macro can only be used with enums!");
    };

    let layer_names = layer_variants.iter().map(|variant| &variant.ident);
    // not sure how this is actually to be implemented
    // compiler keeps complaining that quote! moves layer_names
    // so I cant use it twice because it would be a use of a moved value
    let layer_names_2 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_3 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_4 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_5 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_6 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_7 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_8 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_9 = layer_variants.iter().map(|variant| &variant.ident);
    let layer_names_10 = layer_variants.iter().map(|variant| &variant.ident); // lol

    let layer_types = layer_variants.iter().map(|variant| {
        let variant_fields = match &variant.fields {
            Fields::Unnamed(fields) => &fields.unnamed,
            _ => panic!(
                "Every variant of the enum must be a layer, therefore can only contain one unnamed field which is the actual layer"
            )
        };

        &variant_fields.first().expect("Every variant of the enum must be a layer, therefore can only contain one unnamed field which is the actual layer").ty
    });

    TokenStream::from(quote! {
        #(impl<'a> From<#layer_types> for #enum_name<'a> {
            fn from(layer: #layer_types) -> Self {
                #enum_name::#layer_names(layer)
            }
        })*

        impl<'a> crate::layers::Layer<'a> for #enum_name<'a> {
            fn get_last_inputs(&self) -> Option<&opencl3::memory::Buffer<opencl3::device::cl_float>> {
                match self {
                    #(
                        #enum_name::#layer_names_2(layer) => layer.get_last_inputs(),
                    )*
                }
            }

            fn get_last_outputs(&self) -> Option<&opencl3::memory::Buffer<opencl3::device::cl_float>> {
                match self {
                    #(
                        #enum_name::#layer_names_3(layer) => layer.get_last_outputs(),
                    )*
                }
            }

            fn get_inputs_amount(&self) -> usize {
                match self {
                    #(
                        #enum_name::#layer_names_4(layer) => layer.get_inputs_amount(),
                    )*
                }
            }

            fn get_outputs_amount(&self) -> usize {
                match self {
                    #(
                        #enum_name::#layer_names_5(layer) => layer.get_outputs_amount(),
                    )*
                }
            }

            fn init(
                &mut self,
                queue: &'a opencl3::command_queue::CommandQueue,
                context: &'a opencl3::context::Context,
            ) -> Result<(), crate::types::CompilationOrOpenCLError> {
                match self {
                    #(
                        #enum_name::#layer_names_6(layer) => layer.init(queue, context),
                    )*
                }
            }

            fn clean_up_gpu_state(&mut self) -> () {
                match self {
                    #(
                        #enum_name::#layer_names_7(layer) => layer.clean_up_gpu_state(),
                    )*
                }
            }

            fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), opencl3::error_codes::ClError> {
                match self {
                    #(
                        #enum_name::#layer_names_8(layer) => layer.sync_data_from_gpu_with_cpu(),
                    )*
                }
            }

            fn propagate(
                &mut self, 
                inputs: &opencl3::memory::Buffer<opencl3::device::cl_float>
            ) -> Result<
                &opencl3::memory::Buffer<opencl3::device::cl_float>, 
                opencl3::error_codes::ClError
            > {
                match self {
                    #(
                        #enum_name::#layer_names_9(layer) => layer.propagate(inputs),
                    )*
                }
            }

            fn back_propagate(
                &mut self,
                should_calculate_input_to_error_derivative: bool,
                layer_output_to_error_derivative: &opencl3::memory::Buffer<opencl3::device::cl_float>,
                learning_rate: opencl3::device::cl_float,
            ) -> Result<
                Option<opencl3::memory::Buffer<opencl3::device::cl_float>>, 
                opencl3::error_codes::ClError
            > {
                match self {
                    #(
                        #enum_name::#layer_names_10(layer) => layer.back_propagate(
                            should_calculate_input_to_error_derivative,
                            layer_output_to_error_derivative,
                            learning_rate,
                        ),
                    )*
                }
            }
        }
    })
}

#[proc_macro_derive(ActivationLayer)]
/// Derives the implementation of intricate::layers::Layer for
/// a layer that is an activation function which is the same
/// for all activations.
///
/// Will require for there to be the following constants in scope:
/// PROGRAM_SOURCE, PROPAGATE_KERNEL_NAME, BACK_PROPAGATE_KERNEL_NAME
/// Will require that the intModelLayer::#activation_name exists.
/// Will also require that the struct has the following properties:
///
/// - **inputs_amount**
/// - **opencl_context**
/// - **opencl_queue**
/// - **opencl_program**
/// - **opencl_propagate_kernel**
/// - **opencl_back_propagate_kernel**
/// - **last_outputs_buffer**
/// - **last_inputs_buffer**
pub fn activation_layer(_input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(_input as DeriveInput);
    let activation_name = &input.ident;

    TokenStream::from(quote! {
        impl<'a> #activation_name<'a> {
            pub fn new_raw(inputs_amount: usize) -> #activation_name<'a> {
                #activation_name {
                    inputs_amount,
                    opencl_context: None,
                    opencl_queue: None,
                    opencl_program: None,
                    opencl_propagate_kernel: None,
                    opencl_back_propagate_kernel: None,
                    last_outputs_buffer: None,
                    last_inputs_buffer: None,
                }
            }

            pub fn new(inputs_amount: usize) -> crate::types::ModelLayer<'a> {
                Self::new_raw(inputs_amount).into()
            }
        }

        use opencl3::memory::ClMem;

        impl<'a> crate::layers::Layer<'a> for #activation_name<'a> {
            fn init(
                &mut self,
                queue: &'a opencl3::command_queue::CommandQueue,
                context: &'a opencl3::context::Context,
            ) -> Result<(), crate::types::CompilationOrOpenCLError> {
                let program = opencl3::program::Program::create_and_build_from_source(context, PROGRAM_SOURCE, "")?;

                let propagation_kernel = opencl3::kernel::Kernel::create(&program, PROPAGATE_KERNEL_NAME)?;
                let back_propagation_kernel = opencl3::kernel::Kernel::create(&program, BACK_PROPAGATE_KERNEL_NAME)?;

                self.opencl_program = Some(program);
                self.opencl_propagate_kernel = Some(propagation_kernel);
                self.opencl_back_propagate_kernel = Some(back_propagation_kernel);
                self.opencl_queue = Some(queue);
                self.opencl_context = Some(context);

                Ok(())
            }

            fn get_last_inputs(&self) -> Option<&opencl3::memory::Buffer<opencl3::device::cl_float>> {
                self.last_inputs_buffer.as_ref()
            }

            fn get_last_outputs(&self) -> Option<&opencl3::memory::Buffer<opencl3::device::cl_float>> {
                self.last_outputs_buffer.as_ref()
            }

            fn get_inputs_amount(&self) -> usize {
                self.inputs_amount
            }

            fn get_outputs_amount(&self) -> usize {
                self.inputs_amount
            }

            fn clean_up_gpu_state(&mut self) -> () {
                if self.last_inputs_buffer.is_some() {
                    drop(self.last_inputs_buffer.as_ref().unwrap());
                }

                if self.last_outputs_buffer.is_some() {
                    drop(self.last_outputs_buffer.as_ref().unwrap());
                }
            }

            fn sync_data_from_gpu_with_cpu(&mut self) -> Result<(), opencl3::error_codes::ClError> {
                Ok(())
            }

            fn propagate(&mut self, inputs: &opencl3::memory::Buffer<opencl3::device::cl_float>) -> Result<&opencl3::memory::Buffer<opencl3::device::cl_float>, opencl3::error_codes::ClError> {
                assert!(self.opencl_context.is_some());
                assert!(self.opencl_queue.is_some());

                let context = self.opencl_context.unwrap();
                let queue = self.opencl_queue.unwrap();

                let inputs_size = inputs.size()?;
                let inputs_total_count = inputs_size / std::mem::size_of::<opencl3::device::cl_float>();

                let mut copied_last_inputs_buffer = opencl3::memory::Buffer::<opencl3::device::cl_float>::create(
                    context,
                    opencl3::memory::CL_MEM_READ_ONLY,
                    inputs_total_count,
                    std::ptr::null_mut(),
                )?;

                // TODO: make copying this into the last inputs optional since this is only needed
                // for fitting a model as to make everything more optimized both in RAM usage and computation
                queue
                    .enqueue_copy_buffer(
                        inputs,
                        &mut copied_last_inputs_buffer,
                        0,
                        0,
                        inputs_size,
                        &[],
                    )?
                    .wait()?;

                self.last_inputs_buffer = Some(copied_last_inputs_buffer);

                let outputs_total_count = inputs.size()? / std::mem::size_of::<opencl3::device::cl_float>();

                let outputs_buffer = opencl3::memory::Buffer::<opencl3::device::cl_float>::create(
                    self.opencl_context.unwrap(),
                    opencl3::memory::CL_MEM_READ_WRITE,
                    outputs_total_count,
                    std::ptr::null_mut(),
                )?;

                opencl3::kernel::ExecuteKernel::new(self.opencl_propagate_kernel.as_ref().unwrap())
                    .set_arg(inputs)
                    .set_arg(&outputs_buffer)
                    .set_arg(&(outputs_total_count as opencl3::error_codes::cl_int))
                    .set_global_work_size(outputs_total_count)
                    .enqueue_nd_range(self.opencl_queue.unwrap())?
                    .wait()?;

                self.last_outputs_buffer = Some(outputs_buffer);

                Ok(self.last_outputs_buffer.as_ref().unwrap())
            }

        fn back_propagate(
                &mut self,
                should_calculate_input_to_error_derivative: bool,
                layer_output_to_error_derivative: &opencl3::memory::Buffer<opencl3::device::cl_float>,
                _: opencl3::device::cl_float,
            ) -> Result<Option<opencl3::memory::Buffer<opencl3::device::cl_float>>, opencl3::error_codes::ClError> {
                if should_calculate_input_to_error_derivative {
                    assert!(self.opencl_context.is_some());
                    assert!(self.opencl_queue.is_some());

                    let samples_amount = self.last_outputs_buffer.as_ref().unwrap().size()?
                        / self.inputs_amount
                        / std::mem::size_of::<opencl3::device::cl_float>();

                    assert_eq!(samples_amount % 1, 0);

                    let loss_to_input_derivatives_buffer = opencl3::memory::Buffer::<opencl3::device::cl_float>::create(
                        self.opencl_context.unwrap(),
                        opencl3::memory::CL_MEM_READ_WRITE,
                        self.inputs_amount * samples_amount,
                        std::ptr::null_mut(),
                    )?;

                    opencl3::kernel::ExecuteKernel::new(self.opencl_back_propagate_kernel.as_ref().unwrap())
                        .set_arg(layer_output_to_error_derivative)
                        .set_arg(self.last_outputs_buffer.as_ref().unwrap())
                        .set_arg(&loss_to_input_derivatives_buffer)
                        .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
                        .set_arg(&(samples_amount as opencl3::error_codes::cl_int))
                        .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
                        .set_global_work_sizes(&[samples_amount, self.inputs_amount])
                        .enqueue_nd_range(self.opencl_queue.unwrap())?
                        .wait()?;

                    Ok(Some(loss_to_input_derivatives_buffer))
                } else {
                    Ok(None)
                }
            }
        }
    })
}