//! A library that implements some very useful macros that help writing code for the
//! [Intricate](https://github.com/gabrielmfern/intricate) library.
//!
//! Intricate is GPU accelerated machine learning Rust library that uses OpenCL under the hood to
//! do computations on any device that has OpenCL implementations.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Ident};

#[proc_macro_derive(ErrorsEnum)]
/// Derives all the From<Error> implementations for the enum it is being derived on.
pub fn erors_enum(_input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(_input as DeriveInput);
    let enum_name = &input.ident;

    let error_variants = if let Data::Enum(enm) = input.data {
        enm.variants
    } else {
        panic!("The 'ErrorsEnum' derive macro can only be be used with enums!");
    };

    let error_names = error_variants.iter().filter_map(|variant| {
        let variant_fields = match &variant.fields {
            Fields::Unnamed(fields) => Some(&fields.unnamed),
            _ => None,
        };

        if variant_fields.is_some() && variant_fields.unwrap().len() == 1 {
            Some(&variant.ident)
        } else {
            None
        }
    });

    let error_types = error_variants.iter().filter_map(|variant| {
        let variant_fields = match &variant.fields {
            Fields::Unnamed(fields) => Some(&fields.unnamed),
            _ => None,
        };

        if variant_fields.is_some() && variant_fields.unwrap().len() == 1 {
            Some(variant_fields.unwrap().first().unwrap())
        } else {
            None
        }
    });

    quote! {
        #(impl From<#error_types> for #enum_name {
            fn from(err: #error_types) -> Self {
                #enum_name::#error_names(err)
            }
        })*
    }
    .into()
}

#[proc_macro_derive(LossFunctionEnum)]
/// Derives the implementation of intricate::loss_functions::LossFunction for
/// a enum contaning only variants that are loss functions, such as the Mean Squared and others.
///
/// This will also derive `From<...>` for every loss function in the enum.
pub fn loss_function_enum(_input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(_input as DeriveInput);
    let enum_name = &input.ident;

    let variants = if let Data::Enum(enm) = input.data {
        enm.variants
    } else {
        panic!("The 'LossFunctionEnum' derive macro can only be used with enums!");
    };

    let loss_function_names = variants.iter().map(|variant| &variant.ident);
    let loss_function_names_2 = loss_function_names.clone();
    let loss_function_names_3 = loss_function_names.clone();
    let loss_function_names_4 = loss_function_names.clone();

    let loss_types = variants.iter().map(|variant| {
        let variant_fields = match &variant.fields {
            Fields::Unnamed(fields) => &fields.unnamed,
            _ => panic!(
                "Every variant of the enum must be a loss function, therefore can only contain one unnamed field which is the actual loss function"
            )
        };

        &variant_fields.first().expect("Every variant of the enum must be a loss function, therefore can only contain one unnamed field which is the actual loss function").ty
    });

    quote! {
        #(impl<'a> From<#loss_types> for #enum_name<'a> {
            fn from(layer: #loss_types) -> Self {
                #enum_name::#loss_function_names(layer)
            }
        })*

        impl<'a> crate::loss_functions::LossFunction<'a> for #enum_name<'a> {
            fn compute_loss(
                &self,
                output_samples: &opencl3::memory::Buffer<opencl3::device::cl_float>,
                expected_outputs: &opencl3::memory::Buffer<opencl3::device::cl_float>,
                samples_amount: usize,
            ) -> Result<f32, opencl3::error_codes::ClError> {
                match self {
                #(
                    #enum_name::#loss_function_names_2(lossfn) => lossfn.compute_loss(
                        output_samples, 
                        expected_outputs, 
                        samples_amount
                    ),
                )*
                }
            }

            fn init(
                &mut self,
                opencl_state: &'a OpenCLState,
            ) -> Result<(), opencl3::error_codes::ClError> {
                match self {
                #(
                    #enum_name::#loss_function_names_3(lossfn) => lossfn.init(opencl_state),
                )*
                }
            }

            fn compute_loss_derivative_with_respect_to_output_samples(
                &self,
                output_samples: &opencl3::memory::Buffer<opencl3::device::cl_float>,
                expected_outputs: &opencl3::memory::Buffer<opencl3::device::cl_float>,
                samples_amount: usize,
            ) -> Result<opencl3::memory::Buffer<opencl3::device::cl_float>, opencl3::error_codes::ClError> {
                match self {
                #(
                    #enum_name::#loss_function_names_4(lossfn) =>
                        lossfn.compute_loss_derivative_with_respect_to_output_samples(
                            output_samples,
                            expected_outputs,
                            samples_amount,
                        ),
                )*
                }
            }
        }
    }.into()
}

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
    let layer_names_10 = layer_names_9.clone();
    let layer_names_11 = layer_names_9.clone();
    let layer_names_12 = layer_names_9.clone();

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
                opencl_state: &'a crate::utils::OpenCLState,
            ) -> Result<(), opencl3::error_codes::ClError> {
                match self {
                    #(
                        #enum_name::#layer_names_6(layer) => layer.init(opencl_state),
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

            fn sync_data_from_buffers_to_host(&mut self) -> Result<(), opencl3::error_codes::ClError> {
                match self {
                    #(
                        #enum_name::#layer_names_8(layer) => layer.sync_data_from_buffers_to_host(),
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

            fn compute_gradients(
                &self,
                layer_output_to_error_derivative: &Buffer<cl_float>,
            ) -> Result<LayerGradients, LayerGradientComputationError> {
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

            fn apply_gradients(
                &mut self,
                per_parameter_type_gradients: LayerGradients,
                optimizer: dyn Optimizer,
            ) -> Result<(), LayerGradientApplicationError> {
                match self {
                    #(
                        #enum_name::#layer_names_11(layer) => layer.back_propagate(
                            should_calculate_input_to_error_derivative,
                            layer_output_to_error_derivative,
                            learning_rate,
                        ),
                    )*
                }
            }

            fn compute_loss_to_input_derivatives(
                &self,
                layer_output_to_error_derivative: &Buffer<cl_float>,
            ) -> Result<Buffer<cl_float>, LayerLossToInputDifferentiationError> {
                match self {
                    #(
                        #enum_name::#layer_names_12(layer) => layer.back_propagate(
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
    let compile_activation = Ident::new(
        ("compile_".to_string() + &activation_name.to_string().to_lowercase()).as_str(),
        input.ident.span(),
    );

    TokenStream::from(quote! {
        impl<'a> #activation_name<'a> {
            /// Creates a raw version of the #activation_name activation function, this is good for
            /// being used when you don't want to use the layer in a Model.
            pub fn new_raw(inputs_amount: usize) -> #activation_name<'a> {
                #activation_name {
                    inputs_amount,

                    last_outputs_buffer: None,
                    last_inputs_buffer: None,

                    opencl_state: None,
                }
            }

            /// Creates a ModelLayer version of the #activation_name activation function, to be
            /// used with a Model.
            pub fn new(inputs_amount: usize) -> crate::types::ModelLayer<'a> {
                Self::new_raw(inputs_amount).into()
            }
        }

        pub(crate) fn #compile_activation(opencl_state: &mut OpenCLState) -> Result<(), crate::utils::opencl::EnsureKernelsAndProgramError> {
            let kernels = &[PROPAGATE_KERNEL_NAME.to_string(), BACK_PROPAGATE_KERNEL_NAME.to_string()];

            crate::utils::opencl::ensure_program(
                opencl_state,
                PROGRAM_NAME.to_string(),
                PROGRAM_SOURCE.to_string(),
                "".to_string(),
                kernels,
            )?;

            Ok(())
        }

        use opencl3::memory::ClMem;

        impl<'a> crate::layers::Layer<'a, crate::layers::NoGradients<'a>> for #activation_name<'a> {
            fn init(
                &mut self,
                opencl_state: &'a crate::utils::OpenCLState,
            ) -> Result<(), opencl3::error_codes::ClError> {
                self.opencl_state = Some(opencl_state);

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

            fn sync_data_from_buffers_to_host(&mut self) -> Result<(), opencl3::error_codes::ClError> {
                Ok(())
            }

            fn propagate(&mut self, inputs: &opencl3::memory::Buffer<opencl3::device::cl_float>) -> Result<&opencl3::memory::Buffer<opencl3::device::cl_float>, opencl3::error_codes::ClError> {
                assert!(self.opencl_state.is_some());

                let state = self.opencl_state.unwrap();
                let context = &state.context;
                let queue = state.queues.first().unwrap();

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
                    )?.wait()?;

                self.last_inputs_buffer = Some(copied_last_inputs_buffer);

                let outputs_total_count = inputs.size()? / std::mem::size_of::<opencl3::device::cl_float>();

                let outputs_buffer = opencl3::memory::Buffer::<opencl3::device::cl_float>::create(
                    context,
                    opencl3::memory::CL_MEM_READ_WRITE,
                    outputs_total_count,
                    std::ptr::null_mut(),
                )?;

                let propagate_kernel = state.programs
                    .get(PROGRAM_NAME)
                    .unwrap()
                    .kernels
                    .get(PROPAGATE_KERNEL_NAME)
                    .unwrap();

                opencl3::kernel::ExecuteKernel::new(propagate_kernel)
                    .set_arg(inputs)
                    .set_arg(&outputs_buffer)
                    .set_arg(&(outputs_total_count as opencl3::error_codes::cl_int))
                    .set_global_work_size(outputs_total_count)
                    .enqueue_nd_range(queue)?;

                queue.finish()?;

                self.last_outputs_buffer = Some(outputs_buffer);

                Ok(self.last_outputs_buffer.as_ref().unwrap())
            }

            fn compute_gradients(
                &self,
                _: &opencl3::memory::Buffer<opencl3::device::cl_float>,
            ) -> Result<crate::layers::NoGradients<'a>, crate::layers::LayerGradientComputationError> {
                Ok(crate::layers::NoGradients)
            }

            fn apply_gradients(
                &mut self,
                _per_parameter_type_gradients: crate::layers::NoGradients,
                _optimizer: dyn crate::optimizers::Optimizer,
            ) -> Result<(), crate::layers::LayerGradientApplicationError> {
                Ok(())
            }

            fn compute_loss_to_input_derivatives(
                &mut self,
                layer_output_to_error_derivative: &opencl3::memory::Buffer<opencl3::device::cl_float>,
            ) -> Result<opencl3::memory::Buffer<opencl3::device::cl_float>, crate::layers::LayerLossToInputDifferentiationError> {
                if self.opencl_state.is_none() {
                    return Err(crate::layers::LayerLossToInputDifferentiationError::LayerNotInitializedError);
                }

                let state = self.opencl_state.unwrap();

                let context = &state.context;

                if state.queues.len() == 0 {
                    return Err(crate::layers::LayerLossToInputDifferentiationError::NoCommandQueue);
                }

                let queue = state.queues.first().unwrap();

                if self.last_outputs_buffer.is_none() {
                    return Err(crate::layers::LayerLossToInputDifferentiationError::HasNotPropagatedBeforeCalculation);
                }

                let samples_amount = self.last_outputs_buffer.as_ref().unwrap().size()?
                    / self.inputs_amount
                    / std::mem::size_of::<opencl3::device::cl_float>();

                let loss_to_input_derivatives_buffer = opencl3::memory::Buffer::<opencl3::device::cl_float>::create(
                    context,
                    opencl3::memory::CL_MEM_READ_WRITE,
                    self.inputs_amount * samples_amount,
                    std::ptr::null_mut(),
                )?;

                if !state.programs.contains_key(PROGRAM_NAME) {
                    return Err(crate::layers::LayerLossToInputDifferentiationError::ProgramNotFound(PROGRAM_NAME));
                }

                let program = state.programs.get(PROGRAM_NAME).unwrap();

                if !program.kernels.contains_key(BACK_PROPAGATE_KERNEL_NAME) {
                    return Err(crate::layers::LayerLossToInputDifferentiationError::KernelNotFound(BACK_PROPAGATE_KERNEL_NAME));
                }

                let back_prop_kernel = program.kernels
                    .get(BACK_PROPAGATE_KERNEL_NAME)
                    .unwrap();

                opencl3::kernel::ExecuteKernel::new(back_prop_kernel)
                    .set_arg(layer_output_to_error_derivative)
                    .set_arg(self.last_outputs_buffer.as_ref().unwrap())
                    .set_arg(&loss_to_input_derivatives_buffer)
                    .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
                    .set_arg(&(samples_amount as opencl3::error_codes::cl_int))
                    .set_arg(&(self.inputs_amount as opencl3::error_codes::cl_int))
                    .set_global_work_sizes(&[samples_amount, self.inputs_amount])
                    .enqueue_nd_range(queue)?;

                queue.finish()?;

                Ok(loss_to_input_derivatives_buffer)
            }
        }
    })
}