//! The module that contains some utilities for both compiling, ensuring and running
//! OpenCL programs.

use std::{collections::HashMap, ptr};

use intricate_macros::FromForAllUnnamedVariants;
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{
        get_all_devices, Device, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL,
        CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_CUSTOM, CL_DEVICE_TYPE_GPU,
    },
    error_codes::ClError,
    kernel::Kernel,
    program::Program,
    types::cl_device_type,
};

use crate::{
    layers::compile_layers,
    loss_functions::compile_losses,
    model::compile_model,
    types::{KernelNotFoundError, ProgramNotFoundError},
};

use super::{
    buffer_operations::compile_buffer_operations_program,
    inplace_buffer_operations::compile_inplace_buffer_operations_program,
};

#[derive(Debug)]
/// The state that contains useful OpenCL information that is necessary to keep track of the
/// compilled OpenCL programs and kernels.
pub struct OpenCLState {
    /// OpenCL's Context object that contains some useful information
    pub context: Context,
    /// A vec containing the corresponding Command Queue's for each one of the devices
    pub queues: Vec<CommandQueue>,
    /// A vec containing all of the devices that were found by OpenCL for a ceratin **DeviceType**
    pub devices: Vec<Device>,
    /// A HashMap where the key is the name of the program and value is a struct that contains both
    /// the original OpenCL program and another HashMap with all of the kernels.
    pub programs: HashMap<&'static str, IntricateProgram>,
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An error that happens when the `setup_opencl` function fails.
#[allow(missing_docs)]
pub enum UnableToSetupOpenCLError {
    OpenCL(ClError),
    CompilationErrors(EnsureKernelsAndProgramError),
    NoDeviceFound,
}

#[derive(Debug)]
/// Just a struct that contains both the original OpenCL program and a HashMap containing all of
/// the kernels with their keys as just the names of the kernels and the values as the actual
/// kernels.
pub struct IntricateProgram {
    /// The original OpenCL Program struct.
    pub opencl_program: Program,
    /// A HashMap with the keys as the names of the kernels and the values as the original OpenCL
    /// structs that represent the kernels.
    pub kernels: HashMap<String, Kernel>,
}

impl IntricateProgram {
    /// Safely gets the kernel by name inside of the program.
    pub fn get_krnl(&self, kernel_name: &str) -> Result<&Kernel, KernelNotFoundError> {
        if !self.kernels.contains_key(&kernel_name.to_string()) {
            Err(kernel_name.to_string().into())
        } else {
            Ok(self.kernels.get(&kernel_name.to_string()).unwrap())
        }
    }
}

impl OpenCLState {
    /// Safely gets a program by name inside of the OpenCLState.
    pub fn get_prgm(&self, program_name: &str) -> Result<&IntricateProgram, ProgramNotFoundError> {
        if !self.programs.contains_key(program_name) {
            Err(program_name.to_string().into())
        } else {
            Ok(self.programs.get(program_name).unwrap())
        }
    }
}

#[derive(Debug, FromForAllUnnamedVariants)]
/// An error that happens in the `ensure_program` function, if either the compilation goes wrong of
/// the program or one of the kernels could not be found inside of the program being compiled.
#[allow(missing_docs)]
pub enum EnsureKernelsAndProgramError {
    OpenCL(ClError),
    /// An error that will occur when something goes wrong in kernel compilation
    /// returning a tuple with the error in the code itself and the name of the program
    /// in which it failed
    Compilation(String, String),
}

/// Will compile all of the kernels listed in **kernel_names** inside of the
/// program with source **program_source**, with the options **compile_options**
/// and will insert that program as well with the kernels inside of the **opencl_state**
/// for later usage.
///
/// # Errors
///
/// - Will yield an error if the compilation goes wrong.
/// - Will yield an error if a specified kernel could not be found inside of the program's source.
pub(crate) fn ensure_program(
    opencl_state: &mut OpenCLState,
    program_name: &'static str,
    program_source: &'static str,
    compile_options: &'static str,
    kernel_names: &[&'static str],
) -> Result<(), EnsureKernelsAndProgramError> {
    let context = &opencl_state.context;

    if !opencl_state.programs.contains_key(program_name) {
        let cl_program_result =
            Program::create_and_build_from_source(context, program_source, &compile_options);
        if let Ok(new_cl_program) = cl_program_result {
            opencl_state.programs.insert(
                program_name.clone(),
                IntricateProgram {
                    opencl_program: new_cl_program,
                    kernels: HashMap::default(),
                },
            );
        } else {
            return Err(EnsureKernelsAndProgramError::Compilation(
                cl_program_result.err().unwrap(),
                program_name.to_string(),
            ));
        }
    }

    let program = opencl_state.programs.get_mut(program_name).unwrap();

    for kernel_name in kernel_names.iter() {
        let string_kernel_name = kernel_name.to_string();
        if !program.kernels.contains_key(&string_kernel_name) {
            let kernel = Kernel::create(&program.opencl_program, kernel_name)?;
            program.kernels.insert(string_kernel_name, kernel);
        }
    }

    Ok(())
}

#[derive(Debug)]
/// A enum used for telling Intriate what type of device it should try using with OpenCL.
pub enum DeviceType {
    /// Just the normal and usual **Graphics Processing Unit**
    GPU = CL_DEVICE_TYPE_GPU as isize,
    /// The **Central Processing Unit**
    CPU = CL_DEVICE_TYPE_CPU as isize,
    /// This will allow all types, and in turn, as of v0.3.0, will just get the first device
    /// it is able to find in your computer
    ALL = CL_DEVICE_TYPE_ALL as isize,
    /// A custom device, you can write custom drivers and use them in here if you have them in your
    /// computer.
    CUSTOM = CL_DEVICE_TYPE_CUSTOM as isize,
    #[allow(missing_docs)]
    ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR as isize,
}

/// Finds all of the devices of a certain **device_type**, starts a context for all of the devices,
/// and creates a CommandQueue for each one of the devices.
/// Also will compile some basic Intricate programs after setting up.
///
/// # Errors
///
/// Will return an NoDeviceFound error if it could not find any device of the specified type, or
/// will return an ClError with the respective OpenCL error code if something goes wrong while
/// creating the context or the queues.
pub fn setup_opencl(device_type: DeviceType) -> Result<OpenCLState, UnableToSetupOpenCLError> {
    let device_ids = get_all_devices(device_type as cl_device_type)?;
    if !&device_ids.is_empty() {
        let devices: Vec<Device> = device_ids.iter().map(|id| Device::new(*id)).collect();
        let context = Context::from_devices(&device_ids, &[], None, ptr::null_mut())?;

        // here it can be activated to make profiling on kernels
        let queues: Vec<CommandQueue> = devices
            .iter()
            .map(|device| CommandQueue::create_with_properties(&context, device.id(), 0, 0))
            .collect::<Result<Vec<CommandQueue>, ClError>>()?;

        let mut state = OpenCLState {
            context,
            queues,
            devices,
            programs: HashMap::default(),
        };

        compile_buffer_operations_program(&mut state)?;

        compile_inplace_buffer_operations_program(&mut state)?;

        compile_layers(&mut state)?;

        compile_model(&mut state)?;

        compile_losses(&mut state)?;

        Ok(state)
    } else {
        Err(UnableToSetupOpenCLError::NoDeviceFound)
    }
}