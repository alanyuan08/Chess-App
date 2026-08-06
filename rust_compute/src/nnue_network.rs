use std::fs::File;
use std::io::{Read, BufReader};

/// Matches the exact memory layout of your exported Python binary.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct NnueNetwork {
    // Layer 1: Accumulator (49152 inputs -> 256 outputs)
    pub l1_weights: [[i16; 256]; 49152],
    pub l1_biases: [i16; 256],

    // Layer 2: Hidden 2 (512 inputs -> 64 outputs)
    pub l2_weights: [[i8; 512]; 64], // Transposed: row per output neuron
    pub l2_biases: [i32; 64],

    // Layer 3: Hidden 3 (64 inputs -> 32 outputs)
    pub l3_weights: [[i8; 64]; 32],  // Transposed: row per output neuron
    pub l3_biases: [i32; 32],

    // Layer 4: Output Layer (32 inputs -> 1 output)
    pub output_weights: [[i8; 32]; 1], // Transposed: row per output neuron
    pub output_bias: [i32; 1]
}

impl NnueNetwork {
    pub fn load_from_file(path: &str) -> std::io::Result<Box<Self>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // --- PERFECT PACKED BYTE CALCULATION ---
        // We calculate the exact raw bytes written by Python, completely bypassing 
        // any padding added by the Rust compiler at the trailing end of the struct.
        let packed_data_size = 
            (12582912 * 2) + // l1_weights (i16)
            (256 * 2)      + // l1_biases  (i16)
            (64 * 512 * 1) + // l2_weights (i8)
            (64 * 4)       + // l2_biases  (i32)
            (32 * 64 * 1)  + // l3_weights (i8)
            (32 * 4)       + // l3_biases  (i32)
            (1 * 32 * 1)   + // output_weights (i8)
            (1 * 4);         // output_bias (i32)
            // Total = Exactly 25,201,572 bytes

        // Allocate a zeroed 25MB structure straight into the OS heap memory registry
        let mut network_box = unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "Failed to allocate heap space for NNUE",
                ));
            }
            Box::from_raw(ptr)
        };

        // Map ONLY the exact packed slice footprint instead of `std::mem::size_of::<Self>()`
        // This stops the file stream from requesting the 28 non-existent padding bytes.
        let data_slice = unsafe {
            std::slice::from_raw_parts_mut(
                &mut *network_box as *mut Self as *mut u8,
                packed_data_size,
            )
        };

        reader.read_exact(data_slice)?;
        Ok(network_box)
    }
}

/// The runtime container holding the current calculation buffers.
/// Allocated once per search thread to prevent runtime overhead.
#[repr(C, align(64))]
pub struct NnueInferenceBuffer {
    pub l2_inputs: [i8; 512],
    pub l3_inputs: [i8; 64],
    pub l4_inputs: [i8; 32],
}

impl NnueInferenceBuffer {
    /// Creates a zeroed instance on the stack or search stack allocation block
    pub fn new() -> Self {
        Self {
            l2_inputs: [0i8; 512],
            l3_inputs: [0i8; 64],
            l4_inputs: [0i8; 32],
        }
    }
}
