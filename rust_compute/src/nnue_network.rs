use std::fs::File;
use std::io::{Read, BufReader};

/// Matches the exact memory layout of your exported Python binary.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct NnueNetwork {
    // Layer 1: Accumulator (49152 inputs -> 256 outputs)
    pub l1_weights: [[i16; 256]; 49152],
    pub l1_biases: [i32; 256],

    // Layer 2: Hidden 2 (512 inputs -> 64 outputs)
    pub l2_weights: [[i8; 512]; 64],
    pub l2_biases: [i32; 64],

    // Layer 3: Hidden 3 (64 inputs -> 32 outputs)
    pub l3_weights: [[i8; 64]; 32],
    pub l3_biases: [i32; 32],

    // Layer 4: Output Layer (32 inputs -> 1 output)
    pub output_weights: [[i8; 32]; 1],
    pub output_bias: [i32; 1]
}

impl NnueNetwork {
    pub fn load_from_file(path: &str) -> std::io::Result<Box<Self>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Safe heap allocation to avoid stack overflows
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

        // --- SAFE SEQUENTIAL FIELD READING ---
        // We get raw mutable byte slices for EACH field individually.
        // This naturally skips any hidden padding bytes the compiler inserts!
        unsafe {
            // Helper closure to map any typed slice directly to a mutable u8 byte slice
            let mut read_field = |ptr: *mut u8, elements_count: usize, element_size: usize| -> std::io::Result<()> {
                let byte_slice = std::slice::from_raw_parts_mut(ptr, elements_count * element_size);
                reader.read_exact(byte_slice)
            };

            // 1. Accumulator Layer (l1_weights is i16 = 2 bytes, l1_biases is i32 = 4 bytes)
            read_field(&mut network_box.l1_weights as *mut _ as *mut u8, 49152 * 256, 2)?;
            read_field(&mut network_box.l1_biases as *mut _ as *mut u8, 256, 4)?; // FIXED: element size is 4 for i32

            // 2. Hidden Layer 2
            read_field(&mut network_box.l2_weights as *mut _ as *mut u8, 64 * 512, 1)?;
            read_field(&mut network_box.l2_biases as *mut _ as *mut u8, 64, 4)?;

            // 3. Hidden Layer 3
            read_field(&mut network_box.l3_weights as *mut _ as *mut u8, 32 * 64, 1)?;
            read_field(&mut network_box.l3_biases as *mut _ as *mut u8, 32, 4)?;

            // 4. Output Layer
            read_field(&mut network_box.output_weights as *mut _ as *mut u8, 32, 1)?;
            read_field(&mut network_box.output_bias as *mut _ as *mut u8, 1, 4)?;
        }

        Ok(network_box)
    }
}

/// The runtime container holding the current calculation buffers.
/// Allocated once per search thread to prevent runtime overhead.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct NnueInferenceBuffer {
    pub l2_inputs: [i16; 512],
    pub l3_inputs: [i16; 64],
    pub l4_inputs: [i16; 32],
}

impl NnueInferenceBuffer {    
    /// Explicitly zeroes out all internal scratchpad arrays in-place.
    /// This prevents historical calculations from bleeding into a new evaluation.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.l2_inputs.fill(0);
        self.l3_inputs.fill(0);
        self.l4_inputs.fill(0);
    }
}

impl Default for NnueInferenceBuffer {
    fn default() -> Self {
        Self {
            l2_inputs: [0i16; 512],
            l3_inputs: [0i16; 64],
            l4_inputs: [0i16; 32],
        }
    }
}