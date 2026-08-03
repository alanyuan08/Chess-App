use std::fs::File;
use std::io::Read;
use std::mem;

/// Matches the exact memory layout of your exported Python binary.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct NnueNetwork {
    // Layer 1: Accumulator (49152 inputs -> 256 outputs)
    pub l1_weights: [[i16; 256]; 49152],
    pub l1_biases: [i16; 256],

    // Layer 2: Hidden 2 (256 inputs -> 64 outputs)
    pub l2_weights: [[i8; 256]; 64], // Transposed: row per output neuron
    pub l2_biases: [i32; 64],

    // Layer 3: Hidden 3 (64 inputs -> 32 outputs)
    pub l3_weights: [[i8; 64]; 32],  // Transposed: row per output neuron
    pub l3_biases: [i32; 32],

    // Layer 4: Output Layer (32 inputs -> 1 output)
    pub output_weights: [[i8; 32]; 1], // Transposed: row per output neuron
    pub output_bias: [i32; 1],
}

/// The runtime container holding the current calculation buffers.
/// Allocated once per search thread to prevent runtime overhead.
pub struct NnueInferenceBuffer {
    pub l2_inputs: [i8; 256],
    pub l3_inputs: [i8; 64],
    pub l4_inputs: [i8; 32],
}

pub fn load_network_file(path: &str) -> Box<NnueNetwork> {
    let mut file = File::open(path).expect("Could not find the NNUE file");
    
    // Allocate clean, zeroed memory on the Heap
    let mut network = unsafe {
        let layout = std::alloc::Layout::new::<NnueNetwork>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut NnueNetwork;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr)
    };

    // Calculate total byte size of your struct at compile time
    let total_bytes = mem::size_of::<NnueNetwork>();

    // Safely read the file data directly into our newly allocated heap structure
    unsafe {
        let network_bytes = std::slice::from_raw_parts_mut(
            &mut *network as *mut NnueNetwork as *mut u8, 
            total_bytes
        );
        
        file.read_exact(network_bytes).expect("Binary data size mismatch with NNUE struct!");
    }

    network
}

impl NnueNetwork {
    /// Computes the forward evaluation pass using the current accumulator states.
    /// Returns the final centipawn assessment.
    pub fn evaluate(
        &self, 
        white_acc: &[i16; 256], 
        black_acc: &[i16; 256], 
        buffer: &mut NnueInferenceBuffer
    ) -> i32 {
        
        // --- STEP 1: CONCATENATION & ACTIVATION (L1 -> L2) ---
        // Side-by-side perspective pooling using standard NNUE Clipped ReLU (CReLU).
        // Truncates values between 0.0 and 1.0 (0 and 128 in integer scale).
        for i in 0..128 {
            // White perspective feeds the first 128 inputs of Layer 2
            let w_val = white_acc[i].clamp(0, 128) as i8;
            buffer.l2_inputs[i] = w_val;

            // Black perspective feeds the remaining 128 inputs of Layer 2
            let b_val = black_acc[i].clamp(0, 128) as i8;
            buffer.l2_inputs[i + 128] = b_val;
        }

        // --- STEP 2: HIDDEN LAYER 2 (256 -> 64) ---
        // Inputs: Scale 128 (max value). Weights: Scale 32. 
        // Bias: Scale 4096 (128 * 32).
        for neuron in 0..64 {
            let mut sum: i32 = self.l2_biases[neuron];
            let row = &self.l2_weights[neuron];

            // Linear Dot Product
            for i in 0..256 {
                sum += (buffer.l2_inputs[i] as i32) * (row[i] as i32);
            }

            // Downscale via bit-shift right by 7 (divides by 128)
            // CReLU activation maps outputs up to 1.0 (which is 32 in our new scale)
            let activated = (sum >> 7).clamp(0, 32) as i8;
            buffer.l3_inputs[neuron] = activated;
        }

        // --- STEP 3: HIDDEN LAYER 3 (64 -> 32) ---
        // Inputs: Scale 32 (max value). Weights: Scale 32.
        // Bias: Scale 1024 (32 * 32).
        for neuron in 0..32 {
            let mut sum: i32 = self.l3_biases[neuron];
            let row = &self.l3_weights[neuron];

            for i in 0..64 {
                sum += (buffer.l3_inputs[i] as i32) * (row[i] as i32);
            }

            // Downscale via bit-shift right by 5 (divides by 32)
            // Activation scales up to 1.0 (which translates to 256 in standard output scales)
            let activated = (sum >> 5).clamp(0, 127) as i8; 
            buffer.l4_inputs[neuron] = activated;
        }

        // --- STEP 4: OUTPUT LAYER (32 -> 1) ---
        // Inputs: Scale 256. Weights: Scale 127.
        // Bias: Scale 32768.
        let mut final_sum: i32 = self.output_bias[0];
        let row = &self.output_weights[0];

        for i in 0..32 {
            final_sum += (buffer.l4_inputs[i] as i32) * (row[i] as i32);
        }

        // Final normalization back to centipawn score bounds
        // (Shift right by 8 divides by 256 to clean up the last scaling layer)
        final_sum >> 8
    }
}