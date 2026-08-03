use std::fs::File;
use std::io::Read;
use std::mem;
use core::arch::aarch64::*;

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
    /// Computes the forward evaluation pass using Apple Silicon NEON SIMD intrinsics.
    /// Perfectly tailored for maximum execution speed on an Apple M4 Pro.
    pub fn evaluate(
        &self, 
        white_acc: &[i16; 256], 
        black_acc: &[i16; 256], 
        buffer: &mut NnueInferenceBuffer
    ) -> i32 {
        unsafe {
            // -----------------------------------------------------------------
            // STEP 1: CONCATENATION & ACTIVATION (L1 -> L2) - NEON Optimized
            // -----------------------------------------------------------------
            // Process 16 inputs at a time (128 bits / 8 bits = 16 elements).
            // Clamp values between 0 and 128 natively using vector operations.
            let zero_v = vdupq_n_s16(0);
            let cap_v = vdupq_n_s16(128);

            for i in (0..128).step_by(16) {
                // White Accumulator Side (first 128 inputs)
                let w_low = vld1q_s16(white_acc[i..].as_ptr());
                let w_high = vld1q_s16(white_acc[i + 8..].as_ptr());
                
                let w_low_clamped = vminq_s16(vmaxq_s16(w_low, zero_v), cap_v);
                let w_high_clamped = vminq_s16(vmaxq_s16(w_high, zero_v), cap_v);
                
                // Narrow 16-bit integers down to 8-bit packed integers
                let w_narrow = vqmovn_s16(w_low_clamped);
                let w_combined = vqmovn_high_s16(w_narrow, w_high_clamped);
                vst1q_s8(buffer.l2_inputs[i..].as_mut_ptr(), w_combined);

                // Black Accumulator Side (remaining 128 inputs)
                let b_low = vld1q_s16(black_acc[i..].as_ptr());
                let b_high = vld1q_s16(black_acc[i + 8..].as_ptr());
                
                let b_low_clamped = vminq_s16(vmaxq_s16(b_low, zero_v), cap_v);
                let b_high_clamped = vminq_s16(vmaxq_s16(b_high, zero_v), cap_v);
                
                let b_narrow = vqmovn_s16(b_low_clamped);
                let b_combined = vqmovn_high_s16(b_narrow, b_high_clamped);
                vst1q_s8(buffer.l2_inputs[i + 128..].as_mut_ptr(), b_combined);
            }

            // -----------------------------------------------------------------
            // STEP 2: HIDDEN LAYER 2 (256 -> 64) - NEON Dot Product Vectorization
            // -----------------------------------------------------------------
            for neuron in 0..64 {
                // Initialize the vector sum register with the layer's 32-bit bias
                let mut sum_v = vdupq_n_s32(0);
                let row = &self.l2_weights[neuron];

                // Process 256 inputs in chunks of 16 elements (16 bytes per step)
                for i in (0..256).step_by(16) {
                    let in_v = vld1q_s8(buffer.l2_inputs[i..].as_ptr());
                    let w_v = vld1q_s8(row[i..].as_ptr());

                    // vdotq_s32 multiplies paired i8 values and accumulates them into 4x i32 lanes
                    sum_v = vdotq_s32(sum_v, in_v, w_v);
                }

                // Horizontally add the four 32-bit lane sums into a single scalar value
                let mut final_sum = vaddvq_s32(sum_v) + self.l2_biases[neuron];

                // Downscale by 128 (shift 7) and clamp between 0 and 32
                buffer.l3_inputs[neuron] = (final_sum >> 7).clamp(0, 32) as i8;
            }

            // -----------------------------------------------------------------
            // STEP 3: HIDDEN LAYER 3 (64 -> 32)
            // -----------------------------------------------------------------
            for neuron in 0..32 {
                let mut sum_v = vdupq_n_s32(0);
                let row = &self.l3_weights[neuron];

                // Process 64 inputs in chunks of 16 elements
                for i in (0..64).step_by(16) {
                    let in_v = vld1q_s8(buffer.l3_inputs[i..].as_ptr());
                    let w_v = vld1q_s8(row[i..].as_ptr());

                    sum_v = vdotq_s32(sum_v, in_v, w_v);
                }

                let mut final_sum = vaddvq_s32(sum_v) + self.l3_biases[neuron];

                // Downscale by 32 (shift 5) and clamp between 0 and 127
                buffer.l4_inputs[neuron] = (final_sum >> 5).clamp(0, 127) as i8;
            }

            // -----------------------------------------------------------------
            // STEP 4: OUTPUT LAYER (32 -> 1)
            // -----------------------------------------------------------------
            let mut sum_v = vdupq_n_s32(0);
            let row = &self.output_weights[0];

            // Process 32 inputs in two 16-element chunks
            for i in (0..32).step_by(16) {
                let in_v = vld1q_s8(buffer.l4_inputs[i..].as_ptr());
                let w_v = vld1q_s8(row[i..].as_ptr());

                sum_v = vdotq_s32(sum_v, in_v, w_v);
            }

            let final_sum = vaddvq_s32(sum_v) + self.output_bias[0];

            // Final normalization back to centipawns (shift right by 8)
            final_sum >> 8
        }
    }
}