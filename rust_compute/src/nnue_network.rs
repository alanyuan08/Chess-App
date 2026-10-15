use std::fs::File;
use std::io::{Read, BufReader};

#[cfg_attr(target_arch = "aarch64", repr(C, align(128)))]
#[cfg_attr(not(target_arch = "aarch64"), repr(C, align(64)))]
#[derive(Clone, Copy, Debug)]
// The NNUuE L1 / L2 and L3 weights should align 128 to avoid expensive cache splits
pub struct NnueNetwork {
    // i16 -> 2 bytes. 2*256 / 128 -> 4 Cache Lines * 49152
    // i32 -> 4 bytes. 4*256 / 128 -> 8 Cache Lines
    pub l1_weights: [[i16; 256]; 49152],
    pub l1_biases: [i32; 256],
    
    // i8-> 1 bytes. 512 / 128 -> 4 Cache Lines * 32
    // i32 -> 4 bytes. 4*32 / 128 -> 1 Cache Lines
    pub l2_weights: [[i8; 512]; 32],
    pub l2_biases: [i32; 32],

    // i8-> 1 bytes. 32 * 32 / 128 -> 8 Cache Lines
    // i32 -> 4 bytes. 4*32 / 128 -> 1 Cache Lines
    pub l3_weights: [[i8; 32]; 32],
    pub l3_biases: [i32; 32],

    // i8-> 1 bytes. 32 + 4 bytes + 92 padding -> 1 Cache Line
    pub output_weights: [[i8; 32]; 1],
    pub output_bias: [i32; 1],
    
    // If aarch64 (Mac), use 92 bytes of padding to round up to 128.
    // If x86 (Intel/AMD), use 28 bytes of padding to round up to 64.
    #[cfg(target_arch = "aarch64")] 
    pub _padding: [u8; 92],
    
    #[cfg(not(target_arch = "aarch64"))] 
    pub _padding: [u8; 28],
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
            read_field(&mut network_box.l1_biases as *mut _ as *mut u8, 256, 4)?;

            // 2. Hidden Layer 2
            read_field(&mut network_box.l2_weights as *mut _ as *mut u8, 32 * 512, 1)?;
            read_field(&mut network_box.l2_biases as *mut _ as *mut u8, 32, 4)?;

            // 3. Hidden Layer 3
            read_field(&mut network_box.l3_weights as *mut _ as *mut u8, 32 * 32, 1)?;
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
#[cfg_attr(target_arch = "aarch64", repr(C, align(128)))]
#[cfg_attr(not(target_arch = "aarch64"), repr(C, align(64)))]
#[derive(Clone, Debug)]
pub struct NnueInferenceBuffer {
    // 4 * 128 -> 512
    pub l2_inputs: [i8; 512],
    // 32 + 32 + 64 pading -> 128
    pub l3_inputs: [i8; 32],
    pub l4_inputs: [i8; 32],

    #[cfg(target_arch = "aarch64")] 
     _padding: [u8; 64],
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
            l2_inputs: [0i8; 512],
            l3_inputs: [0i8; 32],
            l4_inputs: [0i8; 32],

            #[cfg(target_arch = "aarch64")]
            _padding: [0u8; 64],
        }
    }
}