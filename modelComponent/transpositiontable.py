

class TranspositionTable():

	TABLE_SIZE_MB = 16 
	ENTRY_SIZE_BYTES = 16 # Assuming 128-bit entry
	num_entries = (TABLE_SIZE_MB * 1024 * 1024) // ENTRY_SIZE_BYTES
	TT_SIZE = 1 << int(num_entries.bit_length() - 1)

	