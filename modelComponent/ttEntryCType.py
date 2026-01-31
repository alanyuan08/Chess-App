# Import Enum
import ctypes

# Inherit from Structure to make it a valid ctype
class TTEntryCType(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_int),
        ("score", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("flag", ctypes.c_int)
    ]