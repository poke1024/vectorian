import ctypes
dyld = ctypes.cdll.LoadLibrary('/usr/lib/system/libdyld.dylib')
namelen = ctypes.c_ulong(1024)
name = ctypes.create_string_buffer(b'\000', namelen.value)
dyld._NSGetExecutablePath(ctypes.byref(name), ctypes.byref(namelen))
print(name.value)
