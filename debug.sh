# export DEBUG_VECTORIAN_FORCE_REBUILD=1
# env DYLD_FORCE_FLAT_NAMESPACE=1 DYLD_INSERT_LIBRARIES=/Library/Developer/CommandLineTools/usr/lib/clang/11.0.3/lib/darwin/libclang_rt.asan_osx_dynamic.dylib
export VECTORIAN_CPP_IMPORT=1
export DEBUG_VECTORIAN_CORE=1
export VECTORIAN_SANITIZE_ADDRESS=1
ASAN_OPTIONS=detect_container_overflow=0
lldb python -- main.py  # debug/contextual.py
