# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

add_kernels_compile()

add_ops_compile_options(ScatterMaxV1 OPTIONS --tiling_key=0,1)
