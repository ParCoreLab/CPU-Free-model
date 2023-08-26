add_subdirectory(jacobi2D)
add_subdirectory(jacobi3D)

add_custom_target(jacobi jacobi2D jacobi2D_nvshmem)
