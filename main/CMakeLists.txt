# Создаем исполняемый файл
add_executable(run_sycl main.cpp)

# Линкуем его с нашей библиотекой
target_link_libraries(run_sycl fdtd_sycl_lib)