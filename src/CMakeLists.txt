# Находим все .cpp файлы в этой папке
file(GLOB srcs "*.cpp")

# Создаем библиотеку
add_library(fdtd_sycl_lib STATIC ${srcs})