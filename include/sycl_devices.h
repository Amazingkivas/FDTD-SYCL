#pragma once

#include <CL/sycl.hpp>

namespace FDTD_sycl {

    template <typename T>
    using sycl_vector = std::vector<T, sycl::usm_allocator<T, sycl::usm::alloc::shared>>;

    class AllDevices
    {
    public:
        AllDevices() : default_device{ sycl::default_selector{} },
            cpu_device{ sycl::cpu_selector{}, sycl::async_handler{} }                
        {
            try {
                gpu_device = sycl::queue { sycl::gpu_selector{}, sycl::async_handler{} };
            } catch (sycl::exception const& e) {
                gpu_device = sycl::queue { sycl::cpu_selector{}, sycl::async_handler{} };
            }
        };
        sycl::queue default_device;
        sycl::queue cpu_device;
        sycl::queue gpu_device;
    };

    extern AllDevices node;

} // namespace FDTD_sycl
