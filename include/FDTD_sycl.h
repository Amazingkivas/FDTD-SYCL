#pragma once

#include "sycl_shared.h"

namespace FDTD_sycl {

template <typename T>
using sycl_vector = std::vector<T, sycl::usm_allocator<T, sycl::usm::alloc::shared>>;

class FDTD {
protected:
    Parameters parameters;

    int Ni, Nj, Nk;
    FP dx, dy, dz, dt;
    FP cur_coef;
    FP coef_E_dx, coef_E_dy, coef_E_dz;
    FP coef_B_dx, coef_B_dy, coef_B_dz;

    sycl_vector<FP> Ex, Ey, Ez;
    sycl_vector<FP> Bx, By, Bz;
    sycl_vector<FP> Jx, Jy, Jz;

    sycl::range<3> grid_range; 
    
    sycl::buffer<FP, 3> bufBx, bufBy, bufBz;
    sycl::buffer<FP, 3> bufEx, bufEy, bufEz;
    sycl::buffer<FP, 3> bufJx, bufJy, bufJz;

    sycl::queue q;
    
    void update_E();
    void update_B();
    void zero_fields();

public:
    FDTD(Parameters _parameters, FP _dt);
    virtual ~FDTD() = default;

    sycl::buffer<FP, 3>& get_field_buffer(Component this_field);
    
    sycl::queue& get_queue() { return q; }

    virtual void update_fields();

    void zeroed_currents();
};

} // namespace FDTD_sycl
