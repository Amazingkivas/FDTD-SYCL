#pragma once

#include "sycl_shared.h"
#include <cstddef>

namespace FDTD_sycl {

class FDTD {
protected:
    Parameters parameters;

    int Ni, Nj, Nk;
    FP dx, dy, dz, dt;
    FP cur_coef;
    FP coef_E_dx, coef_E_dy, coef_E_dz;
    FP coef_B_dx, coef_B_dy, coef_B_dz;

    Field Ex, Ey, Ez;
    Field Bx, By, Bz;
    Field Jx, Jy, Jz;

    std::size_t total_size;
    sycl::range<3> grid_range;
    sycl::range<1> flat_range;

    sycl::buffer<FP, 1> bufBx, bufBy, bufBz;
    sycl::buffer<FP, 1> bufEx, bufEy, bufEz;
    sycl::buffer<FP, 1> bufJx, bufJy, bufJz;

    sycl::queue q;
    DevicePreference device_preference;

    inline std::size_t linear_index(int k, int j, int i) const {
        return static_cast<std::size_t>(k) * static_cast<std::size_t>(Nj) * static_cast<std::size_t>(Ni)
             + static_cast<std::size_t>(j) * static_cast<std::size_t>(Ni)
             + static_cast<std::size_t>(i);
    }

    void update_E();
    void update_B();
    void zero_fields();

public:
    FDTD(Parameters _parameters, FP _dt, DevicePreference preference = DevicePreference::AUTO);
    virtual ~FDTD() = default;

    sycl::buffer<FP, 1>& get_field_buffer(Component this_field);

    sycl::queue& get_queue() { return q; }

    virtual void update_fields();

    void zeroed_currents();
};

} // namespace FDTD_sycl
