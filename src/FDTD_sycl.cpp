#include "FDTD_sycl.h"

#include <iostream>
#include <sycl/CL/sycl.hpp>
#include <stdexcept>

namespace FDTD_sycl {

AllDevices node;

FDTD::FDTD(Parameters _parameters, FP _dt, DevicePreference preference)
    : parameters(_parameters), dt(_dt),
      Ni(_parameters.Ni), Nj(_parameters.Nj), Nk(_parameters.Nk),
      grid_range(_parameters.Nk, _parameters.Nj, _parameters.Ni),
      Jx(Ni * Nj * Nk),
      Jy(Ni * Nj * Nk),
      Jz(Ni * Nj * Nk),
      Ex(Ni * Nj * Nk),
      Ey(Ni * Nj * Nk),
      Ez(Ni * Nj * Nk),
      Bx(Ni * Nj * Nk),
      By(Ni * Nj * Nk),
      Bz(Ni * Nj * Nk),
      bufJx(Jx.data(), grid_range),
      bufJy(Jy.data(), grid_range),
      bufJz(Jz.data(), grid_range),
      bufEx(Ex.data(), grid_range),
      bufEy(Ey.data(), grid_range),
      bufEz(Ez.data(), grid_range),
      bufBx(Bx.data(), grid_range),
      bufBy(By.data(), grid_range),
      bufBz(Bz.data(), grid_range),
      q(select_queue(preference)),
      device_preference(preference)
{
    if (Ni <= 0 || Nj <= 0 || Nk <= 0 || dt <= 0) {
        throw std::invalid_argument("ERROR: Invalid grid size (must be > 0).");
    }

    std::cout << "SYCL FDTD running on device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    const FP cdt = FDTD_const::C * dt;

    coef_E_dx = cdt / parameters.dx;
    coef_E_dy = cdt / parameters.dy;
    coef_E_dz = cdt / parameters.dz;

    dx = parameters.dx;
    dy = parameters.dy;
    dz = parameters.dz;

    coef_B_dx = cdt / (2.0 * dx);
    coef_B_dy = cdt / (2.0 * dy);
    coef_B_dz = cdt / (2.0 * dz);

    cur_coef = -4.0 * FDTD_const::PI * dt;

    zero_fields();
    zeroed_currents();
}

void FDTD::zero_fields() {
    auto kernel = [&](sycl::handler& h) {
        auto Ex_acc = bufEx.get_access<sycl::access::mode::write>(h);
        auto Ey_acc = bufEy.get_access<sycl::access::mode::write>(h);
        auto Ez_acc = bufEz.get_access<sycl::access::mode::write>(h);
        auto Bx_acc = bufBx.get_access<sycl::access::mode::write>(h);
        auto By_acc = bufBy.get_access<sycl::access::mode::write>(h);
        auto Bz_acc = bufBz.get_access<sycl::access::mode::write>(h);

        h.parallel_for(grid_range, [=](sycl::item<3> item) {
            int k = item[0];
            int j = item[1];
            int i = item[2];
            Ex_acc[k][j][i] = 0.0;
            Ey_acc[k][j][i] = 0.0;
            Ez_acc[k][j][i] = 0.0;
            Bx_acc[k][j][i] = 0.0;
            By_acc[k][j][i] = 0.0;
            Bz_acc[k][j][i] = 0.0;
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::zeroed_currents() {
    auto kernel = [&](sycl::handler& h) {
        auto Jx_acc = bufJx.get_access<sycl::access::mode::write>(h);
        auto Jy_acc = bufJy.get_access<sycl::access::mode::write>(h);
        auto Jz_acc = bufJz.get_access<sycl::access::mode::write>(h);

        h.parallel_for(grid_range, [=](sycl::item<3> item) {
            int k = item[0];
            int j = item[1];
            int i = item[2];
            Jx_acc[k][j][i] = 0.0;
            Jy_acc[k][j][i] = 0.0;
            Jz_acc[k][j][i] = 0.0;
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::update_B() {
    sycl::id<3> start(1, 1, 1);
    sycl::range<3> range(Nk - 2, Nj - 2, Ni - 2);

    const int l_Ni = Ni, l_Nj = Nj, l_Nk = Nk;
    const FP l_coef_B_dx = coef_B_dx, l_coef_B_dy = coef_B_dy, l_coef_B_dz = coef_B_dz;

    auto kernel = [&](sycl::handler& h) {
        auto Ex_acc = bufEx.get_access<sycl::access::mode::read>(h);
        auto Ey_acc = bufEy.get_access<sycl::access::mode::read>(h);
        auto Ez_acc = bufEz.get_access<sycl::access::mode::read>(h);

        auto Bx_acc = bufBx.get_access<sycl::access::mode::read_write>(h);
        auto By_acc = bufBy.get_access<sycl::access::mode::read_write>(h);
        auto Bz_acc = bufBz.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(range, start, [=](sycl::item<3> id) {
            int k = id[0], j = id[1], i = id[2];

            int k_next = applyPeriodic(k + 1, l_Nk);
            int j_next = applyPeriodic(j + 1, l_Nj);
            int i_next = applyPeriodic(i + 1, l_Ni);

            Bx_acc[k][j][i] += l_coef_B_dz * (Ey_acc[k_next][j][i] - Ey_acc[k][j][i])
                              - l_coef_B_dy * (Ez_acc[k][j_next][i] - Ez_acc[k][j][i]);
            By_acc[k][j][i] += l_coef_B_dx * (Ez_acc[k][j][i_next] - Ez_acc[k][j][i])
                              - l_coef_B_dz * (Ex_acc[k_next][j][i] - Ex_acc[k][j][i]);
            Bz_acc[k][j][i] += l_coef_B_dy * (Ex_acc[k][j_next][i] - Ex_acc[k][j][i])
                              - l_coef_B_dx * (Ey_acc[k][j][i_next] - Ey_acc[k][j][i]);
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::update_E() {
    sycl::id<3> start(1, 1, 1);
    sycl::range<3> range(Nk - 2, Nj - 2, Ni - 2);

    const int l_Ni = Ni, l_Nj = Nj, l_Nk = Nk;
    const FP l_cur_coef = cur_coef;
    const FP l_coef_E_dx = coef_E_dx, l_coef_E_dy = coef_E_dy, l_coef_E_dz = coef_E_dz;

    auto kernel = [&](sycl::handler& h) {
        auto Jx_acc = bufJx.get_access<sycl::access::mode::read>(h);
        auto Jy_acc = bufJy.get_access<sycl::access::mode::read>(h);
        auto Jz_acc = bufJz.get_access<sycl::access::mode::read>(h);
        auto Bx_acc = bufBx.get_access<sycl::access::mode::read>(h);
        auto By_acc = bufBy.get_access<sycl::access::mode::read>(h);
        auto Bz_acc = bufBz.get_access<sycl::access::mode::read>(h);
        auto Ex_acc = bufEx.get_access<sycl::access::mode::read_write>(h);
        auto Ey_acc = bufEy.get_access<sycl::access::mode::read_write>(h);
        auto Ez_acc = bufEz.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(range, start, [=](sycl::item<3> id) {
            int k = id[0], j = id[1], i = id[2];

            int k_pred = applyPeriodic(k - 1, l_Nk);
            int j_pred = applyPeriodic(j - 1, l_Nj);
            int i_pred = applyPeriodic(i - 1, l_Ni);

            Ex_acc[k][j][i] += l_cur_coef * Jx_acc[k][j][i]
                              + l_coef_E_dy * (Bz_acc[k][j][i] - Bz_acc[k][j_pred][i])
                              - l_coef_E_dz * (By_acc[k][j][i] - By_acc[k_pred][j][i]);

            Ey_acc[k][j][i] += l_cur_coef * Jy_acc[k][j][i]
                              + l_coef_E_dz * (Bx_acc[k][j][i] - Bx_acc[k_pred][j][i])
                              - l_coef_E_dx * (Bz_acc[k][j][i] - Bz_acc[k][j][i_pred]);

            Ez_acc[k][j][i] += l_cur_coef * Jz_acc[k][j][i]
                              + l_coef_E_dx * (By_acc[k][j][i] - By_acc[k][j][i_pred])
                              - l_coef_E_dy * (Bx_acc[k][j][i] - Bx_acc[k][j_pred][i]);
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::update_fields() {
    update_B();
    update_E();
    update_B();
}

sycl::buffer<FP, 3>& FDTD::get_field_buffer(Component this_field) {
    switch (this_field) {
        case Component::JX: return bufJx;
        case Component::JY: return bufJy;
        case Component::JZ: return bufJz;
        case Component::EX: return bufEx;
        case Component::EY: return bufEy;
        case Component::EZ: return bufEz;
        case Component::BX: return bufBx;
        case Component::BY: return bufBy;
        case Component::BZ: return bufBz;
        default: throw std::logic_error("ERROR: Invalid field component");
    }
}

} // namespace FDTD_sycl