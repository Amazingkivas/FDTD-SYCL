#include "FDTD_sycl.h"

#include <iostream>
#include <cstddef>
#include <sycl/CL/sycl.hpp>
#include <stdexcept>

namespace FDTD_sycl {

AllDevices node;

FDTD::FDTD(Parameters _parameters, FP _dt, DevicePreference preference)
    : parameters(_parameters), dt(_dt),
      Ni(_parameters.Ni), Nj(_parameters.Nj), Nk(_parameters.Nk),
      total_size(static_cast<std::size_t>(Ni) * static_cast<std::size_t>(Nj) * static_cast<std::size_t>(Nk)),
      grid_range(_parameters.Nk, _parameters.Nj, _parameters.Ni),
      flat_range(total_size),
      Jx(total_size),
      Jy(total_size),
      Jz(total_size),
      Ex(total_size),
      Ey(total_size),
      Ez(total_size),
      Bx(total_size),
      By(total_size),
      Bz(total_size),
      bufJx(Jx.data(), flat_range),
      bufJy(Jy.data(), flat_range),
      bufJz(Jz.data(), flat_range),
      bufEx(Ex.data(), flat_range),
      bufEy(Ey.data(), flat_range),
      bufEz(Ez.data(), flat_range),
      bufBx(Bx.data(), flat_range),
      bufBy(By.data(), flat_range),
      bufBz(Bz.data(), flat_range),
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
            const int k = static_cast<int>(item[0]);
            const int j = static_cast<int>(item[1]);
            const int i = static_cast<int>(item[2]);
            const std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(Nj) * static_cast<std::size_t>(Ni)
                                  + static_cast<std::size_t>(j) * static_cast<std::size_t>(Ni)
                                  + static_cast<std::size_t>(i);
            Ex_acc[idx] = 0.0;
            Ey_acc[idx] = 0.0;
            Ez_acc[idx] = 0.0;
            Bx_acc[idx] = 0.0;
            By_acc[idx] = 0.0;
            Bz_acc[idx] = 0.0;
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
            const int k = static_cast<int>(item[0]);
            const int j = static_cast<int>(item[1]);
            const int i = static_cast<int>(item[2]);
            const std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(Nj) * static_cast<std::size_t>(Ni)
                                  + static_cast<std::size_t>(j) * static_cast<std::size_t>(Ni)
                                  + static_cast<std::size_t>(i);
            Jx_acc[idx] = 0.0;
            Jy_acc[idx] = 0.0;
            Jz_acc[idx] = 0.0;
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
            const int k = static_cast<int>(id[0]);
            const int j = static_cast<int>(id[1]);
            const int i = static_cast<int>(id[2]);

            const int k_next = applyPeriodic(k + 1, l_Nk);
            const int j_next = applyPeriodic(j + 1, l_Nj);
            const int i_next = applyPeriodic(i + 1, l_Ni);

            const std::size_t idx = static_cast<std::size_t>(k) * l_Nj * l_Ni + static_cast<std::size_t>(j) * l_Ni + static_cast<std::size_t>(i);
            const std::size_t idx_kn = static_cast<std::size_t>(k_next) * l_Nj * l_Ni + static_cast<std::size_t>(j) * l_Ni + static_cast<std::size_t>(i);
            const std::size_t idx_jn = static_cast<std::size_t>(k) * l_Nj * l_Ni + static_cast<std::size_t>(j_next) * l_Ni + static_cast<std::size_t>(i);
            const std::size_t idx_in = static_cast<std::size_t>(k) * l_Nj * l_Ni + static_cast<std::size_t>(j) * l_Ni + static_cast<std::size_t>(i_next);

            Bx_acc[idx] += l_coef_B_dz * (Ey_acc[idx_kn] - Ey_acc[idx])
                         - l_coef_B_dy * (Ez_acc[idx_jn] - Ez_acc[idx]);
            By_acc[idx] += l_coef_B_dx * (Ez_acc[idx_in] - Ez_acc[idx])
                         - l_coef_B_dz * (Ex_acc[idx_kn] - Ex_acc[idx]);
            Bz_acc[idx] += l_coef_B_dy * (Ex_acc[idx_jn] - Ex_acc[idx])
                         - l_coef_B_dx * (Ey_acc[idx_in] - Ey_acc[idx]);
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
            const int k = static_cast<int>(id[0]);
            const int j = static_cast<int>(id[1]);
            const int i = static_cast<int>(id[2]);

            const int k_pred = applyPeriodic(k - 1, l_Nk);
            const int j_pred = applyPeriodic(j - 1, l_Nj);
            const int i_pred = applyPeriodic(i - 1, l_Ni);

            const std::size_t idx = static_cast<std::size_t>(k) * l_Nj * l_Ni + static_cast<std::size_t>(j) * l_Ni + static_cast<std::size_t>(i);
            const std::size_t idx_kp = static_cast<std::size_t>(k_pred) * l_Nj * l_Ni + static_cast<std::size_t>(j) * l_Ni + static_cast<std::size_t>(i);
            const std::size_t idx_jp = static_cast<std::size_t>(k) * l_Nj * l_Ni + static_cast<std::size_t>(j_pred) * l_Ni + static_cast<std::size_t>(i);
            const std::size_t idx_ip = static_cast<std::size_t>(k) * l_Nj * l_Ni + static_cast<std::size_t>(j) * l_Ni + static_cast<std::size_t>(i_pred);

            Ex_acc[idx] += l_cur_coef * Jx_acc[idx]
                         + l_coef_E_dy * (Bz_acc[idx] - Bz_acc[idx_jp])
                         - l_coef_E_dz * (By_acc[idx] - By_acc[idx_kp]);

            Ey_acc[idx] += l_cur_coef * Jy_acc[idx]
                         + l_coef_E_dz * (Bx_acc[idx] - Bx_acc[idx_kp])
                         - l_coef_E_dx * (Bz_acc[idx] - Bz_acc[idx_ip]);

            Ez_acc[idx] += l_cur_coef * Jz_acc[idx]
                         + l_coef_E_dx * (By_acc[idx] - By_acc[idx_ip])
                         - l_coef_E_dy * (Bx_acc[idx] - Bx_acc[idx_jp]);
        });
    };

    q.submit(kernel).wait_and_throw();
}

void FDTD::update_fields() {
    update_B();
    update_E();
    update_B();
}

sycl::buffer<FP, 1>& FDTD::get_field_buffer(Component this_field) {
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
