/*
Монолитный FDTD-SYCL пример (только функции, без классов).

Сборка (без CMake):
  clang++ -std=c++17 -O2 -fsycl fdtd_sycl_monolith.cpp -o fdtd_sycl

Запуск:
  ./fdtd_sycl
  ./fdtd_sycl 256 25 cpu
  ./fdtd_sycl 256 25 gpu
*/

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using FP = double;

namespace cst {
constexpr FP PI = 3.14159265358979323846;
constexpr FP C = 0.5;
}

enum class DevicePreference { AUTO, CPU, GPU };
enum class Component { EX, EY, EZ, BX, BY, BZ, JX, JY, JZ };

struct Parameters {
  int Ni, Nj, Nk;
  FP ax, bx;
  FP ay, by;
  FP az, bz;
  FP dx, dy, dz;
};

struct CurrentParameters {
  FP period;
  FP period_x;
  FP dt;
  int iterations = 0;
  FP period_y = period_x;
  FP period_z = period_x;
};

struct Buffers {
  std::vector<FP> Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz;
  sycl::buffer<FP, 1> bufEx;
  sycl::buffer<FP, 1> bufEy;
  sycl::buffer<FP, 1> bufEz;
  sycl::buffer<FP, 1> bufBx;
  sycl::buffer<FP, 1> bufBy;
  sycl::buffer<FP, 1> bufBz;
  sycl::buffer<FP, 1> bufJx;
  sycl::buffer<FP, 1> bufJy;
  sycl::buffer<FP, 1> bufJz;

  explicit Buffers(std::size_t total)
      : Ex(total), Ey(total), Ez(total),
        Bx(total), By(total), Bz(total),
        Jx(total), Jy(total), Jz(total),
        bufEx(Ex.data(), sycl::range<1>(total)),
        bufEy(Ey.data(), sycl::range<1>(total)),
        bufEz(Ez.data(), sycl::range<1>(total)),
        bufBx(Bx.data(), sycl::range<1>(total)),
        bufBy(By.data(), sycl::range<1>(total)),
        bufBz(Bz.data(), sycl::range<1>(total)),
        bufJx(Jx.data(), sycl::range<1>(total)),
        bufJy(Jy.data(), sycl::range<1>(total)),
        bufJz(Jz.data(), sycl::range<1>(total)) {}
};

int apply_periodic(int i, int N) {
  if (i < 0) return N - 1;
  if (i == N) return 0;
  return i;
}

std::size_t index3d(int k, int j, int i, int Nj, int Ni) {
  return static_cast<std::size_t>(k) * static_cast<std::size_t>(Nj) * static_cast<std::size_t>(Ni) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(Ni) +
         static_cast<std::size_t>(i);
}

sycl::queue make_queue(DevicePreference pref) {
  if (pref == DevicePreference::CPU) return sycl::queue{sycl::cpu_selector_v};
  if (pref == DevicePreference::GPU) return sycl::queue{sycl::gpu_selector_v};
  try {
    return sycl::queue{sycl::gpu_selector_v};
  } catch (...) {
    return sycl::queue{sycl::cpu_selector_v};
  }
}

DevicePreference parse_device(const std::string& s) {
  if (s == "auto") return DevicePreference::AUTO;
  if (s == "cpu") return DevicePreference::CPU;
  if (s == "gpu") return DevicePreference::GPU;
  throw std::invalid_argument("device must be auto|cpu|gpu");
}

sycl::buffer<FP, 1>& field_buffer(Buffers& b, Component c) {
  switch (c) {
    case Component::EX: return b.bufEx;
    case Component::EY: return b.bufEy;
    case Component::EZ: return b.bufEz;
    case Component::BX: return b.bufBx;
    case Component::BY: return b.bufBy;
    case Component::BZ: return b.bufBz;
    case Component::JX: return b.bufJx;
    case Component::JY: return b.bufJy;
    case Component::JZ: return b.bufJz;
  }
  throw std::logic_error("invalid component");
}

void zero_fields(sycl::queue& q, Buffers& b, int Nk, int Nj, int Ni) {
  sycl::range<3> grid(Nk, Nj, Ni);
  q.submit([&](sycl::handler& h) {
    auto Ex = b.bufEx.get_access<sycl::access::mode::write>(h);
    auto Ey = b.bufEy.get_access<sycl::access::mode::write>(h);
    auto Ez = b.bufEz.get_access<sycl::access::mode::write>(h);
    auto Bx = b.bufBx.get_access<sycl::access::mode::write>(h);
    auto By = b.bufBy.get_access<sycl::access::mode::write>(h);
    auto Bz = b.bufBz.get_access<sycl::access::mode::write>(h);

    h.parallel_for(grid, [=](sycl::id<3> id) {
      const std::size_t idx = index3d(id[0], id[1], id[2], Nj, Ni);
      Ex[idx] = 0.0;
      Ey[idx] = 0.0;
      Ez[idx] = 0.0;
      Bx[idx] = 0.0;
      By[idx] = 0.0;
      Bz[idx] = 0.0;
    });
  }).wait_and_throw();
}

void zero_currents(sycl::queue& q, Buffers& b, int Nk, int Nj, int Ni) {
  sycl::range<3> grid(Nk, Nj, Ni);
  q.submit([&](sycl::handler& h) {
    auto Jx = b.bufJx.get_access<sycl::access::mode::write>(h);
    auto Jy = b.bufJy.get_access<sycl::access::mode::write>(h);
    auto Jz = b.bufJz.get_access<sycl::access::mode::write>(h);

    h.parallel_for(grid, [=](sycl::id<3> id) {
      const std::size_t idx = index3d(id[0], id[1], id[2], Nj, Ni);
      Jx[idx] = 0.0;
      Jy[idx] = 0.0;
      Jz[idx] = 0.0;
    });
  }).wait_and_throw();
}

void update_B(sycl::queue& q, Buffers& b, int Nk, int Nj, int Ni,
              FP coef_B_dx, FP coef_B_dy, FP coef_B_dz) {
  sycl::range<3> range(Nk - 2, Nj - 2, Ni - 2);
  sycl::id<3> offset(1, 1, 1);

  q.submit([&](sycl::handler& h) {
    auto Ex = b.bufEx.get_access<sycl::access::mode::read>(h);
    auto Ey = b.bufEy.get_access<sycl::access::mode::read>(h);
    auto Ez = b.bufEz.get_access<sycl::access::mode::read>(h);
    auto Bx = b.bufBx.get_access<sycl::access::mode::read_write>(h);
    auto By = b.bufBy.get_access<sycl::access::mode::read_write>(h);
    auto Bz = b.bufBz.get_access<sycl::access::mode::read_write>(h);

    h.parallel_for(range, offset, [=](sycl::id<3> id) {
      const int k = id[0], j = id[1], i = id[2];
      const int kn = apply_periodic(k + 1, Nk);
      const int jn = apply_periodic(j + 1, Nj);
      const int in = apply_periodic(i + 1, Ni);

      const std::size_t idx = index3d(k, j, i, Nj, Ni);
      const std::size_t idx_kn = index3d(kn, j, i, Nj, Ni);
      const std::size_t idx_jn = index3d(k, jn, i, Nj, Ni);
      const std::size_t idx_in = index3d(k, j, in, Nj, Ni);

      Bx[idx] += coef_B_dz * (Ey[idx_kn] - Ey[idx]) - coef_B_dy * (Ez[idx_jn] - Ez[idx]);
      By[idx] += coef_B_dx * (Ez[idx_in] - Ez[idx]) - coef_B_dz * (Ex[idx_kn] - Ex[idx]);
      Bz[idx] += coef_B_dy * (Ex[idx_jn] - Ex[idx]) - coef_B_dx * (Ey[idx_in] - Ey[idx]);
    });
  }).wait_and_throw();
}

void update_E(sycl::queue& q, Buffers& b, int Nk, int Nj, int Ni,
              FP cur_coef, FP coef_E_dx, FP coef_E_dy, FP coef_E_dz) {
  sycl::range<3> range(Nk - 2, Nj - 2, Ni - 2);
  sycl::id<3> offset(1, 1, 1);

  q.submit([&](sycl::handler& h) {
    auto Jx = b.bufJx.get_access<sycl::access::mode::read>(h);
    auto Jy = b.bufJy.get_access<sycl::access::mode::read>(h);
    auto Jz = b.bufJz.get_access<sycl::access::mode::read>(h);
    auto Bx = b.bufBx.get_access<sycl::access::mode::read>(h);
    auto By = b.bufBy.get_access<sycl::access::mode::read>(h);
    auto Bz = b.bufBz.get_access<sycl::access::mode::read>(h);
    auto Ex = b.bufEx.get_access<sycl::access::mode::read_write>(h);
    auto Ey = b.bufEy.get_access<sycl::access::mode::read_write>(h);
    auto Ez = b.bufEz.get_access<sycl::access::mode::read_write>(h);

    h.parallel_for(range, offset, [=](sycl::id<3> id) {
      const int k = id[0], j = id[1], i = id[2];
      const int kp = apply_periodic(k - 1, Nk);
      const int jp = apply_periodic(j - 1, Nj);
      const int ip = apply_periodic(i - 1, Ni);

      const std::size_t idx = index3d(k, j, i, Nj, Ni);
      const std::size_t idx_kp = index3d(kp, j, i, Nj, Ni);
      const std::size_t idx_jp = index3d(k, jp, i, Nj, Ni);
      const std::size_t idx_ip = index3d(k, j, ip, Nj, Ni);

      Ex[idx] += cur_coef * Jx[idx] + coef_E_dy * (Bz[idx] - Bz[idx_jp]) - coef_E_dz * (By[idx] - By[idx_kp]);
      Ey[idx] += cur_coef * Jy[idx] + coef_E_dz * (Bx[idx] - Bx[idx_kp]) - coef_E_dx * (Bz[idx] - Bz[idx_ip]);
      Ez[idx] += cur_coef * Jz[idx] + coef_E_dx * (By[idx] - By[idx_ip]) - coef_E_dy * (Bx[idx] - Bx[idx_jp]);
    });
  }).wait_and_throw();
}

void update_fields(sycl::queue& q, Buffers& b, int Nk, int Nj, int Ni,
                   FP cur_coef, FP coef_E_dx, FP coef_E_dy, FP coef_E_dz,
                   FP coef_B_dx, FP coef_B_dy, FP coef_B_dz) {
  update_B(q, b, Nk, Nj, Ni, coef_B_dx, coef_B_dy, coef_B_dz);
  update_E(q, b, Nk, Nj, Ni, cur_coef, coef_E_dx, coef_E_dy, coef_E_dz);
  update_B(q, b, Nk, Nj, Ni, coef_B_dx, coef_B_dy, coef_B_dz);
}

void run_fdtd(int n, int iterations, DevicePreference pref) {
  CurrentParameters cur{8.0, 4.0, 0.2};
  cur.iterations = static_cast<int>(cur.period / cur.dt);

  const FP d = cst::C;
  const FP boundary = static_cast<FP>(n) / 2.0 * d;
  Parameters p{n, n, n,
               -boundary, boundary,
               -boundary, boundary,
               -boundary, boundary,
               d, d, d};

  if (p.Ni <= 2 || p.Nj <= 2 || p.Nk <= 2) {
    throw std::invalid_argument("N must be > 2");
  }

  const FP cdt = cst::C * cur.dt;
  const FP coef_E_dx = cdt / p.dx;
  const FP coef_E_dy = cdt / p.dy;
  const FP coef_E_dz = cdt / p.dz;
  const FP coef_B_dx = cdt / (2.0 * p.dx);
  const FP coef_B_dy = cdt / (2.0 * p.dy);
  const FP coef_B_dz = cdt / (2.0 * p.dz);
  const FP cur_coef = -4.0 * cst::PI * cur.dt;

  sycl::queue q = make_queue(pref);
  std::cout << "Running on: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  Buffers b(static_cast<std::size_t>(p.Ni) * p.Nj * p.Nk);

  zero_fields(q, b, p.Nk, p.Nj, p.Ni);
  zero_currents(q, b, p.Nk, p.Nj, p.Ni);

  const FP T = cur.period;
  const FP Tx = cur.period_x;
  const FP Ty = cur.period_y;
  const FP Tz = cur.period_z;

  auto source = [=](FP x, FP y, FP z, FP t) {
    return std::sin(2.0 * cst::PI * t / T) *
           std::pow(std::cos(2.0 * cst::PI * x / Tx), 2.0) *
           std::pow(std::cos(2.0 * cst::PI * y / Ty), 2.0) *
           std::pow(std::cos(2.0 * cst::PI * z / Tz), 2.0);
  };

  const int cur_time = std::min(cur.iterations, iterations);
  const int start_i = static_cast<int>(std::floor((-Tx / 4.0 - p.ax) / p.dx));
  const int start_j = static_cast<int>(std::floor((-Ty / 4.0 - p.ay) / p.dy));
  const int start_k = static_cast<int>(std::floor((-Tz / 4.0 - p.az) / p.dz));
  const int max_i = static_cast<int>(std::floor((Tx / 4.0 - p.ax) / p.dx));
  const int max_j = static_cast<int>(std::floor((Ty / 4.0 - p.ay) / p.dy));
  const int max_k = static_cast<int>(std::floor((Tz / 4.0 - p.az) / p.dz));

  sycl::range<3> current_range(max_k - start_k, max_j - start_j, max_i - start_i);
  sycl::id<3> current_offset(start_k, start_j, start_i);

  auto& Jx_buf = field_buffer(b, Component::JX);
  auto& Jy_buf = field_buffer(b, Component::JY);
  auto& Jz_buf = field_buffer(b, Component::JZ);

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int t = 0; t < cur_time; ++t) {
    const FP time_val = static_cast<FP>(t + 1) * cur.dt;

    q.submit([&](sycl::handler& h) {
      auto Jx = Jx_buf.get_access<sycl::access::mode::write>(h);
      auto Jy = Jy_buf.get_access<sycl::access::mode::write>(h);
      auto Jz = Jz_buf.get_access<sycl::access::mode::write>(h);

      h.parallel_for(current_range, current_offset, [=](sycl::id<3> item) {
        const int k = item[0], j = item[1], i = item[2];
        const std::size_t idx = index3d(k, j, i, p.Nj, p.Ni);
        const FP value = source(static_cast<FP>(i) * p.dx,
                                static_cast<FP>(j) * p.dy,
                                static_cast<FP>(k) * p.dz,
                                time_val);
        Jx[idx] = value;
        Jy[idx] = value;
        Jz[idx] = value;
      });
    });

    update_fields(q, b, p.Nk, p.Nj, p.Ni,
                  cur_coef, coef_E_dx, coef_E_dy, coef_E_dz,
                  coef_B_dx, coef_B_dy, coef_B_dz);
  }

  zero_currents(q, b, p.Nk, p.Nj, p.Ni);

  for (int t = cur_time; t < iterations; ++t) {
    update_fields(q, b, p.Nk, p.Nj, p.Ni,
                  cur_coef, coef_E_dx, coef_E_dy, coef_E_dz,
                  coef_B_dx, coef_B_dy, coef_B_dz);
  }

  q.wait_and_throw();

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Execution time: "
            << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  sycl::host_accessor Ex(field_buffer(b, Component::EX), sycl::read_only);
  const int i = p.Ni / 2;
  for (int j = p.Nj / 2 - 5; j < p.Nj / 2 + 5; ++j) {
    for (int k = p.Nk / 2 - 5; k < p.Nk / 2 + 5; ++k) {
      const std::size_t idx = index3d(k, j, i, p.Nj, p.Ni);
      std::cout << std::setw(12) << std::fixed << std::setprecision(5) << Ex[idx];
    }
    std::cout << "\n";
  }
}

int main(int argc, char** argv) {
  try {
    int N = 256;
    int iterations = 25;
    DevicePreference pref = DevicePreference::AUTO;

    if (argc == 3 || argc == 4) {
      N = std::atoi(argv[1]);
      iterations = std::atoi(argv[2]);
    }
    if (argc == 4) {
      pref = parse_device(argv[3]);
    }

    if (!(argc == 1 || argc == 3 || argc == 4)) {
      std::cout << "Usage: " << argv[0] << " [N Iterations [auto|cpu|gpu]]\n";
      return 1;
    }

    run_fdtd(N, iterations, pref);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
