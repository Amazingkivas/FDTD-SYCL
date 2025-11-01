#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>

#include "FDTD_sycl.h"

using namespace FDTD_sycl;

void spherical_wave(int n, int it, std::string base_path = "") {
    std::cout << "----- 1" << std::endl;
    CurrentParameters cur_param {
        8,
        4,
        0.2,
    };
    double T = cur_param.period;
    double Tx = cur_param.period_x;
    double Ty = cur_param.period_y;
    double Tz = cur_param.period_z;
    cur_param.iterations = static_cast<int>(static_cast<double>(cur_param.period) / cur_param.dt);
    std::cout << "----- 2" << std::endl;
    
    auto cur_func = [=](double x, double y, double z, double t) {
        return sin(2.0 * FDTD_const::PI * t / T) 
             * pow(cos(2.0 * FDTD_const::PI * x / Tx), 2.0) 
             * pow(cos(2.0 * FDTD_const::PI * y / Ty), 2.0) 
             * pow(cos(2.0 * FDTD_const::PI * z / Tz), 2.0);
    };
    std::cout << "----- 3" << std::endl;
    
    double d = FDTD_const::C;
    double boundary = static_cast<double>(n) / 2.0 * d;

    std::cout << "----- 4" << std::endl;

    Parameters params {
        n, n, n,
        -boundary, boundary,
        -boundary, boundary,
        -boundary, boundary,
        d, d, d
    };

    std::cout << "----- 5" << std::endl;

    FDTD_sycl::FDTD method(params, cur_param.dt);
    std::cout << "----- 6" << std::endl;

    int cur_time = std::min(cur_param.iterations, it);

    

    int start_i = static_cast<int>(floor((-Tx / 4.0 - params.ax) / params.dx));
    int start_j = static_cast<int>(floor((-Ty / 4.0 - params.ay) / params.dy));
    int start_k = static_cast<int>(floor((-Tz / 4.0 - params.az) / params.dz));
    int max_i = static_cast<int>(floor((Tx / 4.0 - params.ax) / params.dx));
    int max_j = static_cast<int>(floor((Ty / 4.0 - params.ay) / params.dy));
    int max_k = static_cast<int>(floor((Tz / 4.0 - params.az) / params.dz));

    auto& q = method.get_queue();
    auto Jx_buf = method.get_field_buffer(Component::JX);
    auto Jy_buf = method.get_field_buffer(Component::JY);
    auto Jz_buf = method.get_field_buffer(Component::JZ);

    std::cout << "----- 1" << std::endl;
    sycl::range<3> current_range(max_k - start_k, max_j - start_j, max_i - start_i);
    std::cout << "----- 2" << std::endl;
    sycl::id<3> current_offset(start_k, start_j, start_i);

    
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < cur_time; t++) {
        double time_val = static_cast<double>(t + 1) * cur_param.dt;
        
        q.submit([&](sycl::handler& h) {
            auto Jx_acc = Jx_buf.get_access<sycl::access::mode::write>(h);
            auto Jy_acc = Jy_buf.get_access<sycl::access::mode::write>(h);
            auto Jz_acc = Jz_buf.get_access<sycl::access::mode::write>(h);
            
            h.parallel_for(current_range, current_offset, [=](sycl::id<3> item) {
                int k = item[0];
                int j = item[1];
                int i = item[2];

                double value = cur_func(static_cast<double>(i) * params.dx,
                                        static_cast<double>(j) * params.dy,
                                        static_cast<double>(k) * params.dz,
                                        time_val);


                
                Jx_acc[k][j][i] = value;
                Jy_acc[k][j][i] = value;
                Jz_acc[k][j][i] = value;
            });
        });

        method.update_fields();
    }
    
    method.zeroed_currents();
    for (int t = cur_time; t < it; t++) {
        method.update_fields();
    }

    q.wait_and_throw();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " s" << std::endl;

    std::cout << "SYCL results: \n" << std::endl;
    int i = params.Ni / 2;
    sycl::host_accessor Ex_host_acc(method.get_field_buffer(Component::EX), sycl::read_only);

    for (int j = params.Nj / 2 - 5; j < params.Nj/2 + 5; j++) {
        for (int k = params.Nk/2 - 5; k < params.Nk/2 + 5; k++) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(5) 
                      << Ex_host_acc[i][j][k];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        std::vector<char*> arguments(argv, argv + argc);
        if (argc == 1) {
            int N = 512;
            int Iterations = 25;
            spherical_wave(N, Iterations, "../../");
        } else if (argc == 3) {
            
            int N = std::atoi(arguments[1]);
            int Iterations = std::atoi(arguments[2]);
            std::cout << N << " | " << Iterations << std::endl;
            spherical_wave(N, Iterations, "../../");
        } else {
            std::cout << "ERROR: Incorrect number of parameters" << std::endl;
            exit(1);
        }
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
