#pragma once

#include <CL/sycl.hpp>

#include "Structures.h"
#include "Constants.h"
#include "Enums.h"
#include "FP.h"
#include "allocate.h"
#include "sycl_devices.h"

namespace FDTD_sycl {

    using namespace FDTD_enums;
    using namespace FDTD_struct;

    using Field = std::vector<FP, no_init_allocator<FP>>;

    inline int applyPeriodic(const int& i, const int& N) {
        int i_isMinusOne = (i < 0);
    
        int i_isNi = (i == N);
    
        int new_i = (N - 1) * i_isMinusOne + i *
            !(i_isMinusOne || i_isNi);
        
        return new_i;
    }

} // namespace FDTD_sycl
