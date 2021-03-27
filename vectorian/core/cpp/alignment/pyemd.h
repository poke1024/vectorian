#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-compare"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#include "EMD_DEFS.hpp"
#include "flow_utils.hpp"
//#include "emd_hat.hpp"

template<typename NUM_T, FLOW_TYPE_T FLOW_TYPE= NO_FLOW>
struct emd_hat_gd_metric {
    NUM_T operator()(const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
                     const std::vector< std::vector<NUM_T> >& C,
                     NUM_T extra_mass_penalty= -1,
                     std::vector< std::vector<NUM_T> >* F= NULL);
};

template<typename NUM_T, FLOW_TYPE_T FLOW_TYPE= NO_FLOW>
struct emd_hat {
    NUM_T operator()(const std::vector<NUM_T>& P, const std::vector<NUM_T>& Q,
                     const std::vector< std::vector<NUM_T> >& C,
                     NUM_T extra_mass_penalty= -1,
                     std::vector< std::vector<NUM_T> >* F= NULL);

};

#include "emd_hat_impl.hpp"

#pragma GCC diagnostic pop

#pragma clang diagnostic pop
