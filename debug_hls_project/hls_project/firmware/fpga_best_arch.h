#ifndef FPGA_BEST_ARCH_H_
#define FPGA_BEST_ARCH_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void fpga_best_arch(
    input_t x[3*32*32],
    result_t layer21_out[10]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
