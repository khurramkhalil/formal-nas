#ifndef FPGA_STREAM_H_
#define FPGA_STREAM_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void fpga_stream(
    input_t x[3*32*32],
    result_t layer27_out[10]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
