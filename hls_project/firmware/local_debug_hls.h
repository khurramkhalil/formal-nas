#ifndef LOCAL_DEBUG_HLS_H_
#define LOCAL_DEBUG_HLS_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void local_debug_hls(
    input_t x[3*32*32],
    result_t layer20_out[10]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
