#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer22_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<38,18> stem_0_result_t;
typedef ap_fixed<16,6> stem_0_weight_t;
typedef ap_fixed<16,6> stem_0_bias_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<18,8> cell_edges_1_op_0_table_t;
typedef ap_fixed<41,21> cell_edges_1_op_1_result_t;
typedef ap_fixed<16,6> cell_edges_1_op_1_weight_t;
typedef ap_fixed<16,6> cell_edges_1_op_1_bias_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<18,8> cell_edges_3_op_0_table_t;
typedef ap_fixed<37,17> cell_edges_3_op_1_result_t;
typedef ap_fixed<16,6> cell_edges_3_op_1_weight_t;
typedef ap_fixed<16,6> cell_edges_3_op_1_bias_t;
typedef ap_fixed<42,22> add_result_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<18,8> cell_edges_2_op_0_table_t;
typedef ap_fixed<41,21> cell_edges_2_op_1_result_t;
typedef ap_fixed<16,6> cell_edges_2_op_1_weight_t;
typedef ap_fixed<16,6> cell_edges_2_op_1_bias_t;
typedef ap_fixed<42,22> add_1_result_t;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<18,8> cell_edges_5_op_0_table_t;
typedef ap_fixed<41,21> cell_edges_5_op_1_result_t;
typedef ap_fixed<16,6> cell_edges_5_op_1_weight_t;
typedef ap_fixed<16,6> cell_edges_5_op_1_bias_t;
typedef ap_fixed<43,23> add_2_result_t;
typedef ap_fixed<43,23> layer19_t;
typedef ap_fixed<64,34> result_t;
typedef ap_fixed<16,6> classifier_weight_t;
typedef ap_fixed<16,6> classifier_bias_t;
typedef ap_uint<1> layer21_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
