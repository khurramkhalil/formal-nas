#include <iostream>

#include "fpga_best_arch.h"
#include "parameters.h"


void fpga_best_arch(
    input_t x[3*32*32],
    result_t layer21_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer21_out 
    //#pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<stem_0_weight_t, 432>(w2, "w2.txt");
        nnet::load_weights_from_txt<stem_0_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<cell_edges_1_op_1_weight_t, 2304>(w5, "w5.txt");
        nnet::load_weights_from_txt<cell_edges_1_op_1_bias_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<cell_edges_3_op_1_weight_t, 256>(w8, "w8.txt");
        nnet::load_weights_from_txt<cell_edges_3_op_1_bias_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<cell_edges_2_op_1_weight_t, 2304>(w12, "w12.txt");
        nnet::load_weights_from_txt<cell_edges_2_op_1_bias_t, 16>(b12, "b12.txt");
        nnet::load_weights_from_txt<cell_edges_5_op_1_weight_t, 2304>(w16, "w16.txt");
        nnet::load_weights_from_txt<cell_edges_5_op_1_bias_t, 16>(b16, "b16.txt");
        nnet::load_weights_from_txt<classifier_weight_t, 160>(w21, "w21.txt");
        nnet::load_weights_from_txt<classifier_bias_t, 10>(b21, "b21.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer22_t layer22_out[32*32*3];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0

    stem_0_result_t layer2_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer4_t layer4_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    cell_edges_1_op_1_result_t layer5_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer7_t layer7_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    cell_edges_3_op_1_result_t layer8_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    add_result_t layer10_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    layer11_t layer11_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    cell_edges_2_op_1_result_t layer12_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    add_1_result_t layer14_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    layer15_t layer15_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    cell_edges_5_op_1_result_t layer16_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    add_2_result_t layer18_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0

    layer19_t layer19_out[1*1*16];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0

    auto& layer20_out = layer19_out;
    nnet::transpose<input_t, layer22_t, config22>(x, layer22_out); // transpose_input_for_x

    nnet::conv_2d_cl<layer22_t, stem_0_result_t, config2>(layer22_out, layer2_out, w2, b2); // stem_0

    nnet::relu<stem_0_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // cell_edges_1_op_0

    nnet::conv_2d_cl<layer4_t, cell_edges_1_op_1_result_t, config5>(layer4_out, layer5_out, w5, b5); // cell_edges_1_op_1

    nnet::relu<stem_0_result_t, layer7_t, relu_config7>(layer2_out, layer7_out); // cell_edges_3_op_0

    nnet::pointwise_conv_2d_cl<layer7_t, cell_edges_3_op_1_result_t, config24>(layer7_out, layer8_out, w8, b8); // cell_edges_3_op_1

    nnet::add<cell_edges_1_op_1_result_t, cell_edges_3_op_1_result_t, add_result_t, config10>(layer5_out, layer8_out, layer10_out); // add

    nnet::relu<stem_0_result_t, layer11_t, relu_config11>(layer2_out, layer11_out); // cell_edges_2_op_0

    nnet::conv_2d_cl<layer11_t, cell_edges_2_op_1_result_t, config12>(layer11_out, layer12_out, w12, b12); // cell_edges_2_op_1

    nnet::add<cell_edges_2_op_1_result_t, stem_0_result_t, add_1_result_t, config14>(layer12_out, layer2_out, layer14_out); // add_1

    nnet::relu<add_result_t, layer15_t, relu_config15>(layer10_out, layer15_out); // cell_edges_5_op_0

    nnet::conv_2d_cl<layer15_t, cell_edges_5_op_1_result_t, config16>(layer15_out, layer16_out, w16, b16); // cell_edges_5_op_1

    nnet::add<add_1_result_t, cell_edges_5_op_1_result_t, add_2_result_t, config18>(layer14_out, layer16_out, layer18_out); // add_2

    nnet::pooling2d_cl<add_2_result_t, layer19_t, config19>(layer18_out, layer19_out); // global_pool

    nnet::dense<layer19_t, result_t, config21>(layer20_out, layer21_out, w21, b21); // classifier

}

