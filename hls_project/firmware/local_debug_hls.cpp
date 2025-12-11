#include <iostream>

#include "local_debug_hls.h"
#include "parameters.h"


void local_debug_hls(
    input_t x[3*32*32],
    result_t layer20_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer20_out 
    ////////#pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<stem_0_weight_t, 432>(w2, "w2.txt");
        nnet::load_weights_from_txt<stem_0_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 16384>(s21, "s21.txt");
        nnet::load_weights_from_txt<bias21_t, 16384>(b21, "b21.txt");
        nnet::load_weights_from_txt<cell_edges_3_op_1_weight_t, 2304>(w7, "w7.txt");
        nnet::load_weights_from_txt<cell_edges_3_op_1_bias_t, 16>(b7, "b7.txt");
        nnet::load_weights_from_txt<cell_edges_2_op_1_weight_t, 2304>(w11, "w11.txt");
        nnet::load_weights_from_txt<cell_edges_2_op_1_bias_t, 16>(b11, "b11.txt");
        nnet::load_weights_from_txt<cell_edges_4_op_1_weight_t, 2304>(w14, "w14.txt");
        nnet::load_weights_from_txt<cell_edges_4_op_1_bias_t, 16>(b14, "b14.txt");
        nnet::load_weights_from_txt<classifier_weight_t, 160>(w20, "w20.txt");
        nnet::load_weights_from_txt<classifier_bias_t, 10>(b20, "b20.txt");
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

    bn_mul_result_t layer21_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0

    layer6_t layer6_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    cell_edges_3_op_1_result_t layer7_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    add_result_t layer9_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    layer10_t layer10_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    cell_edges_2_op_1_result_t layer11_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    layer13_t layer13_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    cell_edges_4_op_1_result_t layer14_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    add_1_result_t layer16_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    add_2_result_t layer17_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0

    layer18_t layer18_out[1*1*16];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0

    auto& layer19_out = layer18_out;
    nnet::transpose<input_t, layer22_t, config22>(x, layer22_out); // transpose_input_for_x

    nnet::conv_2d_cl<layer22_t, stem_0_result_t, config2>(layer22_out, layer2_out, w2, b2); // stem_0

    nnet::normalize<stem_0_result_t, bn_mul_result_t, config21>(layer2_out, layer21_out, s21, b21); // bn_mul

    nnet::relu<bn_mul_result_t, layer6_t, relu_config6>(layer21_out, layer6_out); // cell_edges_3_op_0

    nnet::conv_2d_cl<layer6_t, cell_edges_3_op_1_result_t, config7>(layer6_out, layer7_out, w7, b7); // cell_edges_3_op_1

    nnet::add<stem_0_result_t, cell_edges_3_op_1_result_t, add_result_t, config9>(layer2_out, layer7_out, layer9_out); // add

    nnet::relu<stem_0_result_t, layer10_t, relu_config10>(layer2_out, layer10_out); // cell_edges_2_op_0

    nnet::conv_2d_cl<layer10_t, cell_edges_2_op_1_result_t, config11>(layer10_out, layer11_out, w11, b11); // cell_edges_2_op_1

    nnet::relu<bn_mul_result_t, layer13_t, relu_config13>(layer21_out, layer13_out); // cell_edges_4_op_0

    nnet::conv_2d_cl<layer13_t, cell_edges_4_op_1_result_t, config14>(layer13_out, layer14_out, w14, b14); // cell_edges_4_op_1

    nnet::add<cell_edges_2_op_1_result_t, cell_edges_4_op_1_result_t, add_1_result_t, config16>(layer11_out, layer14_out, layer16_out); // add_1

    nnet::add<add_1_result_t, add_result_t, add_2_result_t, config17>(layer16_out, layer9_out, layer17_out); // add_2

    nnet::pooling2d_cl<add_2_result_t, layer18_t, config18>(layer17_out, layer18_out); // global_pool

    nnet::dense<layer18_t, result_t, config20>(layer19_out, layer20_out, w20, b20); // classifier

}

