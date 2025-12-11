#include <iostream>

#include "fpga_stream.h"
#include "parameters.h"


void fpga_stream(
    input_t x[3*32*32],
    result_t layer27_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,layer27_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<stem_0_weight_t, 432>(w2, "w2.txt");
        nnet::load_weights_from_txt<stem_0_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<cell_edges_0_op_1_weight_t, 2304>(w5, "w5.txt");
        nnet::load_weights_from_txt<cell_edges_0_op_1_bias_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<cell_edges_1_op_1_weight_t, 2304>(w8, "w8.txt");
        nnet::load_weights_from_txt<cell_edges_1_op_1_bias_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<cell_edges_3_op_1_weight_t, 2304>(w11, "w11.txt");
        nnet::load_weights_from_txt<cell_edges_3_op_1_bias_t, 16>(b11, "b11.txt");
        nnet::load_weights_from_txt<cell_edges_2_op_1_weight_t, 2304>(w15, "w15.txt");
        nnet::load_weights_from_txt<cell_edges_2_op_1_bias_t, 16>(b15, "b15.txt");
        nnet::load_weights_from_txt<cell_edges_4_op_1_weight_t, 2304>(w18, "w18.txt");
        nnet::load_weights_from_txt<cell_edges_4_op_1_bias_t, 16>(b18, "b18.txt");
        nnet::load_weights_from_txt<cell_edges_5_op_1_weight_t, 2304>(w22, "w22.txt");
        nnet::load_weights_from_txt<cell_edges_5_op_1_bias_t, 16>(b22, "b22.txt");
        nnet::load_weights_from_txt<classifier_weight_t, 160>(w27, "w27.txt");
        nnet::load_weights_from_txt<classifier_bias_t, 10>(b27, "b27.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer28_t layer28_out[32*32*3];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0

    stem_0_result_t layer2_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer4_t layer4_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    cell_edges_0_op_1_result_t layer5_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer7_t layer7_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    cell_edges_1_op_1_result_t layer8_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer10_t layer10_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    cell_edges_3_op_1_result_t layer11_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    add_result_t layer13_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    layer14_t layer14_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    cell_edges_2_op_1_result_t layer15_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    layer17_t layer17_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0

    cell_edges_4_op_1_result_t layer18_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0

    add_1_result_t layer20_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0

    layer21_t layer21_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0

    cell_edges_5_op_1_result_t layer22_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0

    add_2_result_t layer24_out[32*32*16];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0

    layer25_t layer25_out[1*1*16];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0

    auto& layer26_out = layer25_out;
    nnet::transpose<input_t, layer28_t, config28>(x, layer28_out); // transpose_input_for_x

    nnet::conv_2d_cl<layer28_t, stem_0_result_t, config2>(layer28_out, layer2_out, w2, b2); // stem_0

    nnet::relu<stem_0_result_t, layer4_t, relu_config4>(layer2_out, layer4_out); // cell_edges_0_op_0

    nnet::conv_2d_cl<layer4_t, cell_edges_0_op_1_result_t, config5>(layer4_out, layer5_out, w5, b5); // cell_edges_0_op_1

    nnet::relu<stem_0_result_t, layer7_t, relu_config7>(layer2_out, layer7_out); // cell_edges_1_op_0

    nnet::conv_2d_cl<layer7_t, cell_edges_1_op_1_result_t, config8>(layer7_out, layer8_out, w8, b8); // cell_edges_1_op_1

    nnet::relu<cell_edges_0_op_1_result_t, layer10_t, relu_config10>(layer5_out, layer10_out); // cell_edges_3_op_0

    nnet::conv_2d_cl<layer10_t, cell_edges_3_op_1_result_t, config11>(layer10_out, layer11_out, w11, b11); // cell_edges_3_op_1

    nnet::add<cell_edges_1_op_1_result_t, cell_edges_3_op_1_result_t, add_result_t, config13>(layer8_out, layer11_out, layer13_out); // add

    nnet::relu<stem_0_result_t, layer14_t, relu_config14>(layer2_out, layer14_out); // cell_edges_2_op_0

    nnet::conv_2d_cl<layer14_t, cell_edges_2_op_1_result_t, config15>(layer14_out, layer15_out, w15, b15); // cell_edges_2_op_1

    nnet::relu<cell_edges_0_op_1_result_t, layer17_t, relu_config17>(layer5_out, layer17_out); // cell_edges_4_op_0

    nnet::conv_2d_cl<layer17_t, cell_edges_4_op_1_result_t, config18>(layer17_out, layer18_out, w18, b18); // cell_edges_4_op_1

    nnet::add<cell_edges_2_op_1_result_t, cell_edges_4_op_1_result_t, add_1_result_t, config20>(layer15_out, layer18_out, layer20_out); // add_1

    nnet::relu<add_result_t, layer21_t, relu_config21>(layer13_out, layer21_out); // cell_edges_5_op_0

    nnet::conv_2d_cl<layer21_t, cell_edges_5_op_1_result_t, config22>(layer21_out, layer22_out, w22, b22); // cell_edges_5_op_1

    nnet::add<add_1_result_t, cell_edges_5_op_1_result_t, add_2_result_t, config24>(layer20_out, layer22_out, layer24_out); // add_2

    nnet::pooling2d_cl<add_2_result_t, layer25_t, config25>(layer24_out, layer25_out); // global_pool

    nnet::dense<layer25_t, result_t, config27>(layer26_out, layer27_out, w27, b27); // classifier

}

