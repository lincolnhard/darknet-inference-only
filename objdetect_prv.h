#ifndef OBJDETECT_PRV_H
#define OBJDETECT_PRV_H

#include "utils.h"

#define PERSON_IDX_IN_VOC 14

struct layerst;
typedef struct layerst layer_struct;
struct networkst;
typedef struct networkst network_struct;

typedef struct
    {
    int index;
    int class_num;
    float **probs;
    }sortable_bbox;

struct layerst
    {
    void (*activation)
        (
        float*,
        int
        );
    void (*forward)
        (
        struct layerst,
        struct networkst
        );
    int n; // filter numbers (output channels)
    int size; // filter size
    int stride; // for maxpool layer only
    int pad; // for conv layer only
    int h; // input height
    int w; // input width
    int c; // input channels
    int inputs; // total input pixels
    float *weights;
    float *biases;
    int out_h; // output height
    int out_w; // output width
    int out_c; // output channels
    int outputs; // total output pixels
    float *scales; // batch normalization coefficients
    float *rolling_mean; // batch normalization coefficients
    float *rolling_variance; // batch normalization coefficients
    float *output; // layer result
    int classes; // class number, for last region layer only
    };

struct networkst
    {
    int n; // total layer number
    int h; // input image height
    int w; // input image width
    int c; // input image channels
    float *input; // input src (note it will be the buffer for every layer's input)
    layer_struct *layers; //pointer to each layer
    box_struct *result_boxes;
    float **result_probs;
    };

typedef struct
    {
    int class_num;
    float thresh;
    float nms_thresh;
    network_struct net;
    unsigned char* src;
    int srcw;
    int srch;
    int output[401]; // TODO: remove hard coded number here ([0]: count, [rest]: obj box)
    }objdetect_struct;

void forward_convolutional_layer
    (
    layer_struct l,
    network_struct net
    );

void logistic_activate
    (
    float *x,
    int num
    );

void relu_activate
    (
    float *x,
    int num
    );

void linear_activate
    (
    float *x,
    int num
    );

void leaky_activate
    (
    float *x,
    int num
    );

void forward_maxpool_layer
    (
    const layer_struct l,
    network_struct net
    );

void forward_region_layer
    (
    const layer_struct l,
    network_struct net
    );

float* preprocessed
    (
    objdetect_struct* objdet_wksp
    );

void network_predict
    (
    network_struct net
    );

void get_region_boxes
    (
    objdetect_struct* objdet_wksp
    );

void nonmax_suppression
    (
    objdetect_struct* objdet_wksp
    );

void draw_result
    (
    objdetect_struct* objdet_wksp
    );

void person_detection
    (
    objdetect_struct* objdet_wksp
    );

void clear_network
    (
    network_struct net
    );

#endif // OBJDETECT_PRV_H
