#include "utils.h"
#include "objdetect_prv.h"

static objdetect_struct* objdet_info = NULL;

void objdetect_free
    (
    void
    )
{
    free_stack();
}

void objdetect_init
    (
    char* weight_file_path
    )
{
    //0. initialize stack
    init_stack();
    //1. initialize detector, fill in options
    objdet_info = (objdetect_struct*)alloc_from_stack
            (sizeof(objdetect_struct));
    objdet_info->thresh = 0.24f;
    objdet_info->nms_thresh = 0.3f;
    objdet_info->class_num = 20; // voc-pascal dataset
    char* class_names[] = {
                         "aeroplane",
                         "bicycle",
                         "bird",
                         "boat",
                         "bottle",
                         "bus",
                         "car",
                         "cat",
                         "chair",
                         "cow",
                         "diningtable",
                         "dog",
                         "horse",
                         "motorbike",
                         "person",
                         "pottedplant",
                         "sheep",
                         "sofa",
                         "train",
                         "tvmonitor"
                         };
    objdet_info->names = class_names;

    //2. initialize net
    objdet_info->net.n = 16; // darknet reference network
    objdet_info->net.layers = (layer_struct*)alloc_from_stack
            (objdet_info->net.n * sizeof(layer_struct));
    objdet_info->net.w = 416;
    objdet_info->net.h = 416;
    objdet_info->net.c = 3;
    //objdet_info->net.inputs = objdet_info->net.w * objdet_info->net.h * objdet_info->net.c;

    //3. load weights file
    FILE *fp = fopen(weight_file_path, "rb");
    if(!fp)
        {
        printf("failed to open weight file\n");
        exit(EXIT_FAILURE);
        }
    //TODO: remove these header
    int header[4];
    fread(header, sizeof(int), 4, fp);

    //4. initialize layers in net
    layer_struct *layer_ptr = NULL;
    layer_struct *prev_layer_ptr = NULL;

    // layer 0 (convolution, stride = 1)
    layer_ptr = objdet_info->net.layers;
    layer_ptr->n = 16;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = objdet_info->net.w;
    layer_ptr->h = objdet_info->net.h;
    layer_ptr->c = objdet_info->net.c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    int num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 1 (maxpool, pad = 0)
    layer_ptr->size = 2;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->h * layer_ptr->w * layer_ptr->c;
    layer_ptr->out_w = layer_ptr->w / layer_ptr->stride;
    layer_ptr->out_h = layer_ptr->h / layer_ptr->stride;
    layer_ptr->out_c = layer_ptr->c;
    layer_ptr->outputs = layer_ptr->out_h * layer_ptr->out_w * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_maxpool_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 2 (convolution, stride = 1)
    layer_ptr->n = 32;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 3 (maxpool, pad = 0)
    layer_ptr->size = 2;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->h * layer_ptr->w * layer_ptr->c;
    layer_ptr->out_w = layer_ptr->w / layer_ptr->stride;
    layer_ptr->out_h = layer_ptr->h / layer_ptr->stride;
    layer_ptr->out_c = layer_ptr->c;
    layer_ptr->outputs = layer_ptr->out_h * layer_ptr->out_w * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_maxpool_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 4 (convolution, stride = 1)
    layer_ptr->n = 64;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 5 (maxpool, pad = 0)
    layer_ptr->size = 2;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->h * layer_ptr->w * layer_ptr->c;
    layer_ptr->out_w = layer_ptr->w / layer_ptr->stride;
    layer_ptr->out_h = layer_ptr->h / layer_ptr->stride;
    layer_ptr->out_c = layer_ptr->c;
    layer_ptr->outputs = layer_ptr->out_h * layer_ptr->out_w * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_maxpool_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 6 (convolution, stride = 1)
    layer_ptr->n = 128;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 7 (maxpool, pad = 0)
    layer_ptr->size = 2;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->h * layer_ptr->w * layer_ptr->c;
    layer_ptr->out_w = layer_ptr->w / layer_ptr->stride;
    layer_ptr->out_h = layer_ptr->h / layer_ptr->stride;
    layer_ptr->out_c = layer_ptr->c;
    layer_ptr->outputs = layer_ptr->out_h * layer_ptr->out_w * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_maxpool_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 8 (convolution, stride = 1)
    layer_ptr->n = 256;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 9 (maxpool, pad = 0)
    layer_ptr->size = 2;
    layer_ptr->stride = 2;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->h * layer_ptr->w * layer_ptr->c;
    layer_ptr->out_w = layer_ptr->w / layer_ptr->stride;
    layer_ptr->out_h = layer_ptr->h / layer_ptr->stride;
    layer_ptr->out_c = layer_ptr->c;
    layer_ptr->outputs = layer_ptr->out_h * layer_ptr->out_w * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_maxpool_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 10 (convolution, stride = 1)
    layer_ptr->n = 512;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 11 (maxpool, pad = 0)
    layer_ptr->size = 2;
    layer_ptr->stride = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->h * layer_ptr->w * layer_ptr->c;
    layer_ptr->out_w = layer_ptr->w / layer_ptr->stride;
    layer_ptr->out_h = layer_ptr->h / layer_ptr->stride;
    layer_ptr->out_c = layer_ptr->c;
    layer_ptr->outputs = layer_ptr->out_h * layer_ptr->out_w * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->forward = forward_maxpool_layer;
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 12 (convolution, stride = 1)
    layer_ptr->n = 1024;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 13 (convolution, stride = 1)
    layer_ptr->n = 1024;
    layer_ptr->size = 3;
    layer_ptr->pad = 1;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->scales = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_mean = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->rolling_variance = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = leaky_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->scales, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_mean, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->rolling_variance, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 14 (convolution, stride = 1)
    layer_ptr->n = 125;
    layer_ptr->size = 1;
    layer_ptr->pad = 0;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->inputs = layer_ptr->w * layer_ptr->h * layer_ptr->c;
    num_weights = layer_ptr->n * layer_ptr->c * layer_ptr->size * layer_ptr->size;
    layer_ptr->weights = (float *)alloc_from_stack(num_weights * sizeof(float));
    layer_ptr->biases = (float *)alloc_from_stack(layer_ptr->n * sizeof(float));
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->n;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->activation = linear_activate;
    layer_ptr->forward = forward_convolutional_layer;
    fread(layer_ptr->biases, sizeof(float), layer_ptr->n, fp);
    fread(layer_ptr->weights, sizeof(float), num_weights, fp);
    prev_layer_ptr = layer_ptr;
    ++layer_ptr;

    // layer 15 (region)
    layer_ptr->n = 5;
    layer_ptr->w = prev_layer_ptr->out_w;
    layer_ptr->h = prev_layer_ptr->out_h;
    layer_ptr->c = prev_layer_ptr->out_c;
    layer_ptr->out_w = layer_ptr->w;
    layer_ptr->out_h = layer_ptr->h;
    layer_ptr->out_c = layer_ptr->c;
    layer_ptr->outputs = layer_ptr->out_w * layer_ptr->out_h * layer_ptr->out_c;
    layer_ptr->output = (float *)alloc_from_stack(layer_ptr->outputs * sizeof(float));
    layer_ptr->classes = objdet_info->class_num;
    layer_ptr->activation = logistic_activate;
    layer_ptr->forward = forward_region_layer;
    layer_ptr->biases = (float *)alloc_from_stack(2 * layer_ptr->n * sizeof(float));
    //well picked region proposals
    layer_ptr->biases[0] = 1.08f;
    layer_ptr->biases[1] = 1.19f;
    layer_ptr->biases[2] = 3.42f;
    layer_ptr->biases[3] = 4.41f;
    layer_ptr->biases[4] = 6.63f;
    layer_ptr->biases[5] = 11.38f;
    layer_ptr->biases[6] = 9.42f;
    layer_ptr->biases[7] = 5.11f;
    layer_ptr->biases[8] = 16.62f;
    layer_ptr->biases[9] = 10.52f;

    // result buffer in network structure
    int out_grid_num = layer_ptr->w * layer_ptr->h * layer_ptr->n;
    objdet_info->net.result_boxes = (box_struct *)alloc_from_stack(out_grid_num * sizeof(box_struct));
    objdet_info->net.result_probs = (float **)alloc_from_stack(out_grid_num * sizeof(float*));
    int i = 0;
    for(i = 0; i < out_grid_num; ++i)
        {
        objdet_info->net.result_probs[i] = (float *)alloc_from_stack((layer_ptr->classes + 1) * sizeof(float));
        }
    fclose(fp);
}

void objdetect_main
    (
    unsigned char *im,
    int imw,
    int imh
    )
{
    objdet_info->src = im;
    objdet_info->srcw = imw;
    objdet_info->srch = imh;
    objdet_info->net.input = preprocessed(objdet_info);

    double time = what_time_is_it_now();
    network_predict(objdet_info->net);
    printf("Predicted in %f seconds.\n", what_time_is_it_now()-time);

    get_region_boxes(objdet_info);
    nonmax_suppression(objdet_info);
    draw_result(objdet_info);
}
