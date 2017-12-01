#include "objdetect_prv.h"
#include <omp.h>

void softmax
    (
    float *input,
    float *output,
    int stride, // l.w * l.h
    int num_class
    )
{
    int i = 0;
    int j = 0;
    float* inptr = input;
    float* outptr = output;
    for(i = 0; i < stride; ++i)
        {
        inptr = input + i;
        outptr = output + i;
        float sum = 0;
        float largest = -FLT_MAX;
        for(j = 0; j < num_class; ++j)
            {
            if(inptr[j * stride] > largest)
                {
                largest = inptr[j * stride];
                }
            }
        for(j = 0; j < num_class; ++j)
            {
            float e = exp(inptr[j * stride] - largest);
            sum += e;
            outptr[j * stride] = e;
            }
        for(j = 0; j < num_class; ++j)
            {
            outptr[j * stride] /= sum;
            }
        }
}

void logistic_activate
    (
    float *x,
    int num
    )
{
    int i = 0;
    for(i = 0; i < num; ++i)
        {
        x[i] = 1.0f / (1.0f + exp(-x[i]));
        }
}

void relu_activate
    (
    float *x,
    int num
    )
{
    int i = 0;
    for(i = 0; i < num; ++i)
        {
        if(x[i] < 0.0f)
            {
            x[i] = 0.0f;
            }
        }
}

void linear_activate
    (
    float *x,
    int num
    )
{
    return;
}

void leaky_activate
    (
    float *x,
    int num
    )
{
    int i = 0;
    for(i = 0; i < num; ++i)
        {
        if(x[i] < 0.0f)
            {
            x[i] = 0.1f * x[i];
            }
        }
}

void add_bias
    (
    float *output,
    float *biases,
    int n,
    int size
    )
{
    int i = 0;
    int j = 0;
    for(i = 0; i < n; ++i)
        {
        for(j = 0; j < size; ++j)
            {
            output[i * size + j] += biases[i];
            }
        }
}

void scale_bias
    (
    float *output,
    float *scales,
    int n,
    int size
    )
{
    int i = 0;
    int j = 0;
    for(i = 0; i < n; ++i)
        {
        for(j = 0; j < size; ++j)
            {
            output[i * size + j] *= scales[i];
            }
        }
}

void batch_normalize
    (
    float *src,
    float *mean,
    float *variance,
    int filters,
    int spatial
    )
{
    int f = 0;
    int i = 0;
    for(f = 0; f < filters; ++f)
        {
        for(i = 0; i < spatial; ++i)
            {
            int index = f * spatial + i;
            //TODO: deal with sqrt first
            src[index] = (src[index] - mean[f]) / (sqrt(variance[f]) + 0.000001f);
            }
        }
}

void gemm
    (
    int M,
    int N,
    int K,
    float *A,
    float *B,
    float *C
    )
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i)
        {
        for(k = 0; k < K; ++k)
            {
            register float A_PART = A[i * K + k];
            for(j = 0; j < N; ++j)
                {
                C[i * N + j] += A_PART * B[k * N + j];
                }
            }
        }
}

float im2col_get_pixel
    (
    float *im,
    int height,
    int width,
    int row,
    int col,
    int channel,
    int pad
    )
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        {
        return 0;
        }
    return im[col + width*(row + height*channel)];
}

void im2col
    (
    float* data_im,
    int channels,
    int height,
    int width,
    int ksize,
    int pad,
    float* data_col
    )
{
    int c,h,w;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
        {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height; ++h)
            {
            for (w = 0; w < width; ++w)
                {
                int im_row = h_offset + h;
                int im_col = w_offset + w;
                int col_index = (c * height + h) * width + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, im_row, im_col, c_im, pad);
                }
            }
        }
}

void forward_convolutional_layer
    (
    layer_struct l,
    network_struct net
    )
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;
    float *a = l.weights;
    float *b = (float*)alloc_from_stack(l.out_h * l.out_w * l.size * l.size * l.c * sizeof(float));
    float *c = l.output;
    im2col(net.input, l.c, l.h, l.w, l.size, l.pad, b);
    gemm(m, n, k, a, b, c);
    if(l.size != 1) // size 1 convolution layer mean fully-connected layer
        {
        // normalized by the training sample mean, variance and scales
        batch_normalize(c, l.rolling_mean, l.rolling_variance, l.out_c, n);
        scale_bias(c, l.scales, l.out_c, n);
        }
    add_bias(c, l.biases, l.out_c, n);
    l.activation(c, l.outputs);
    partial_free_from_stack(l.out_h * l.out_w * l.size * l.size * l.c * sizeof(float));
}

void forward_maxpool_layer
    (
    const layer_struct l,
    network_struct net
    )
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    int ste = l.stride;
    int i = 0;
    int j = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    for(k = 0; k < c; ++k)
        {
        for(i = 0; i < h; ++i)
            {
            for(j = 0; j < w; ++j)
                {
                int out_index = j + w*(i + h * k);
                float max = -FLT_MAX;
                for(n = 0; n < l.size; ++n)
                    {
                    for(m = 0; m < l.size; ++m)
                        {
                        int cur_h = i * ste + n;
                        int cur_w = j * ste + m;
                        int index = cur_w + l.w*(cur_h + l.h * k);
                        int valid = (cur_h >= 0 && cur_h < l.h && cur_w >= 0 && cur_w < l.w);
                        float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                        max = (val > max) ? val : max;
                        }
                    }
                l.output[out_index] = max;
                }
            }
        }
}

void forward_region_layer
    (
    const layer_struct l,
    network_struct net
    )
{
    int n = 0;
    int idx1 = 0;
    int idx2 = 0;
    int idx3 = 0;

    memcpy(l.output, net.input, l.outputs * sizeof(float));
    for(n = 0; n < l.n; ++n)
        {
        // every time jump w*h*(num class + 4 + 1)
        l.activation(l.output + idx1, 2 * l.w * l.h); // x, y
        idx2 = idx1 + 4 * l.w * l.h; // P(Obj)
        l.activation(l.output + idx2, l.w * l.h);
        idx3 = idx1 + 5 * l.w * l.h; // P(class)
        softmax(net.input + idx3, l.output + idx3, l.w * l.h, l.classes);
        idx1 += l.w * l.h * (l.classes + 5);
        }
}

float* preprocessed
    (
    objdetect_struct* objdet_wksp
    )
{
    int netw = objdet_wksp->net.w;
    int neth = objdet_wksp->net.h;
    int neww = 0;
    int newh = 0;
    int srcw = objdet_wksp->srcw;
    int srch = objdet_wksp->srch;
    unsigned char* src = objdet_wksp->src;
    if(srcw > srch)
        {
        neww = netw;
        newh = neww * srch / srcw;
        }
    else
        {
        newh = neth;
        neww = newh * srcw / srch;
        }
    float *boxim = (float *)alloc_from_stack(netw * neth * 3 * sizeof(float));
    float *resizeimbuf = (float *)alloc_from_stack(neww * newh * 3 * sizeof(float));
    float *partimbuf = (float *)alloc_from_stack(neww * srch * 3 * sizeof(float));
    float *imbuf = (float *)alloc_from_stack(srcw * srch * 3 * sizeof(float));
    //1. split into whole b, whole g, whole r buffer, and normalize
    int idx = 0;
    int flatsize = srcw * srch;
    float* imbufptr = imbuf;
    unsigned char* srcptr = src + 2;
    //r
    for(idx = 0; idx < flatsize; ++idx)
        {
        imbufptr[idx] = (*srcptr) / 255.;
        srcptr += 3;
        }
    //g
    imbufptr = imbuf + flatsize;
    srcptr = src + 1;
    for(idx = 0; idx < flatsize; ++idx)
        {
        imbufptr[idx] = (*srcptr) / 255.;
        srcptr += 3;
        }
    //b
    imbufptr = imbuf + (flatsize << 1);
    srcptr = src;
    for(idx = 0; idx < flatsize; ++idx)
        {
        imbufptr[idx] = (*srcptr) / 255.;
        srcptr += 3;
        }
    //2. resize to net input size, but keep aspect ratio unchanged
    float w_scale = (float)(srcw - 1) / (neww - 1);
    float h_scale = (float)(srch - 1) / (newh - 1);
    int r = 0;
    int c = 0;
    int k = 0;
    for(k = 0; k < 3; ++k)
        {
        for(r = 0; r < srch; ++r)
            {
            for(c = 0; c < neww; ++c)
                {
                float val = 0;
                if(c == neww - 1)
                    {
                    val = imbuf[k*srcw*srch + r*srcw + srcw-1];
                    }
                else
                    {
                    float sx = c * w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * imbuf[k*srcw*srch + r*srcw + ix] + dx * imbuf[k*srcw*srch + r*srcw + ix+1];
                    }
                partimbuf[k*neww*srch + r*neww + c] = val;
                }
            }
        }
    partial_free_from_stack(srcw * srch * 3 * sizeof(float));

    for(k = 0; k < 3; ++k)
        {
        for(r = 0; r < newh; ++r)
            {
            float sy = r * h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < neww; ++c)
                {
                float val = (1 - dy) * partimbuf[k*neww*srch + iy*neww + c];
                resizeimbuf[k*neww*newh + r*neww + c] = val;
                }
            if(r == newh-1)
                {
                continue;
                }
            for(c = 0; c < neww; ++c)
                {
                float val = dy * partimbuf[k*neww*srch + (iy+1)*neww + c];
                resizeimbuf[k*neww*newh + r*neww + c] += val;
                }
            }
        }
    partial_free_from_stack(neww * srch * 3 * sizeof(float));
    //3. fill in box
    for(idx = 0; idx < neth * netw * 3; ++idx)
        {
        boxim[idx] = 0.5f;
        }
    int dx = (netw - neww) >> 1;
    int dy = (neth - newh) >> 1;
    for(k = 0; k < 3; ++k)
        {
        for(r = 0; r < newh; ++r)
            {
            for(c = 0; c < neww; ++c)
                {
                float val = resizeimbuf[k*neww*newh + r*neww + c];
                boxim[k*netw*neth + (dy+r)*netw + dx + c] = val;
                }
            }
        }
    partial_free_from_stack(neww * newh * 3 * sizeof(float));
    return boxim;
}

void network_predict
    (
    network_struct net
    )
{
    // forward propagation
    int i = 0;
    for(i = 0; i < net.n; ++i)
        {
        layer_struct l = net.layers[i];
        l.forward(l, net);
        net.input = l.output;
        }
}

void get_region_boxes
    (
    objdetect_struct* objdet_wksp
    )
{
    layer_struct l = objdet_wksp->net.layers[objdet_wksp->net.n - 1];
    box_struct *boxes = objdet_wksp->net.result_boxes;
    float **probs = objdet_wksp->net.result_probs;
    float *predictions = l.output;
    float *biases = l.biases;
    float thresh = objdet_wksp->thresh;
    int outw = l.w;
    int outh = l.h;
    int outsize = l.w * l.h;
    int group_num = l.n;
    int class_num = objdet_wksp->class_num;
    int group_size = outsize * (class_num + 5);
    int prediction_num = group_num * outsize;
    int i = 0;
    int j = 0;
    int n = 0;
    int netw = objdet_wksp->net.w;
    int neth = objdet_wksp->net.h;
    int neww = 0;
    int newh = 0;
    int srcw = objdet_wksp->srcw;
    int srch = objdet_wksp->srch;
    if(srcw > srch)
        {
        neww = netw;
        newh = neww * srch / srcw;
        }
    else
        {
        newh = neth;
        neww = newh * srcw / srch;
        }
    float box_pad_ratio_width = (netw - neww) / 2.0 / netw;
    float box_pad_ratio_height = (neth - newh) / 2.0 / neth;
    float netnew_wratio = (float)netw / neww;
    float netnew_hratio = (float)neth / newh;

    // output plane channel order is: x, y, w, h, p(obj), p(class1), p(class2), ...
    for(i = 0; i < outsize; ++i)
        {
        int row = i / outw;
        int col = i % outw;
        int obj_idx = outsize * 4 + i;
        int box_idx = i;
        int cls_idx = outsize * 5 + i;
        for(n = 0; n < group_num; ++n)
            {
            int index = n*outsize + i;
            float scale = predictions[obj_idx];
            float maxp = 0;
            boxes[index].x = (col + predictions[box_idx]) / outw;
            boxes[index].y = (row + predictions[box_idx + outsize]) / outh;
            boxes[index].w = exp(predictions[box_idx + (outsize << 1)]) * biases[2 * n] / outw;
            boxes[index].h = exp(predictions[box_idx + (outsize * 3)]) * biases[2 * n + 1] / outh;
            for(j = 0; j < class_num; ++j)
                {
                probs[index][j] = 0.0f;
                }
            for(j = 0; j < class_num; ++j)
                {
                float prob = scale * predictions[cls_idx];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if(prob > maxp)
                    {
                    maxp = prob;
                    }
                cls_idx += outsize;
                }
            probs[index][class_num] = maxp;
            obj_idx += group_size;
            box_idx += group_size;
            cls_idx += prediction_num;
            }
        }
    for(i = 0; i < prediction_num; ++i)
        {
        box_struct b = boxes[i];
        b.x = (b.x - box_pad_ratio_width) * netnew_wratio;
        b.y = (b.y - box_pad_ratio_height) * netnew_hratio;
        b.w *= netnew_wratio;
        b.h *= netnew_hratio;
        boxes[i] = b;
        }
}

float overlap
    (
    float x1,
    float w1,
    float x2,
    float w2
    )
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection
    (
    box_struct a,
    box_struct b
    )
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0)
        {
        return 0;
        }
    float area = w * h;
    return area;
}

float box_iou
    (
    box_struct a,
    box_struct b
    )
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return i / u;
}

int nms_comparator
    (
    const void *pa,
    const void *pb
    )
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class_num] - b.probs[b.index][b.class_num];
    if(diff < 0)
        {
        return 1;
        }
    else if(diff > 0)
        {
        return -1;
        }
    return 0;
}

void nonmax_suppression
    (
    objdetect_struct* objdet_wksp
    )
{
    int i = 0;
    int j = 0;
    int k = 0;
    layer_struct l = objdet_wksp->net.layers[objdet_wksp->net.n - 1];
    box_struct *boxes = objdet_wksp->net.result_boxes;
    float **probs = objdet_wksp->net.result_probs;
    int classes = objdet_wksp->class_num;
    int total = l.w * l.h * l.n;
    float nms = objdet_wksp->nms_thresh;
    sortable_bbox *s = (sortable_bbox *)alloc_from_stack(total * sizeof(sortable_bbox));
    for(i = 0; i < total; ++i)
        {
        s[i].index = i;
        s[i].class_num = classes;
        s[i].probs = probs;
        }
    // after sorting: probs[s[0].index][classes] > probs[s[1].index][classes] > ...
    qsort(s, total, sizeof(sortable_bbox), nms_comparator);
    //non maximum suppression
    for(i = 0; i < total; ++i)
        {
        if(probs[s[i].index][classes] == 0)
            {
            continue;
            }
        box_struct a = boxes[s[i].index];
        for(j = i + 1; j < total; ++j)
            {
            box_struct b = boxes[s[j].index];
            if(box_iou(a, b) > nms)
                {
                for(k = 0; k < classes+1; ++k)
                    {
                    probs[s[j].index][k] = 0;
                    }
                }
            }
        }
    partial_free_from_stack(total * sizeof(sortable_bbox));
}

void draw_result
    (
    objdetect_struct* objdet_wksp
    )
{
    unsigned char* src = objdet_wksp->src;
    layer_struct l = objdet_wksp->net.layers[objdet_wksp->net.n - 1];
    box_struct *boxes = objdet_wksp->net.result_boxes;
    float **probs = objdet_wksp->net.result_probs;
    int total = l.w * l.h * l.n;
    int classes = objdet_wksp->class_num;
    float thresh = objdet_wksp->thresh;
    int srcw = objdet_wksp->srcw;
    int srch = objdet_wksp->srch;
    int i = 0;
    int j = 0;
    for(i = 0; i < total; ++i)
        {
        float *p = probs[i];
        float maxp = p[0];
        int max_cls_idx = 0;
        for(j = 1; j < classes; ++j)
            {
            if(p[j] > maxp)
                {
                maxp = p[j];
                max_cls_idx = j;
                }
            }
        float prob = p[max_cls_idx];
        if(prob > thresh)
            {
            printf("%f\n", prob);
            box_struct b = boxes[i];
            int left  = (b.x - b.w/2.) * srcw;
            int right = (b.x + b.w/2.) * srcw;
            int top   = (b.y - b.h/2.) * srch;
            int bot   = (b.y + b.h/2.) * srch;
            if(left < 0)
                {
                left = 0;
                }
            if(right > srcw - 1)
                {
                right = srcw-1;
                }
            if(top < 0)
                {
                top = 0;
                }
            if(bot > srch - 1)
                {
                bot = srch-1;
                }
            memset(src + 3 * (top * srcw + left), 125, 3 * (right - left));
            memset(src + 3 * (bot * srcw + left), 125, 3 * (right - left));
            }
        }
}

void person_detection
    (
    objdetect_struct* objdet_wksp
    )
{
    layer_struct l = objdet_wksp->net.layers[objdet_wksp->net.n - 1];
    box_struct *boxes = objdet_wksp->net.result_boxes;
    float **probs = objdet_wksp->net.result_probs;
    int total = l.w * l.h * l.n;
    int classes = objdet_wksp->class_num;
    float thresh = objdet_wksp->thresh;
    int srcw = objdet_wksp->srcw;
    int srch = objdet_wksp->srch;
    int i = 0;
    int j = 0;
    int count = 0;
    for(i = 0; i < total; ++i)
        {
        float *p = probs[i];
        float maxp = p[0];
        int max_cls_idx = 0;
        for(j = 1; j < classes; ++j)
            {
            if(p[j] > maxp)
                {
                maxp = p[j];
                max_cls_idx = j;
                }
            }
        float prob = p[max_cls_idx];
        if(prob > thresh && max_cls_idx == PERSON_IDX_IN_VOC)
            {
            box_struct b = boxes[i];
            int left  = (b.x - b.w/2.) * srcw;
            int right = (b.x + b.w/2.) * srcw;
            int top   = (b.y - b.h/2.) * srch;
            int bot   = (b.y + b.h/2.) * srch;
            if(left < 0)
                {
                left = 0;
                }
            if(right > srcw - 1)
                {
                right = srcw-1;
                }
            if(top < 0)
                {
                top = 0;
                }
            if(bot > srch - 1)
                {
                bot = srch-1;
                }
            objdet_wksp->output[count*4+1] = left;
            objdet_wksp->output[count*4+2] = top;
            objdet_wksp->output[count*4+3] = right - left;
            objdet_wksp->output[count*4+4] = bot - top;
            ++count;
            }
        }
    objdet_wksp->output[0] = count;
}

void clear_network
    (
    network_struct net
    )
{
    int i = 0;
    for(i = 0; i < net.n; ++i)
        {
        layer_struct l = net.layers[i];
        memset(l.output, 0, l.outputs * sizeof(float));
        }
}
