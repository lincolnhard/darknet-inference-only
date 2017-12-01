#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include "objdetect_pub.h"

#define PIC_DEBUGx 1

int main
    (
    int ac,
    char **av
    )
{
    if(ac != 2)
        {
        cout << "usage: ./objdet [weight file]" << endl;
        return EXIT_FAILURE;
        }

    objdetect_init(av[1]);

#ifdef PIC_DEBUG
    Mat im = imread("1.jpg");
#else
    Mat im;
    VideoCapture cap;
    cap.open(0);
    if(!cap.isOpened())
        {
        cout << "failed to open camera" << endl;
        return EXIT_FAILURE;
        }
    while (true)
        {
        cap >> im;
#endif
        int *out = objdetect_main(im.data, im.cols, im.rows);
        for(int i = 0; i < out[0]; ++i)
            {
            rectangle(im, Rect(out[i*4+1], out[i*4+2], out[i*4+3], out[i*4+4]), Scalar(0, 0, 255), 2);
            }
        imshow("demo", im);

#ifdef PIC_DEBUG
        waitKey(0);
#else
        if(waitKey(5) == 27)
            {
            break;
            }
        }
#endif

    objdetect_free();
    return EXIT_SUCCESS;
}
