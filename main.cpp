#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include "objdetect_pub.h"

#define PIC_DEBUG 1

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
#ifdef PIC_DEBUG
    Mat im = imread("dog.jpg");
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
        objdetect_init(av[1]);
        objdetect_main(im.data, im.cols, im.rows);
        objdetect_free();

        imshow("demo", im);
#ifdef PIC_DEBUG
    waitKey(0);
#else
        if(waitKey(10) == 27)
            {
            break;
            }
        }
#endif
    return EXIT_SUCCESS;
}
