#include "cxcore.h"
#include <cv.h>
#include <highgui.h>
#include "opencv2/contrib/contrib.hpp"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double MCC(Mat Im_ref,Mat Im_seg);
double F1score(Mat Im_ref,Mat Im_seg);
double Hafiane(Mat Im_ref,Mat Im_seg);

