#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>

///OPENCV
#include "cxcore.h"
#include <cv.h>
#include <highgui.h>
#include "opencv2/contrib/contrib.hpp"
#include<opencv2/opencv.hpp>
using namespace std;

using namespace cv;


int getdir (string dir, vector<string> &files)
{
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    cout << "Error(" << errno << ") opening " << dir << endl;
    return errno;
  }
  
  while ((dirp = readdir(dp)) != NULL) {
    files.push_back(string(dirp->d_name));
  }
  closedir(dp);
  
  return 0;
}


int main(int argc, char **argv) {
  
  
  string dir,file_im_gray1,base_image;
  vector<string> file_img = vector<string>();
  Mat image,image_seg_rgb,image_seg_ycbcr1,image_seg_ycbcr2,image_seg_HSV,image_seg_HLS; //(almost) raw frame
  Mat ycbcr;
  Mat HSV;
  Mat HLS;
  
  
  
  /// read arguments
  if (argc == 2) {
    dir = argv[1];
    
  }
  else{
    cerr<<"Missing argument try ./color_seg <PATH_DATASET>"<<endl;
    cerr<<"or  try ./color_seg /home/aortega/proyectos/fire/data/fire_sequences/Securite_civile/Exp1/visible/" <<endl;
    
    return(0);
  }
  
  
  getdir(dir,file_img);
  
  
  //sort data
  std::sort(file_img.begin(),file_img.end());
  for (unsigned int i = 0;i < file_img.size();i++) 
  {
    if(file_img[i].substr(file_img[i].find_last_of(".") + 1) == "bmp")//our version     
    {
      
      
      file_im_gray1=dir +"/"+ file_img[i];
      cout <<"Reading images..."<<file_im_gray1 <<endl;
      
      image = imread(file_im_gray1,CV_LOAD_IMAGE_COLOR);
      
      cout<<"frame "<< i<<endl;
      cv::cvtColor(image,ycbcr,CV_RGB2YCrCb,0);
      cv::cvtColor(image,HSV,CV_RGB2HSV_FULL,0);
       cv::cvtColor(image,HLS,CV_RGB2HLS_FULL,0);
      
      image_seg_rgb=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_ycbcr1=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_ycbcr2=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HSV=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HLS=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      
      int c0=22987;
      int c1=-11698;
      int c2=-5636;
      int c3=29049;
      
      int val_y=0;
      int val_cr=0;
      int val_cb=0;
      
      for (unsigned int ii_=0; ii_<image.rows;ii_++)
	for (unsigned int jj_=0; jj_<image.cols;jj_++)
	{
	  val_y=val_y+ycbcr.at<cv::Vec3b>(ii_,jj_)[0];
	  val_cb=val_cb+ycbcr.at<cv::Vec3b>(ii_,jj_)[1];
	  val_cr=val_cr+ycbcr.at<cv::Vec3b>(ii_,jj_)[2];
	}
	
	float Y_mean=val_y/(image.rows*image.cols);  
      float Cb_mean=val_cb/(image.rows*image.cols);
      float Cr_mean=val_cr/(image.rows*image.cols);
      
      // cout<<"Y mean "<<Y_mean<<" Cr_mean "<<Cr_mean<<"Cb_mean"<<Cb_mean<<endl;
      
      for (unsigned int ii=0; ii<image.rows;ii++)
	for (unsigned int jj=0; jj<image.cols;jj++)
	{
	  int r = image.at<cv::Vec3b>(ii,jj)[2];
	  int g=  image.at<cv::Vec3b>(ii,jj)[1];
	  int  b= image.at<cv::Vec3b>(ii,jj)[0];
	  
	  int y = ycbcr.at<cv::Vec3b>(ii,jj)[0];
	  int Cb=  ycbcr.at<cv::Vec3b>(ii,jj)[1];
	  int  Cr= ycbcr.at<cv::Vec3b>(ii,jj)[2];
	  
	  int H = HSV.at<cv::Vec3b>(ii,jj)[0];
	  int S=  HSV.at<cv::Vec3b>(ii,jj)[1];
	  int V= HSV.at<cv::Vec3b>(ii,jj)[2];
	  
	  int H2 = HLS.at<cv::Vec3b>(ii,jj)[0];
	  int I2=  HLS.at<cv::Vec3b>(ii,jj)[1];
	  int S2= HLS.at<cv::Vec3b>(ii,jj)[2];
	  
	  // cv::Vec3f pixel_ycbcr = ycbcr.at<cv::Vec3f>(ii,jj);
	  //cout<<"R "<<r<<" G "<<g<<" B "<<b<<endl;
	  // cout<<"S "<<S<<endl;
	  // BGR
	  if((r>g) && (g>b) && (r>160)){
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  //YCRCB
	  if((y>Cb) && (Cr>Cb))
	  {
	    image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_ycbcr1.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  
	  
	  if((y>Y_mean) && (Cr>Cr_mean)&& (Cb>Cb_mean))
	  {
	    
	    image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_ycbcr2.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  
	  if(S>=((255-r)*65/135))
	  {
	    
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	    if(S2>=((255-r)*65/135))
	  {
	    
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	}
	
	
	
	imshow("frame ",image);
	imshow("Segmentation rgb",image_seg_rgb);
	imshow("Segmentation YCrCb",image_seg_ycbcr1);
	
	imshow("Segmentation YCrCb2",image_seg_ycbcr2);
	imshow("Segmentation HSV",image_seg_HSV);
	imshow("Segmentation HLS",image_seg_HLS);
	
	
	if(cv::waitKey(1000) >= 0) break;
    }
  }
  
  
  return 0;
}
