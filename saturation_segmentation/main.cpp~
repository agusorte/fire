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
  Mat Gradient_density_gmm;
  std::vector<std::vector<cv::Point> > contours;
  
  /// read arguments
  if (argc == 2) {
    dir = argv[1];
    
  }
  else{
    cerr<<"Missing argument try ./saturation_seg <PATH_DATASET>"<<endl;
    cerr<<"or  try ./saturation_seg /home/aortega/proyectos/fire/data/fire_sequences/Securite_civile/Exp1/visible/" <<endl;
    
    return(0);
  }
  
  
  getdir(dir,file_img);
  
  Mat element=getStructuringElement(cv::MORPH_ELLIPSE,Size(2,2));
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

      cv::cvtColor(image,HSV,CV_RGB2HSV_FULL,0);
       cv::cvtColor(image,HLS,CV_RGB2HLS_FULL,0);
      
      image_seg_rgb=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
       Gradient_density_gmm=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HSV=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
      image_seg_HLS=Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0,0,0));
    
      int st=55;
      int rt=115;
      for (unsigned int ii=0; ii<image.rows;ii++){
	for (unsigned int jj=0; jj<image.cols;jj++){
	  int r = image.at<cv::Vec3b>(ii,jj)[2];
	  int g=  image.at<cv::Vec3b>(ii,jj)[1];
	  int  b= image.at<cv::Vec3b>(ii,jj)[0];
	  
	 // cout<<"R "<<r<<" G "<<g<<" B "<<b<<endl;
	  int H = HSV.at<cv::Vec3b>(ii,jj)[0];
	  int S=  HSV.at<cv::Vec3b>(ii,jj)[1];
	  int V= HSV.at<cv::Vec3b>(ii,jj)[2];
	  
	  int H2 = HLS.at<cv::Vec3b>(ii,jj)[0];
	  int I2=  HLS.at<cv::Vec3b>(ii,jj)[1];
	  int S2= HLS.at<cv::Vec3b>(ii,jj)[2];
	  
	  
	  if((r>g) && (g>b) && (r>rt)){
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_rgb.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	  
	  
	  
	  if((S>=(((255-r)*st)/rt)) && (r>g) && (g>b) && (r>rt))
	  {
	    
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HSV.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	  
	    if(S2>=(((255-r)*st)/rt) && (r>g) && (g>b) && (r>rt))
	  {
	    
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[2]=r;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[1]=g;
	    image_seg_HLS.at<cv::Vec3b>(ii,jj)[0]=b;
	  }
	}
	
      }
	
	//gradient 
	 Mat im_bin_2=image_seg_HSV>0;
	 Mat canny_output;
	 Canny( image_seg_HSV, canny_output, 100, 255, 3 );
	 
	 cv::dilate(canny_output,canny_output,element);
	 //cv::erode(canny_output,canny_output,element);
	 
	 findContours(canny_output,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	 
	  cout<<"Regions Number ->"<<contours.size()<<endl;
	  
	 for(unsigned int i_=0;i_<contours.size();i_++){
	     Rect rec2=cv::boundingRect(contours[i_]);     
	     double val_sat=0;
	      for(unsigned int j_=0;j_<contours[i_].size();j_++){
		Point2f po=contours[i_][j_];
		int H = image_seg_HSV.at<cv::Vec3b>(po.y,po.x)[0];
		double S = image_seg_HSV.at<cv::Vec3b>(po.y,po.x)[1];
		int V = image_seg_HSV.at<cv::Vec3b>(po.y,po.x)[2];
		val_sat=val_sat+S;
		 //cout<<"H "<<H<<" S "<<S<<" V "<<V<<endl;
		//int H = canny_output.at<cv::Vec3b>(p,ii)[0];
		 cv::circle(Gradient_density_gmm, contours[i_][j_], 1, cv::Scalar(255, 0,0), CV_FILLED, CV_AA);
	      }

	    double per=cv::arcLength(contours[i_],true);
	    
	    if(per==0)
	     per=-10;
	    
	    double Ds=(val_sat)/per;

	    cout<<"DS "<<Ds<<" Val "<<val_sat<<" per "<<per<<endl;
	    
	 if(0.8<=Ds){
	      for(unsigned int ii=rec2.x; ii<=rec2.x+rec2.width;ii++){
		for(unsigned int jj=rec2.y; jj<=rec2.y+rec2.height;jj++){
		  int val= cv::pointPolygonTest(contours[i_],Point2f(ii,jj),false);
		  if(val==1){
 		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[2] = image.at<cv::Vec3b>(jj,ii)[2];
 		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[1] = image.at<cv::Vec3b>(jj,ii)[1];
 		    Gradient_density_gmm.at<cv::Vec3b>(jj,ii)[0] = image.at<cv::Vec3b>(jj,ii)[0];

		    
		  }
		}
	      }
	      
	   }
	    
	  }
	
	imshow("frame ",image);
	imshow("Segmentation rgb",image_seg_rgb);
	
	imshow("Segmentation HSV",image_seg_HSV);
	imshow("Segmentation HLS",image_seg_HLS);
	
	
	imshow("Gradiente ",Gradient_density_gmm);
	imshow("Canny ",canny_output);
	
	if(cv::waitKey(1000) >= 0) break;
    }
  }
  
  
  return 0;
}
