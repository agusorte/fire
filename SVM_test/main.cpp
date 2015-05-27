
///OPENCV
#include "cxcore.h"
#include <cv.h>
#include <highgui.h>
#include "opencv2/contrib/contrib.hpp"
#include<opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>


#include "utilities.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv) {
  
  ///Path data
  string dir,file_im_gray1,base_image,dir_test;
  vector<string> file_img = vector<string>();
  vector<string> file_img_test = vector<string>();
  
  ///COLOR segmentation
  Mat image,image_seg_rgb,image_seg_ycbcr1,image_seg_ycbcr2,image_seg_HSV,image_seg_HLS; //(almost) raw frame
  Mat ycbcr;
  Mat HSV;
  Mat HLS;
  Mat Union_color_gmm;
  
  ///GMMM
  Mat frame; //(almost) raw frame
  
  ///SVM
  CvSVM SVM;
  
  //SVM
  
  
  /// read arguments
  if (argc == 3) {
    dir = argv[1];
    dir_test = argv[2];
  }
  else{
    cerr<<"Missing argument try ./svn_test <PATH_TRAININGDATASET><PATH_TESTDATASET>"<<endl;
    cerr<<"or  try ./svn_test /home/aortega/proyectos/fire/data/fire_sequences/Training/ /home/aortega/proyectos/fire/data/fire_sequences/Letia/visible/Exp1/" <<endl;
    ///media/1606C02006C0032B/Documents and Settings/Agustin/Experiments_fire/4dec2013/exp4_12_13/Exp1/Fire/Cam2 rgb_12.04.13
    return(0);
  }
  
  
  
  getdir(dir,file_img);
  
  //sort data
  std::sort(file_img.begin(),file_img.end());
  
  
  // Set up SVM's parameters
  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
  unsigned long iter=0;
  
 // int N_images=5;//10 is good
   int N_images=15;//10 is good
  unsigned long all;
  
  file_im_gray1=dir +"/"+ file_img[6];
  frame = imread(file_im_gray1,CV_LOAD_IMAGE_COLOR);
  
  Mat labelsMat(frame.rows*frame.cols*N_images, 1, CV_32FC1);
  Mat trainingDataMat(frame.rows*frame.cols*N_images, 3, CV_32FC1);
  
 
  for (unsigned int i = 0;i < N_images;i++) 
  // for (unsigned int i = 0;i < N_images-2;i++) 
  {
    cout<<"data-->  "<<file_img[i]<<endl;
    cout<<i<<endl;
    if(file_img[i].substr(file_img[i].find_last_of(".") + 1) == "bmp" || 
      file_img[i].substr(file_img[i].find_last_of(".") + 1) == "ppm"){
      
      file_im_gray1=dir +"/"+ file_img[i];
  //  cout <<"Reading images..."<<file_im_gray1 <<endl<<endl;
    
    frame = imread(file_im_gray1,CV_LOAD_IMAGE_COLOR);
    
    //image_seg_rgb   = Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
    
    
    //all=frame.rows*frame.cols*file_img.size();
    
    for (unsigned int ii=0; ii<frame.rows;ii++){
      for (unsigned int jj=0; jj<frame.cols;jj++)
      {
	
	int r = frame.at<cv::Vec3b>(ii,jj)[2];
	int g = frame.at<cv::Vec3b>(ii,jj)[1];
	int b = frame.at<cv::Vec3b>(ii,jj)[0];
	///////////////////////////////////////////////
	/// BGR
	//////////////////////////////////////////////
	if((r>g) && (g>b) && (r>160))
	  labelsMat.at<float>(iter,0)=1;//this is fire 1
	else///classify as non fire
	  labelsMat.at<float>(iter,0)=-1;//this is fire 1

	trainingDataMat.at<float>(iter,0)=r;
	trainingDataMat.at<float>(iter,1)=g;
	trainingDataMat.at<float>(iter,2)=b;
	
	iter=iter+1;
      }
 
      // cout<<"iter  "<<iter<<endl;
    }
    
    // 
      }
      
      
      
  }

  ///trainign data
  
  cout<<"Training data ......"<<endl;
  SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
  
  string name_svn_data;
  stringstream ss2;
    ss2 << (N_images);
  name_svn_data="Training_sec_civil2_"+ss2.str()+".dat";
  
  SVM.save(name_svn_data.c_str());
  
  getdir(dir_test,file_img_test);
  
  //sort data
  std::sort(file_img_test.begin(),file_img_test.end());
  
  for (unsigned int i = 0;i < file_img_test.size();i++) 
  {
    if(file_img_test[i].substr(file_img_test[i].find_last_of(".") + 1) == "bmp" || 
      file_img_test[i].substr(file_img_test[i].find_last_of(".") + 1) == "ppm"){
      
      
      file_im_gray1=dir_test +"/"+ file_img_test[i];
    frame = imread(file_im_gray1,CV_LOAD_IMAGE_COLOR);
    cout <<"Reading images..."<<file_im_gray1 <<endl<<endl;
    image_seg_rgb   = Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
    
    for (unsigned int ii=0; ii<frame.rows;ii++){
      for (unsigned int jj=0; jj<frame.cols;jj++)
      {
	int r = frame.at<cv::Vec3b>(ii,jj)[2];
	int g = frame.at<cv::Vec3b>(ii,jj)[1];
	int b = frame.at<cv::Vec3b>(ii,jj)[0];
	
	
	Mat sampleMat = (Mat_<float>(1,3) << r,g,b);
	float response = SVM.predict(sampleMat);
	
	if (response == 1){
	  image_seg_rgb.at<Vec3b>(ii,jj)[2]  = r;
	  image_seg_rgb.at<Vec3b>(ii,jj)[1]  = g;
	  image_seg_rgb.at<Vec3b>(ii,jj)[0]  = b;
	}
	
      }
    }
    
    imshow("frame ",frame); 
    imshow("Segmentation rgb",image_seg_rgb);
    
    if(cv::waitKey(10) >= 0) break;
    
    string path="/home/aortega/proyectos/fire/Experiments_SVM/";
    stringstream ss;
    ss << i;
    ///save images
     imwrite(  path+"seg_svn" +ss.str()+".jpg", image_seg_rgb);
  
    
  }
 }
  
  
  return 0;

}
