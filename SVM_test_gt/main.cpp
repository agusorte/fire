
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
#include "evaluation_criteria.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv) {
  
  ///Path data
  string dir,dir_test,file_im_gray1,base_image,base_im,base_im_seg;
  vector<string> file_img = vector<string>();
  vector<string> file_img_test = vector<string>();
   string im, im_seg;
  ///COLOR segmentation
  Mat image,image_seg_rgb,image_seg_ycbcr1,image_seg_ycbcr2,image_seg_HSV,image_seg_HLS; //(almost) raw frame
  Mat ycbcr;
  Mat HSV;
  Mat HLS;
  Mat Union_color_gmm;
  Mat  segmentation;
  Mat frame_gray;
  
   vector<vector<Point> > contours_rgb;
   
  ///GMMM
  Mat frame; //(almost) raw frame
  
  ///SVM
  CvSVM SVM;
  
  int n_im;
    
  
  
  //SVM
  ///Creating files
  ofstream myfile;
  myfile.open ("/home/aortega/proyectos/fire/data/Experiments_SVM/test.txt");
  
  /// read arguments
  if (argc == 6) {
       dir = argv[1];
       dir_test = argv[2];
     base_im = argv[3];
     base_im_seg = argv[4];
     n_im=atoi(argv[5]);
  }
  else{
    cerr<<"Missing argument try ./svn_test <PATH_TRAININGDATASET><PATH_DATASET><BASE_IMG_SEGMENTATION><BASE_GT><N_IMAGES>"<<endl;
    cerr<<"or  try ./svn_test_gt /home/aortega/proyectos/fire/data/fire_sequences_dynamic/Training/ /home/aortega/proyectos/fire/data/GroundTrouth/Labeled_sequences/Aerodrome1/ SC1_rgb.bmp SC1_gt.bmp 32" <<endl;
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
   int N_images=11;//10 is good
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
   
 double mean1=0;
 double mean2=0;
 double mean3=0;
  for (unsigned int i = 1;i <=n_im;i++){
     stringstream ss;
     ss << i;
    if(i<10){
       im=dir_test+"00"+ss.str()+base_im;
       im_seg=dir_test+"00"+ss.str()+base_im_seg;
     }else if(10<=i && i<100){
       im=dir_test+"0"+ss.str()+base_im;
       im_seg=dir_test+"0"+ss.str()+base_im_seg;
     }else{
       im=dir_test+ss.str()+base_im;
       im_seg=dir_test+ss.str()+base_im_seg;
     }
     
//       if(i<10){
//        im=dir_test+"0"+ss.str()+base_im;
//        im_seg=dir_test+"0"+ss.str()+base_im_seg;
//      }else if(10<=i && i<100){
//        im=dir_test+ss.str()+base_im;
//        im_seg=dir_test+ss.str()+base_im_seg;
//      }else{
//        im=dir_test+ss.str()+base_im;
//        im_seg=dir_test+ss.str()+base_im_seg;
//      }
     
      cout <<"Reading images..."<<im <<endl<<endl;
      cout <<"Reading images..."<<im_seg <<endl<<endl;
     /// Read image base
      frame = imread(im,CV_LOAD_IMAGE_COLOR);
      
       image_seg_rgb   = Mat(frame.rows,frame.cols,CV_8UC3,cv::Scalar(0,0,0));
      ///read seg_image
      
      segmentation=imread(im_seg,CV_LOAD_IMAGE_GRAYSCALE);
      
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
//     Mat im_bin_rgb=image_seg_rgb>0;
//     Mat canny_output;
//     Canny( im_bin_rgb, canny_output, 0, 255, 3 );
//     findContours(canny_output,contours_rgb,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    cvtColor(image_seg_rgb,frame_gray,CV_RGB2GRAY,0);
    double f1score_result=F1score(segmentation,frame_gray);
	double MCC_result=MCC(segmentation,frame_gray);  
	double hafiane_result=Hafiane(segmentation,frame_gray);
	stringstream ss_f1score,ss_Mcc,ss_hafiane;
	
	
	mean1=mean1+f1score_result;
	mean2=mean2+MCC_result;
	mean3=mean3+hafiane_result;
	
	ss_f1score << f1score_result;
	ss_Mcc << MCC_result;
	ss_hafiane  << hafiane_result;
	// write data
	
	string input;
	
	input=ss_f1score.str()+" "+ss_Mcc.str()+" "+ss_hafiane.str()+"\n";
	
	myfile<<input;
	
	imshow("Segmentation binary",image_seg_rgb);
	imshow("Reference",segmentation);
        if(cv::waitKey(1000) >= 0) break;
	
	string path="/home/aortega/proyectos/fire/data/Experiments_SVM/";
	stringstream sss;
	sss << i;
	///save images
	imwrite(  path+"seg_svn" +sss.str()+".jpg", image_seg_rgb);  
  }
  string input2;
  stringstream ss_f1score2,ss_Mcc2,ss_hafiane2;
	ss_f1score2 << mean1/n_im;
	ss_Mcc2 << mean2/n_im;
	ss_hafiane2  << mean3/n_im;
	input2="Mean "+ss_f1score2.str()+" "+ss_Mcc2.str()+" "+ss_hafiane2.str()+"\n";
	
	myfile<<input2;
	
	myfile.close();
  
  return 0;

}
