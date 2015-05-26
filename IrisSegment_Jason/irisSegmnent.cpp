#include "stdio.h"
#include "iostream"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>

#include <ctime>

#include "my_image_utils.h"
#include "my_iris_segmentation.h"

#include "fit_curves.h"



using namespace cv;
using namespace std;

void removeSmallBlobs(cv::Mat& im, double size);
void cal_canny_high_thresd(Mat src, float ratio, int& high_thresd);

void main()
{
	Mat img = imread("F:\\nir_face\\Fine images\\eyeImages\\015.jpg",1);
	//Mat img = imread("C:\\Users\\pebble\\Desktop\\66_Jun_Liu_iris_15_04_22_04_49_49.jpg",1);

	int rows = img.rows;
	int cols = img.cols;

	//Mat img = img1(Rect(Point(cols/4,rows/4),Point(cols*3/4,rows*3/4))).clone();

 	if (img.type() != CV_8UC1)
 	{
 		cvtColor(img,img,CV_BGR2GRAY);
 	}
	
	time_t cur_time, last_time;
	last_time = clock();

	//resize(img,img,Size(200,150));

	Mat img_enhanced;
	//equalizeHist(img,img);

 	medianBlur(img,img_enhanced,7);

	

	int high_threshold = 0;
	int low_threshold = 0;

 	calc_threshold(img_enhanced,0,0.9,low_threshold,high_threshold);

	Mat bw_img, bw_big;
	threshold(img,bw_img,high_threshold,255,0);

	bw_big = bw_img.clone();

	removeSmallBlobs(bw_big,1000);

	for(int i = 0; i < bw_img.rows; i++)
		for(int j = 0; j < bw_img.cols; j++)
		{
			if (bw_big.at<unsigned char>(i,j) != 0)
			{
				bw_img.at<unsigned char>(i,j) = 0;
			}
		}

	int dilation_type = MORPH_ELLIPSE;
	int dilation_size = 15;

	Mat element = getStructuringElement( MORPH_ELLIPSE, Size(dilation_size,dilation_size));

  /// Apply the dilation operation
	Mat dil_img;
	dilate( bw_img, dil_img, element );

	GaussianBlur(img_enhanced,img_enhanced,Size(7,7),0);

	//Mat img_smooth, edgemap;
	//get_rtv_l1_contour(img_enhanced,edgemap,img_smooth);

	//cvNamedWindow( "better_result", 1 );
	//cvNamedWindow( "img", 1 );
	//imshow("better_result", img_smooth);
	//imshow("img", edgemap);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );
	//cvDestroyWindow( "img" );

	//return;

	int canny_h_thresd = 0;
	cal_canny_high_thresd(img_enhanced,0.3,canny_h_thresd);

	int canny_l_threshold = canny_h_thresd*0.5;

	Mat canny_ret_img;
	Canny(img_enhanced, canny_ret_img, canny_l_threshold, canny_h_thresd);
//	Canny( img_enhanced, canny_ret_img, 50, 70);

	for(int i = 0; i < img_enhanced.rows; i++)
		for(int j = 0; j < img_enhanced.cols; j++)
		{
			if (dil_img.ptr<unsigned char>(i)[j] != 0)
			{
				canny_ret_img.ptr<unsigned char>(i)[j] = 0;
			}
		}

	//circle(img,Point(93,60),23,cv::Scalar(255));

	//cvNamedWindow( "better_result", 1 );
	//cvNamedWindow( "img", 1 );
	//imshow("better_result", img);
	//imshow("img", canny_ret_img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );
	//cvDestroyWindow( "img" );

	Mat canny_ret_copy = canny_ret_img.clone();

	int points_gap = 10;
	int radiusRange[2] = {20, 60};
	int searchRange = 10;

	//int points_gap = 5;
	//int radiusRange[2] = {10, 33};
	//int searchRange = 5;

	Point center_point(0,0);

	findCenter(canny_ret_copy,radiusRange,points_gap,center_point);

	//cout<<"center_point = "<<center_point<<endl;

	Point iris_center(0,0);
	Point pupil_center(0,0);
	int pupil_rds = 0;
	int iris_rds = 0;
	 
	find_radius(canny_ret_copy,center_point,radiusRange, searchRange,pupil_center,pupil_rds);

	cout<<"pupil_center = "<<pupil_center<<endl;
	cout<<"pupil_rds = "<<pupil_rds<<endl;

	circle(img,pupil_center,pupil_rds,cv::Scalar(255));

	cur_time = clock();
	double time_cost = cur_time - last_time;
	cout<<"  pupil location time_cost = "<<time_cost<<endl;
	last_time = clock();

	cout<<"44444"<<endl;

	Point left_top(((pupil_center.x - 8*pupil_rds) > 5?(pupil_center.x - 8*pupil_rds):5),((pupil_center.y - 8*pupil_rds) > 5?(pupil_center.y - 8*pupil_rds):5));
	Point right_bottom(((pupil_center.x + 8*pupil_rds) < (img_enhanced.cols -5)?(pupil_center.x + 8*pupil_rds):(img_enhanced.cols -5)),((pupil_center.y + 8*pupil_rds) < (img_enhanced.rows - 5)?(pupil_center.y + 8*pupil_rds):(img_enhanced.rows - 5)));
	
	Rect iris_area(left_top,right_bottom);
	Mat iris_img = img_enhanced(iris_area).clone();
	Mat iris_mask;

	GaussianBlur(iris_img,iris_img,Size(7,7),0);
	GaussianBlur(iris_img,iris_img,Size(7,7),0);

	iris_preprecessing(iris_img, Point(iris_area.width/2,iris_area.height/2), pupil_rds,iris_mask);

	double thresh = 15;
	Mat new_canny_result;
	soble_double_direction(iris_img,iris_mask,thresh,new_canny_result);

	removeSmallBlobs(new_canny_result,10);	

//	thin2(new_canny_result,new_canny_result);

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", new_canny_result);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );


	//Point left_top((pupil_center.x - 1.2*pupil_rds + cols/4),(pupil_center.y - 1.2*pupil_rds + rows/4));
	//Point right_bottom((pupil_center.x + 1.2*pupil_rds + cols/4),(pupil_center.y + 1.2*pupil_rds + rows/4));
	//Rect iris_area(left_top,right_bottom);
	//Mat iris_img = img(iris_area).clone();

	//calc_threshold(iris_img,0,0.7,low_threshold,high_threshold);

	//cout<<"333"<<endl;

	//threshold(iris_img,bw_img,high_threshold,255,0);

	////bw_big = bw_img.clone();

	////removeSmallBlobs(bw_big,1000);

	////for(int i = 0; i < bw_img.rows; i++)
	////	for(int j = 0; j < bw_img.cols; j++)
	////	{
	////		if (bw_big.at<unsigned char>(i,j) != 0)
	////		{
	////			bw_img.at<unsigned char>(i,j) = 0;
	////		}
	////	}

	//

	//element = getStructuringElement( MORPH_ELLIPSE, Size(dilation_size,dilation_size));

 // /// Apply the dilation operation
	//dilate( bw_img, dil_img, element );
	//

	////cvNamedWindow( "better_result", 1 );
	////imshow("better_result", iris_img);
	////cvWaitKey(0);
	////cvDestroyWindow( "better_result" );

	////cout<<"iris_center = "<<iris_center<<endl;
	//equalizeHist(iris_img,iris_img);

	//GaussianBlur(iris_img,iris_img,Size(7,7),0);
	//GaussianBlur(iris_img,iris_img,Size(7,7),0);
	//cout<<"222"<<endl;

	//cal_canny_high_thresd(iris_img,0.15,canny_h_thresd);

	//canny_l_threshold = canny_h_thresd*0.5;
	//Mat new_canny_result;
	//Canny( iris_img, new_canny_result, canny_l_threshold, canny_h_thresd);

	//for(int i = 0; i < iris_img.rows; i++)
	//	for(int j = 0; j < iris_img.cols; j++)
	//	{
	//		int distance = (i-pupil_center.y)*(i-pupil_center.y) + (j - pupil_center.x)*(j - pupil_center.x);
	//		if (abs(distance-pupil_rds*pupil_rds) < 0.5*0.5*pupil_rds*pupil_rds)
	//		{
	//			new_canny_result.at<unsigned char>(i,j) = 0;
	//		}
	//	}

	//for(int i = 0; i < iris_img.rows; i++)
	//	for(int j = 0; j < iris_img.cols; j++)
	//	{
	//		if (dil_img.ptr<unsigned char>(i)[j] != 0)
	//		{
	//			new_canny_result.ptr<unsigned char>(i)[j] = 0;
	//		}
	//	}

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", new_canny_result);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	Point new_pupil_center(pupil_center.x-iris_area.x,pupil_center.y-iris_area.y);

	int irisRange[2] = {(int)pupil_rds*1.5, (int)pupil_rds*7};
	find_iris_radius(new_canny_result,new_pupil_center, irisRange, searchRange,iris_center,iris_rds);

	//circle(new_canny_result,iris_center,iris_rds,cv::Scalar(255));

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", new_canny_result);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	cout<<"iris_center = "<<iris_center<<endl;
	cout<<"iris_rds = "<<iris_rds<<endl;
 
	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  pupil location time_cost = "<<time_cost<<endl;
	last_time = clock();


	//cur_time = clock();
	//time_cost = cur_time - last_time;

	Point rst_iris_center(iris_center.x+iris_area.x,iris_center.y+iris_area.y), rst_pupil_center((pupil_center.x),(pupil_center.y));
	int rst_iris_rds = iris_rds;
	int rst_pupil_rds = pupil_rds;

	circle(img,rst_iris_center,rst_iris_rds,cv::Scalar(255));
	circle(img,rst_pupil_center,rst_pupil_rds,cv::Scalar(255));

	

	//cvNamedWindow( "better_result", 1 );
	//imshow("better_result", img);
	//cvWaitKey(0);
	//cvDestroyWindow( "better_result" );

	//Mat coffs_lower(3,1,CV_32FC1), coffs_upper(3,1,CV_32FC1);
	Mat coffs_lower, coffs_upper;

	double range[2] = {0.4,1.1};
	fit_lower_eyelid(img_enhanced,rst_iris_center,rst_iris_rds,rst_pupil_center,rst_pupil_rds,range,coffs_lower);

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  lower eyelid location time_cost = "<<time_cost<<endl;
	last_time = clock();

	fit_upper_eyelid(img_enhanced,rst_iris_center,rst_iris_rds,rst_pupil_center,rst_pupil_rds,range,345,false,coffs_upper);

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  Upper eyelid location time_cost = "<<time_cost<<endl;
	last_time = clock();

	if (!coffs_lower.empty())
	{
		drawPolynomial(coffs_lower,1,img);
	}
	
	if (!coffs_upper.empty())
	{
		drawPolynomial(coffs_upper,2,img);
	}
	
	cout<<"After draw Poly"<<endl;

	Mat mask(img_enhanced.size(),CV_8UC1);
	//mask.setTo(255);

	//double thresh[2] = {0.1,0.8};
	//process_ES_region(img_enhanced, mask, rst_iris_center, rst_iris_rds, coffs_upper, thresh);

	Vector<cv::Point> eyelash_points;
	eyelash_pixels_location(img_enhanced,rst_iris_center,rst_iris_rds,pupil_center,pupil_rds,eyelash_points);
	mask.setTo(0);

	cur_time = clock();
	time_cost = cur_time - last_time;
	cout<<"  Mask pixels location time_cost = "<<time_cost<<endl;
	last_time = clock();

	for (int i = 0; i < eyelash_points.size(); i++)
	{
		img.ptr<unsigned char>(eyelash_points[i].y)[eyelash_points[i].x] = 255;
		mask.ptr<unsigned char>(eyelash_points[i].y)[eyelash_points[i].x] = 255;
	}

	//mask_lower_region(img_enhanced,iris_center,iris_rds,extend,mask,thresh_high,thresh_low,cir_correct);

	//imwrite("mask.bmp",mask);
	imwrite("F:\\nir_face\\Fine images\\eyeImages\\final_result.bmp",img);

	cvNamedWindow( "better_result", 1 );
	cvNamedWindow( "img", 1 );
	imshow("better_result", img);
	imshow("img", new_canny_result);
	cvWaitKey(0);
	cvDestroyWindow( "better_result" );
	cvDestroyWindow( "img" );

//	getchar();
}



void cal_canny_high_thresd(Mat src, float ratio, int& high_thresd)
{
	const int channels[1]={0};
    const int histSize[1]={256};
    float hranges[2]={0,255};
    const float* ranges[1]={hranges};
	Mat hist = Mat::zeros(256,1,CV_8UC1);
    calcHist(&src,1,channels,Mat(),hist,1,histSize,ranges);

	float num_thresd = src.cols*src.rows*ratio;
	
	int num_of_pixel = 0;
	for (int i = 0; i < 256; i++)
	{
		num_of_pixel += hist.at<float>(i,0);
		if (num_of_pixel > num_thresd)
		{
			high_thresd = i;
			break;
		}
		
	}
}