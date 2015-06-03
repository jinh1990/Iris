#ifndef MY_IMAGE_UTILS_H
#define MY_IMAGE_UTILS_H

#include <opencv2\core\core.hpp>
#include <opencv2/gpu/gpu.hpp>

#define INF 0.0000001
#define PI 3.141592654

using namespace std;
using namespace cv;

#define CONVOLUTION_FULL 1
#define CONVOLUTION_SAME 2
#define CONVOLUTION_VALID 4

int SSREnhancement(const cv::gpu::GpuMat src, Mat &dst);

int SSREnhancement(Mat src, Mat &dst);

int StructureMap(const char *subjName, Mat src, float lmbdax, float threshx, Mat &dstx);

int convolute(Mat src, Mat filter, Mat &dst);

int scale255(Mat src, Mat &dst);

int thin(Mat src, Mat &dst);
int thin2(Mat src, Mat &dst);

int calc_threshold(Mat src, float low_percents, float high_percents, int &low_thresh, int &high_thresh);

int reflection_removal(Mat src, Mat &mask, Mat &dest);

int resize_small(Mat src, Mat &dst);

int LeastSquares(vector<cv::Point> points, cv::Point& center, int& rds);

int interp2(Mat src, double* xo, double* yo, int width, int height, Mat& dst);

int findCenter(Mat edgemap, int* radiusRange, int points_gap, Point &center_point);

void find_radius(Mat edgemap, Point center_point, int* radiusRange, int searchRange, Point& output_center, int& output_radius);

//Find iris radius using counting distance between each edge points to pupil center
int find_iris_radius(Mat edgemap, Point center_point, int* radiusRange, int searchRange, Point& output_center, int& output_radius);

//Find pupil radius using counting distance between each edge points to pupil center
int find_pupil_radius(Mat edgemap, Point center_point, int* radiusRange, int searchRange, Point& output_center, int& output_radius);

void mask_lower_region(Mat img, Point center, int radius, double* extend, Mat& mask, double* thresh_high, double* thresh_low, double& cir_correct);

void mask_upper_region(Mat img, Mat& mask, Point center, int radius, double* thresh);

void thresh_angle_range(Mat img, Point center, double* radius_range, double* angles, double& thresh, double& thresh_low, double& quality);

void get_sector_region(Point center, double* radii, double* angles, int* size, Mat& region);

void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);

void cal_hist_thresh( Mat masked_img, double& thresh_high, double& thresh_low, double& quality);

void segment_angle_range(Mat img, Mat& mask, Point center, int radius, double* angles, double thresh, double* radius_range);

void get_pupil_region( Mat img, Mat reflection, Point center, int radius, double* thresh);

//Fitting lower eyelid using poly fitting
int fit_lower_eyelid( Mat img, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, double* range, double* thresh_canny, Mat& coffs);

//Fitting upper eyelid using poly fitting
int fit_upper_eyelid( Mat& img, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, double* range, double offset, bool save_image, Mat& coffs);

//remove small area blobs, which area less than size
void removeSmallBlobs(cv::Mat& im, double size);

int process_ES_region( Mat img, Mat& mask, Point center, int rds, Mat& coffs, double* thresh);

//calculate soble at both x, y direction, and combine by weight
void soble_double_direction(Mat img, Mat mask, double thresh, Mat& result);

//Locate eyelash ad shadow pixels
int eyelash_pixels_location(Mat& src, Point iris_center, int iris_rds, Point pupil_center, int pupil_rds, Vector<cv::Point>& eyelash_points);

//Image preprecessing before iris location
int iris_preprecessing(Mat src, Point pupil_center, int pupil_rds,Mat& iris_mask);

int get_rtv_l1_contour(Mat img, Mat& edgemap, Mat& im_smooth);

int rtv_l1_smooth2(Mat img, double lambda, double theta, double sigma, double ep, double maxIter, Mat& dst);

Mat conv2(const Mat &img, const Mat& ikernel, int type);

int computeU(Mat& u, Mat v, Mat f,double lambda,double theta, double sigma, double ep, int k);

int spdiags(Mat src, SparseMat& sparse_dst, int* d, int m, int n);

void cholesky_decomposition( const Mat& A, Mat& L );

int computeV(Mat u, Mat f, Mat& v, double theta);

int cal_energy(Mat im_s, Mat im, double lambda, Mat G, double ep, Mat& rtv_e);

int fspecial(double sigma, Mat& G_kernal);

int gau_filter(Mat in, Mat g, Mat& dst);

int iris_normalization(Mat& src, Mat& dst, Mat& iris_mask, Mat& mask_dst, Point pupil_center, int pupil_rds, Point iris_center, int iris_rds, int radpixels, int angulardiv);

int cal_thresh_using_hist(Mat src, double low_thresh_per, double high_thresh_per, int& low_thresh, int& high_thresh);

#endif