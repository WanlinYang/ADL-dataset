#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

typedef enum{IDLE, RECORD_BACK, MASK_VIEW, CROP} Mode_t;
static Mode_t Mode = IDLE;

cv::Mat frame;
cv::Mat background;
cv::Mat object;
cv::Mat mask;
cv::Mat mask_blur;
cv::Mat mask_ellipse;
cv::Mat mask_overlay;
cv::Mat cropped_img;
cv::Rect boundRect;
int keyboard;
const int SEMI_MINOR_MAX = 200;
const int SEMI_MAJOR_MAX = 200;
int semi_minor = 0;
int semi_major = 0;
const int THRESHOLD_MAX = 100;
int threshold = 0;

void subtractBackground();
void image_cb(const sensor_msgs::ImageConstPtr& input){
  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::BGR8);
  }
  catch(cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  frame = cv_ptr->image;
  switch(Mode){
    case IDLE:
      // draw ellipse
      {cv::namedWindow("frame", 1);
      cv::createTrackbar("semi major", "frame", &semi_major, SEMI_MAJOR_MAX);
      cv::createTrackbar("semi minor", "frame", &semi_minor, SEMI_MINOR_MAX);
      cv::ellipse(frame, cv::Point(frame.cols/2,frame.rows/2), cv::Size(semi_major,semi_minor),0,0,360,
          cv::Scalar(0,0,255),1,8);
      cv::imshow("frame", frame);
      }
      break;

    case RECORD_BACK:
      // record background
      {background = frame;}
      break;

    case MASK_VIEW:
      // subtract background, show mask and rectangle
      {
      cv::namedWindow("mask", 1);
      cv::createTrackbar("threshold", "mask", &threshold, THRESHOLD_MAX);

      subtractBackground();

      boundRect.x = 295;
      boundRect.y = 180;
      boundRect.width = 50;
      boundRect.height = 70;
      cv::rectangle(frame, boundRect, cv::Scalar(0,255,100),2,8,0);
      cv::imshow("frame", frame);
      cv::imshow("mask", mask_overlay);
      }
      break;

    case CROP:
      // crop image and imwrite
      {subtractBackground();

      cropped_img = frame(boundRect);
      cv::imshow("cropped_img", cropped_img);
      cv::rectangle(frame, boundRect, cv::Scalar(0,255,100),2,8,0);
      cv::imshow("frame", frame);
      cv::imshow("mask", mask_overlay);
      std::cout << boundRect.x << " " << boundRect.y << std::endl;
      std::cout << (boundRect.x+boundRect.width) << " " << (boundRect.y+boundRect.height) << std::endl;
      }
      break;

    default:
      {std::cout << "This is default" << std::endl;}

  }

  keyboard = cv::waitKey(30);

  if((char)keyboard == 'b'){
    Mode = RECORD_BACK;
  }else if((char)keyboard == 'm'){
    Mode = MASK_VIEW;
  }else if((char)keyboard == 'c'){
    Mode = CROP;
  }else if((char)keyboard == 'q'){
    Mode = IDLE;
  }

}

int main(int argc, char** argv){
  ros::init(argc, argv, "iamge_listener");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("camera/rgb/image_color", 1, image_cb);
  ros::spin();
}

void subtractBackground(){

  cv::absdiff(frame, background, object);
  mask = cv::Mat::zeros(object.rows, object.cols, CV_8UC1);
  mask_ellipse = cv::Mat::zeros(object.rows, object.cols, CV_8UC1);
  mask_overlay = cv::Mat::zeros(object.rows, object.cols, CV_8UC1);
  cv::ellipse(mask_ellipse, cv::Point(object.cols/2,object.rows/2),cv::Size(semi_major,semi_minor),
      0,0,360,cv::Scalar(255,255,255),-1,8);

  float dist;

  for(int i=0; i<object.rows; i++){
    for(int j=0; j<object.cols; j++){
      cv::Vec3b pix = object.at<cv::Vec3b>(i,j);
      dist = (pix[0]*pix[0]+pix[1]*pix[1]+pix[2]*pix[2]);
      dist = sqrt(dist);

      if(dist > threshold)
        mask.at<unsigned char>(i,j) = 255;
    }
  }
  // erode and blur the mask
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2), cv::Point(-1,-1));
  cv::erode(mask, mask_blur, element);
  cv::blur(mask_blur, mask_blur, cv::Size(15,15),cv::Point(-1,-1));
  // overlay mask_ellipse
  mask_blur.copyTo(mask_overlay, mask_ellipse);
  // find contours
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask_overlay, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
  // find the biggest contour
  int max_area = 0;
  int max_contour_index = 0;
  for( int i = 0; i < contours.size(); i++){
    double a = contourArea( contours[i], false);
    if(a > max_area){
      max_area = a;
      max_contour_index = i;
      boundRect = cv::boundingRect( contours[i]);
    }
  }

}
