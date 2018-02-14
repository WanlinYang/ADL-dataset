#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <limits.h>
#include <cstdlib>

std::string category = "no input";
std::string ROOT_PATH;

typedef enum{IDLE, RECORD_BACK, MASK_VIEW, CROP, POSE} Mode_t;
static Mode_t Mode = IDLE;

ros::Publisher pub;
static int output_count = 0;

cv::Mat frame;
cv::Mat background;
cv::Mat object;
cv::Mat mask;
cv::Mat mask_blur;
cv::Mat mask_ellipse;
cv::Mat mask_overlay;
cv::Mat frame_crop;
cv::Mat mask_crop;
cv::Mat depth;
cv::Mat depth_crop;
cv::Rect boundRect;
int keyboard;

bool verbose = true;
bool said = false;

const int SEMI_MINOR = 72;  // y-axis
const int SEMI_MAJOR = 88;   // x-axis
const int THRESHOLD_MAX = 100; // trackbar range
int threshold = 30; // dafault value, could be adjusted by trackbar

void subtractBackground();

void manuallyboundRect(){
/*
For some white objects such as pill bottle,
denoting the bounding box by hand seems better
*/
  boundRect.x = 295;
  boundRect.y = 180;
  boundRect.width = 50;
  boundRect.height = 70;
}

std::string getexepath(){
/*
Get the path
*/
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  return std::string(result, (count > 0) ? count : 0);
}

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input){

  // Convert to PCL
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*input, *cloud);

  // PassThrough filter to narrow the view
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_passthrough (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.3);
  pass.setFilterFieldName ("y");
  pass.setFilterLimits (-0.3, 1.0);
  //pass.setFilterFieldName ("x");
  //pass.setFilterLimits (-0.2, 0.2);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*cloud_passthrough);

  if(Mode == MASK_VIEW || Mode == CROP){
    // Remove the table
    // remove the planar model with RANSAC
    pcl::ModelCoefficients::Ptr coeff_table (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inlier_table (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg_table;
    seg_table.setOptimizeCoefficients (true);
    seg_table.setModelType (pcl::SACMODEL_PLANE);
    seg_table.setMethodType (pcl::SAC_RANSAC);
    seg_table.setDistanceThreshold (0.01);
    seg_table.setInputCloud (cloud_passthrough);
    seg_table.segment (*inlier_table, *coeff_table);
    // Projection
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ProjectInliers<pcl::PointXYZRGB> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setIndices (inlier_table);
    proj.setInputCloud (cloud_passthrough);
    proj.setModelCoefficients (coeff_table);
    proj.filter (*cloud_projected);
    // Hull
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ConvexHull<pcl::PointXYZRGB> chull;
    chull.setInputCloud (cloud_projected);
    chull.reconstruct (*cloud_hull);
    // Extract Polygonal Prism
    double z_min = 0.063, z_max = 0.3; // 0.067 for plate
    pcl::PointIndices::Ptr prism_indices (new pcl::PointIndices);
    pcl::ExtractPolygonalPrismData<pcl::PointXYZRGB> prism;
    prism.setInputCloud (cloud_passthrough);
    prism.setInputPlanarHull (cloud_hull);
    prism.setHeightLimits (z_min, z_max);
    prism.segment (*prism_indices);
    // Remian the part between z_min and z_max
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rmtable (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> extract_table;
    extract_table.setInputCloud (cloud_passthrough);
    extract_table.setIndices (prism_indices);
    extract_table.setNegative (false);
    extract_table.filter (*cloud_rmtable);

    // Euclidean cluster segmentation
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGB> ());
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.0065);

    int nr_points = (int) cloud_rmtable->points.size();
    // While 60% of the original cloud is still there
    while (cloud_rmtable->points.size() > 0.5*nr_points){
      // Segment the largest plane from the remaining cloud
      seg.setInputCloud (cloud_rmtable);
      seg.segment (*inliers, *coefficients);
      if (inliers->indices.size() == 0){
        break;
      }
      // Extract the planar inliers from the input cloud
      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      extract.setInputCloud (cloud_rmtable);
      extract.setIndices (inliers);
      extract.setNegative (false);
      // Get the points associated with the planar surface
      extract.filter (*cloud_plane);
      // Remove the planar inliers, extract the rest
      extract.setNegative (true);
      extract.filter (*cloud_seg);
      *cloud_rmtable = *cloud_seg;
    }

    // remove outlier
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rmtable_2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_2;
    sor_2.setInputCloud (cloud_rmtable);
    sor_2.setMeanK (50);
    sor_2.setStddevMulThresh (1.0);
    sor_2.filter (*cloud_rmtable_2);

    // Output to pcd. file and publish to rviz
    pcl::PointCloud<pcl::PointXYZRGB> output(*cloud_rmtable_2);

    if(Mode == CROP){
      std::stringstream ss_pcd, ss_pcdcrop;
      ss_pcdcrop << ROOT_PATH << "data/segmented_pcd/"
                 << category <<"_"<< output_count << ".pcd";
      pcl::io::savePCDFile (ss_pcdcrop.str(), output, true);  // save as binary
    }
    pub.publish(output);
  }
  if (Mode == IDLE || Mode == RECORD_BACK || Mode == POSE){
    pcl::PointCloud<pcl::PointXYZRGB> output(*cloud_passthrough);
    pub.publish(output);
  }
}

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
      // do nothing
      {
        cv::ellipse(frame, cv::Point(frame.cols/2,frame.rows/2),
                    cv::Size(SEMI_MAJOR,SEMI_MINOR),0,0,360,
                    cv::Scalar(0,0,255),1,8);
        cv::imshow("frame", frame);
        if (verbose && !said){
          std::cout << "Press b to record background imgae" << std::endl;
          said = true;
        }
      }
      break;

    case RECORD_BACK:
      // record background
      {
        background = frame;
        std::stringstream ss_background;
        ss_background << ROOT_PATH << "data/"<< category<<"_"<<"background.png";
        cv::imwrite(ss_background.str(), background);
        if (verbose && !said){
          std::cout << "Background image is recoreded" << std::endl;
          std::cout << "Press m to view mask" << std::endl;
          said = true;
        }
      }
      break;

    case MASK_VIEW:
      // subtract background, show mask and rectangle
      {
        cv::namedWindow("mask", 1);
        cv::createTrackbar("threshold", "mask", &threshold, THRESHOLD_MAX);
        subtractBackground();
        //manuallyboundRect();

        cv::rectangle(frame, boundRect, cv::Scalar(0,255,100),2,8,0);
        cv::imshow("frame", frame);
        cv::imshow("mask", mask_overlay);
        if (verbose && !said){
          std::cout << "Slide to adjust threshold" << std::endl;
          std::cout << "Press c to record" << std::endl;
          said = true;
        }
      }
      break;

    case CROP:
      // crop image and imwrite
      {
        subtractBackground();
        //manuallyboundRect();

        frame_crop = frame(boundRect);
        mask_crop = mask(boundRect);
        int x_1 = boundRect.x;
        int y_1 = boundRect.y;
        int x_2 = boundRect.x + boundRect.width;
        int y_2 = boundRect.y + boundRect.height;

        std::stringstream ss_frame, ss_framecrop, ss_maskcrop, ss_loc;
        ss_frame << ROOT_PATH << "data/full_img/" << category
                 <<"_"<< output_count <<"_"<< "rgboriginal.png";
        ss_framecrop << ROOT_PATH << "data/cropped_img/"<< category
                     <<"_"<< output_count <<"_"<< "rgbcrop.png";
        ss_maskcrop << ROOT_PATH << "data/cropped_img/" << category
                    <<"_"<< output_count <<"_"<< "maskcrop.png";
        ss_loc << ROOT_PATH << "data/cropped_img/" << category
               <<"_"<< output_count <<"_"<< "loc.txt";

        cv::imshow("frame_crop", frame_crop);

        cv::imwrite(ss_frame.str(), frame);
        cv::imwrite(ss_framecrop.str(), frame_crop);
        cv::imwrite(ss_maskcrop.str(), mask_crop);

        std::ofstream locfile;
        locfile.open(ss_loc.str().c_str());
        locfile << x_1 << "," << y_1 <<"\n" << x_2 << "," << y_2 <<"\n";
        locfile.close();

        //output_count ++;

        cv::rectangle(frame, boundRect, cv::Scalar(0,255,100),2,8,0);
        cv::imshow("frame", frame);
        cv::imshow("mask", mask_overlay);
        if (verbose){
          std::cout << "Recording #" << output_count << std::endl;
          std::cout << "Press p to stop" << std::endl;
        }
      }
      break;

    case POSE:
      {
        float angle_step = 360.0/(output_count-1);
        for(int i=0; i<output_count; i++){
          std::stringstream ss_pose;
          ss_pose << ROOT_PATH << "data/pose/" << category
                  <<"_"<< i <<"_"<<"pose.txt";
          std::ofstream posefile;
          posefile.open(ss_pose.str().c_str());
          posefile << i*angle_step << "\n";
          posefile.close();
        }
        std::cout << "Positions are saved" << std::endl;
        Mode = IDLE;
      }
      break;

    default:
      {std::cout << "This is default" << std::endl;}

  }

  keyboard = cv::waitKey(30);
}

void depth_cb(const sensor_msgs::ImageConstPtr& input){
  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch(cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge depth exception: %s", e.what());
  }

  depth = cv_ptr->image;
  depth_crop = depth(boundRect);
  if(Mode == CROP){
    std::stringstream ss_depth, ss_depthcrop;
    ss_depth << ROOT_PATH << "data/full_img/" << category
             <<"_"<< output_count <<"_"<< "depthoriginal.png";
    ss_depthcrop << ROOT_PATH << "data/cropped_img/" << category
                 <<"_"<< output_count <<"_"<< "depthcrop.png";
    cv::imwrite(ss_depth.str(), depth);
    cv::imwrite(ss_depthcrop.str(), depth_crop);
    output_count ++;
  }

  if((char)keyboard == 'b'){
    Mode = RECORD_BACK;
    said = false;
  }else if((char)keyboard == 'm'){
    Mode = MASK_VIEW;
    said = false;
  }else if((char)keyboard == 'c'){
    Mode = CROP;
    said = false;
  }else if((char)keyboard == 'p'){
    Mode = POSE;
    said = false;
  }else if((char)keyboard == 'q'){
    Mode = IDLE;
    said = false;
  }
}


int main(int argc, char** argv){
  category = argv[1];
  std::cout << "category is " << category << std::endl;

  system("mkdir -p /home/wanlin/agitr/dddd");

  // Get work directory
  std::string full_path = getexepath();
  std::string ws_name = "ADL_dataset_ws";
  ROOT_PATH = full_path.substr(0, full_path.find(ws_name)
                                  + ws_name.length() + 1);
  // Make dir
  std::stringstream dir_cropped_img, dir_full_img, dir_pose, dir_segmented_pcd;
  dir_cropped_img << "mkdir -p " << ROOT_PATH << "data/cropped_img";
  dir_full_img << "mkdir -p " << ROOT_PATH << "data/full_img";
  dir_pose << "mkdir -p " << ROOT_PATH << "data/pose";
  dir_segmented_pcd << "mkdir -p " << ROOT_PATH << "data/segmented_pcd";

  system(dir_cropped_img.str().c_str());
  system(dir_full_img.str().c_str());
  system(dir_pose.str().c_str());
  system(dir_segmented_pcd.str().c_str());

  // Initialize ROS
  ros::init(argc, argv, "pcl_cv_datasb");
  ros::NodeHandle nh_pcl;
  ros::NodeHandle nh_img;
  ros::NodeHandle nh_depth;

  // Cloud subscriber, image subscriber, and depth subscriber
  ros::Subscriber pcl_sub = nh_pcl.subscribe("/camera/depth_registered/points", 1, cloud_cb);
  image_transport::ImageTransport it_img(nh_img);
  image_transport::Subscriber img_sub = it_img.subscribe("/camera/rgb/image_color", 1, image_cb);
  image_transport::ImageTransport it_depth(nh_depth);
  image_transport::Subscriber depth_sub = it_img.subscribe("/camera/depth/image_raw", 1, depth_cb);

  // Publish pointcloud to rviz
  pub = nh_pcl.advertise<sensor_msgs::PointCloud2> ("output", 1);

  ros::spin();
}

void subtractBackground(){

  cv::absdiff(frame, background, object);
  mask = cv::Mat::zeros(object.rows, object.cols, CV_8UC1);
  mask_ellipse = cv::Mat::zeros(object.rows, object.cols, CV_8UC1);
  mask_overlay = cv::Mat::zeros(object.rows, object.cols, CV_8UC1);
  cv::ellipse(mask_ellipse, cv::Point(object.cols/2,object.rows/2),
              cv::Size(SEMI_MAJOR,SEMI_MINOR),0,0,360,cv::Scalar(255,255,255),-1,8);

  float dist;
  for(int i=0; i<object.rows; i++){
    for(int j=0; j<object.cols; j++){
      cv::Vec3b pix = object.at<cv::Vec3b>(i,j);
      dist = (pix[0]*pix[0]+pix[1]*pix[1]+pix[2]*pix[2]);
      dist = sqrt(dist);

      if(dist > threshold)
        mask.at<unsigned char>(i,j) = 255;
      // remove shadow, assume BGR sequence
      // averge of shadow is R:26.5528 G:29.6117 B:29.7687
      if(pix[0]>=28 && pix[0]<=31 &&
          pix[1]>=28 && pix[1]<=31 &&
          pix[2]>=25 && pix[2]<=28)
        mask.at<unsigned char>(i,j) = 0;
    }
  }
  // erode and blur the mask
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                              cv::Size(2,2), cv::Point(-1,-1));
  cv::erode(mask, mask_blur, element);
  cv::blur(mask_blur, mask_blur, cv::Size(10,10),cv::Point(-1,-1));
  // overlay mask_ellipse
  mask_blur.copyTo(mask_overlay, mask_ellipse);
  // find contours
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask_overlay, contours, hierarchy, CV_RETR_EXTERNAL,
                   CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
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
