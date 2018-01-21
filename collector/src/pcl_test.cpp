#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

#include <pcl/filters/conditional_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


ros::Publisher pub;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input){

  // Convert to PCL
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*input, *cloud);

  // Downsampling
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled (new pcl::PointCloud<pcl::PointXYZRGB>);
//  pcl::VoxelGrid<pcl::PointXYZRGB> vg;
//  vg.setInputCloud (cloud);
//  vg.setLeafSize (0.003f, 0.003f, 0.003f);   // 0.003 is the limit
//  vg.filter (*cloud_downsampled);

  // Passthrough filter
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_passthrough (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits (0.0, 1.3);
  pass.setFilterFieldName("y");
  pass.setFilterLimits (-0.3, 1.0);
  //pass.setFilterFieldName("x");
  //pass.setFilterLimits (-0.2, 0.2);
  pass.filter (*cloud_passthrough);

  // Remove the table
  // RANSAC
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
  double z_min = 0.063, z_max = 0.4;
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

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rmturntable (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor_2;
  sor_2.setInputCloud (cloud_rmtable);
  sor_2.setMeanK (50);
  sor_2.setStddevMulThresh (1.0);
  sor_2.filter (*cloud_rmturntable);

// Output and publish
  pcl::PointCloud<pcl::PointXYZRGB> output(*cloud_rmturntable);
  //pcl::io::savePCDFileASCII ("xyz_pcd.pcd", output);
  pub.publish(output);
}


int main(int argc, char **argv){

  // Initialize ROS
  ros::init(argc, argv, "pcl_listener");
  ros::NodeHandle nh;

  // Creat a ROS subscriber for the input cloud
  ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

  // Spin
  ros::spin();

  return 0;
}
