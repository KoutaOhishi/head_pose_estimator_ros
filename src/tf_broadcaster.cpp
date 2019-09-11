#include <iostream>
#include <cstdlib>
#include <fstream>
#include <time.h>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <limits>
#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>

//for image input
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Bool.h>
#include <ros/package.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>

// PCL specific includes
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

//  Visualization Marker
#include <visualization_msgs/Marker.h>

#include <head_pose_estimator/HeadPose.h>
#include <head_pose_estimator/FaceRect.h>

#include <float.h>
#define ARRAY_LENGTH(array) (sizeof(array) / sizeof(array[0]))

class TFBroadcaster{
private:
  ros::NodeHandle nh_;

  ros::Subscriber sub_point_cloud_;
  ros::Subscriber sub_head_pose_;

  ros::Publisher pub_marker_;

  tf::TransformBroadcaster br_;
  tf::TransformListener listener_;

  pcl::PointCloud<pcl::PointXYZ> cloud_local_;

  std::string cloud_topic_name_;
  std::string camera_frame_name_;
  std::string base_frame_name_;

  bool is_get_point_cloud_ = false;

public:
  TFBroadcaster(){
    ros::param::get("sub_cloud_name", cloud_topic_name_);
    ros::param::get("camera_frame_name", camera_frame_name_);
    ros::param::get("base_frame_name", base_frame_name_);
    ROS_INFO("cloud_topic_name:[%s]", cloud_topic_name_.c_str());
    ROS_INFO("camera_frame_name:[%s]", camera_frame_name_.c_str());
    ROS_INFO("base_frame_name:[%s]", base_frame_name_.c_str());

    sub_point_cloud_ = nh_.subscribe(cloud_topic_name_, 1, &TFBroadcaster::pointCloudCallback, this);
    sub_head_pose_ = nh_.subscribe("/head_pose_estimator/head_pose", 1, &TFBroadcaster::headPoseCallback, this);

    pub_marker_ = nh_.advertise<visualization_msgs::Marker>("/head_pose_estimator/marker", 10);

    ROS_INFO("tf_broadcaster initialize OK");
  }

  void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& input){
    pcl::fromROSMsg(*input, cloud_local_);

    if(cloud_local_.points.size() == 0){
      ROS_ERROR("No PointCloud");
      is_get_point_cloud_ = false;
      return;
    }
    else{
      is_get_point_cloud_ = true;
      // /ROS_INFO("Get PointCloud");
    }
  }//pointCloudCallback

  void headPoseCallback(const head_pose_estimator::HeadPose& msg){
    if( is_get_point_cloud_ == false ){
      ROS_WARN("Waiting point_cloud...");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud_transform;
    //std::string base_frame_name = "base_footprint";

    //座標変換ができるかどうかを確認
    bool is_possible_transform = listener_.canTransform(base_frame_name_, camera_frame_name_, ros::Time(0));
    if(is_possible_transform){
      pcl_ros::transformPointCloud(base_frame_name_, ros::Time(0), cloud_local_, camera_frame_name_, cloud_transform, listener_);
    }//座標変換成功

    else{
      ROS_WARN("Failure PointCloud Transform");
      cloud_transform = cloud_local_; //元の座標系のままで処理を行う
      base_frame_name_ = camera_frame_name_; //座標変換を行わないようにするためにフレーム名を変更する
    }//座標変換失敗

    pcl::PointXYZ object_pt;
    double shortest_distance = DBL_MAX;

    //BoundingBoxの中心付近のみの点群を用いるために、上下左右の端は無視する
    double point_thresh = 0.25;
    //double point_lower_rate = 0.4;
    //double point_upper_rate = 0.4;

    int shortest_distance_y = 0;

    //顔のbbox
    float face_rect_x = msg.face_rect.x;
    float face_rect_y = msg.face_rect.y;
    float face_rect_width = msg.face_rect.width;
    float face_rect_height = msg.face_rect.height;

    for(int temp_y = face_rect_height*point_thresh; temp_y < face_rect_height*(1.0-point_thresh); temp_y++)
    {
      int object_y = face_rect_y + temp_y;
      for(int temp_x = 0; temp_x < face_rect_width; temp_x++)
      {
        int object_x = face_rect_width + temp_x;
        if(cloud_transform.points[cloud_transform.width*object_y+object_x].x < shortest_distance)
        {
          shortest_distance = cloud_transform.points[cloud_transform.width*object_y+object_x].x;
          object_pt = cloud_transform.points[cloud_transform.width*object_y+object_x];
          shortest_distance_y = cloud_transform.width*object_y;
        }
      }
    }

    //同じ高さの点の平均値を算出
    pcl::PointXYZ object_ave_pt;
    double temp_count = 0;
    for(int temp_x = face_rect_width*point_thresh; temp_x < face_rect_width*(1-point_thresh); temp_x++)
    {
      double object_x = face_rect_x + temp_x;
      if(std::isnan(cloud_transform.points[shortest_distance_y+object_x].x)
      || std::isnan(cloud_transform.points[shortest_distance_y+object_x].y)
      || std::isnan(cloud_transform.points[shortest_distance_y+object_x].z)) { continue; }
      object_ave_pt.x += cloud_transform.points[shortest_distance_y+object_x].x;
      object_ave_pt.y += cloud_transform.points[shortest_distance_y+object_x].y;
      object_ave_pt.z += cloud_transform.points[shortest_distance_y+object_x].z;
      temp_count += 1;
    }

    if(temp_count != 0){
      object_ave_pt.x /= temp_count;
      object_ave_pt.y /= temp_count;
      object_ave_pt.z /= temp_count;
    }

    if(std::isnan(object_ave_pt.x)==false && std::isnan(object_ave_pt.y)==false && std::isnan(object_ave_pt.z)==false){
      br_.sendTransform(
        tf::StampedTransform(tf::Transform(tf::Quaternion(msg.head_rotation.x, msg.head_rotation.y, msg.head_rotation.z, msg.head_rotation.w),
        tf::Vector3(object_ave_pt.x,object_ave_pt.y,object_ave_pt.z)),
        ros::Time::now(),
        base_frame_name_,
        "head"));}

      geometry_msgs::Pose pose;
      //pose.position = geometry_msgs::Point(object_ave_pt.x,object_ave_pt.y,object_ave_pt.z);
      //pose.orientation = geometry_msgs::Quaternion(msg.head_rotation.x,msg.head_rotation.y,msg.head_rotation.z,msg.head_rotation.w);
      pose.position.x = object_ave_pt.x;
      pose.position.y = object_ave_pt.y;
      pose.position.z = object_ave_pt.z;

      pose.orientation.x = msg.head_rotation.x;
      pose.orientation.y = msg.head_rotation.y;
      pose.orientation.z = msg.head_rotation.z;
      pose.orientation.w = msg.head_rotation.w;

      publishMarker(pose);


  }//headPoseCallback

  void publishMarker(geometry_msgs::Pose pose){
    visualization_msgs::Marker marker;
    marker.header.frame_id = base_frame_name_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "head_arrow";
    marker.id = 0;

    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.lifetime = ros::Duration();

    marker.scale.x = 0.5;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    marker.pose.position = pose.position;
    marker.pose.orientation = pose.orientation;

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;

    pub_marker_.publish(marker);


  }//publishMarker

}; //class TFBroadcaster

int main(int argc, char** argv){
  ros::init(argc, argv, "tf_broadcaster");
  TFBroadcaster tb;
  ros::spin();
  return 0;
}
