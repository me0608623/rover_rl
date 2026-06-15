/*
    FILE: dynamicDetector.cpp
    ---------------------------------
    function implementation of dynamic osbtacle detector
*/
#include <onboard_detector/dynamicDetector.h>
#include <sstream>  // 動態障礙終端輸出用

namespace onboardDetector{
    dynamicDetector::dynamicDetector() : rclcpp::Node("dynamic_detector"){
        this->ns_ = "onboard_detector";
        this->hint_ = "[onboardDetector]";
        this->initParam();
        this->registerPub();
        this->registerCallback();
    }

    void dynamicDetector::initParam(){
        // localization mode
        this->localizationMode_ = this->getParamHelper<int>("localization_mode", 0);
        cout << this->hint_ << ": Localization mode (0=pose, 1=odom): " << this->localizationMode_ << endl;

        // visualization frame：發布座標 = localization 來源座標（odom mode 下是 odom frame），
        // 硬編 "map" 會在沒 NDT 時讓 RViz 全部畫不出來、有 NDT 時畫偏 map→odom 位移
        this->visFrame_ = this->getParamHelper<std::string>("vis_frame", "map");
        cout << this->hint_ << ": Visualization frame: " << this->visFrame_ << endl;

        // depth topic name
        this->depthTopicName_ = this->getParamHelper<std::string>("depth_image_topic", "/camera/depth/image_raw");
        cout << this->hint_ << ": Depth topic: " << this->depthTopicName_ << endl;

        // color topic name
        this->colorImgTopicName_ = this->getParamHelper<std::string>("color_image_topic", "/camera/color/image_raw");
        cout << this->hint_ << ": Color image topic: " << this->colorImgTopicName_ << endl;

        // lidar topic name
        this->lidarTopicName_ = this->getParamHelper<std::string>("lidar_pointcloud_topic", "/cloud_registered");
        cout << this->hint_ << ": Lidar pointcloud topic: " << this->lidarTopicName_ << endl;

        // pose / odom topic name
        this->poseTopicName_ = this->getParamHelper<std::string>("pose_topic", "/CERLAB/quadcopter/pose");
        this->odomTopicName_ = this->getParamHelper<std::string>("odom_topic", "/CERLAB/quadcopter/odom");
        if (this->localizationMode_ == 0){
            cout << this->hint_ << ": Pose topic: " << this->poseTopicName_ << endl;
        }
        else{
            cout << this->hint_ << ": Odom topic: " << this->odomTopicName_ << endl;
        }

        // depth intrinsics
        std::vector<double> depthIntrinsics = this->getParamHelper<std::vector<double>>("depth_intrinsics", std::vector<double>{});
        if (depthIntrinsics.size() != 4){
            RCLCPP_ERROR(this->get_logger(), "[dynamicDetector]: Please check depth camera intrinsics!");
        }
        else{
            this->fx_ = depthIntrinsics[0];
            this->fy_ = depthIntrinsics[1];
            this->cx_ = depthIntrinsics[2];
            this->cy_ = depthIntrinsics[3];
            cout << this->hint_ << ": fx, fy, cx, cy: [" << this->fx_ << ", " << this->fy_ << ", " << this->cx_ << ", " << this->cy_ << "]" << endl;
        }

        // color intrinsics
        std::vector<double> colorIntrinsics = this->getParamHelper<std::vector<double>>("color_intrinsics", std::vector<double>{});
        if (colorIntrinsics.size() != 4){
            RCLCPP_ERROR(this->get_logger(), "[dynamicDetector]: Please check color camera intrinsics!");
        }
        else{
            this->fxC_ = colorIntrinsics[0];
            this->fyC_ = colorIntrinsics[1];
            this->cxC_ = colorIntrinsics[2];
            this->cyC_ = colorIntrinsics[3];
            cout << this->hint_ << ": fxC, fyC, cxC, cyC: [" << this->fxC_ << ", " << this->fyC_ << ", " << this->cxC_ << ", " << this->cyC_ << "]" << endl;
        }

        // depth scale factor
        this->depthScale_ = this->getParamHelper<double>("depth_scale_factor", 1000.0);
        cout << this->hint_ << ": Depth scale factor: " << this->depthScale_ << endl;

        // depth min value
        this->depthMinValue_ = this->getParamHelper<double>("depth_min_value", 0.2);
        cout << this->hint_ << ": Depth min value: " << this->depthMinValue_ << endl;

        // depth max value
        this->depthMaxValue_ = this->getParamHelper<double>("depth_max_value", 5.0);
        this->raycastMaxLength_ = this->depthMaxValue_;
        cout << this->hint_ << ": Depth max value: " << this->depthMaxValue_ << endl;

        // depth filter margin
        this->depthFilterMargin_ = this->getParamHelper<int>("depth_filter_margin", 0);
        cout << this->hint_ << ": Depth filter margin: " << this->depthFilterMargin_ << endl;

        // depth skip pixel
        this->skipPixel_ = this->getParamHelper<int>("depth_skip_pixel", 1);
        cout << this->hint_ << ": Depth skip pixel: " << this->skipPixel_ << endl;

        // image columns / rows
        this->imgCols_ = this->getParamHelper<int>("image_cols", 640);
        cout << this->hint_ << ": Depth image columns: " << this->imgCols_ << endl;
        this->imgRows_ = this->getParamHelper<int>("image_rows", 480);
        cout << this->hint_ << ": Depth image rows: " << this->imgRows_ << endl;
        this->projPoints_.resize(this->imgCols_ * this->imgRows_ / (this->skipPixel_ * this->skipPixel_));
        this->pointsDepth_.resize(this->imgCols_ * this->imgRows_ / (this->skipPixel_ * this->skipPixel_));

        // transform matrix: body to camera depth
        std::vector<double> body2CamDepthVec = this->getParamHelper<std::vector<double>>("body_to_camera_depth", std::vector<double>{});
        if (body2CamDepthVec.size() != 16){
            RCLCPP_ERROR(this->get_logger(), "[dynamicDetector]: Please check body to camera depth matrix!");
        }
        else{
            for (int i=0; i<4; ++i){
                for (int j=0; j<4; ++j){
                    this->body2CamDepth_(i, j) = body2CamDepthVec[i * 4 + j];
                }
            }
        }

        // transform matrix: body to camera color
        std::vector<double> body2CamColorVec = this->getParamHelper<std::vector<double>>("body_to_camera_color", std::vector<double>{});
        if (body2CamColorVec.size() != 16){
            RCLCPP_ERROR(this->get_logger(), "[dynamicDetector]: Please check body to camera color matrix!");
        }
        else{
            for (int i=0; i<4; ++i){
                for (int j=0; j<4; ++j){
                    this->body2CamColor_(i, j) = body2CamColorVec[i * 4 + j];
                }
            }
        }

        // transform matrix: body to lidar
        std::vector<double> body2LidarVec = this->getParamHelper<std::vector<double>>("body_to_lidar", std::vector<double>{});
        if (body2LidarVec.size() != 16){
            RCLCPP_ERROR(this->get_logger(), "[dynamicDetector]: Please check body to lidar matrix!");
        }
        else{
            for (int i=0; i<4; ++i){
                for (int j=0; j<4; ++j){
                    this->body2Lidar_(i, j) = body2LidarVec[i * 4 + j];
                }
            }
        }

        // time step
        this->dt_ = this->getParamHelper<double>("time_step", 0.033);
        cout << this->hint_ << ": Time step for the system is set to: " << this->dt_ << endl;

        // ground height
        this->groundHeight_ = this->getParamHelper<double>("ground_height", 0.1);
        cout << this->hint_ << ": Ground height is set to: " << this->groundHeight_ << endl;

        // roof height
        this->roofHeight_ = this->getParamHelper<double>("roof_height", 2.0);
        cout << this->hint_ << ": Roof height is set to: " << this->roofHeight_ << endl;

        // lidar 最小距離（lidar frame XY 平面距離），濾掉車自身結構與 VLP-16 盲區雜點
        this->lidarMinRange_ = this->getParamHelper<double>("lidar_min_range", 0.0);
        cout << this->hint_ << ": Lidar min range is set to: " << this->lidarMinRange_ << endl;

        // 動態判定的最小淨位移（m），0=停用；濾掉無實際移動的尾流/鬼框誤判
        this->dynamicMinDisp_ = this->getParamHelper<double>("dynamic_min_displacement", 0.0);
        cout << this->hint_ << ": Dynamic min displacement is set to: " << this->dynamicMinDisp_ << endl;

        // 淨位移檢查窗（幀），0=自動取 2×consistency
        this->dynamicDispWindow_ = this->getParamHelper<int>("dynamic_disp_window", 0);
        cout << this->hint_ << ": Dynamic displacement window is set to: " << this->dynamicDispWindow_ << endl;

        // 位移檢查最少需要的幀數（少於此太短、不評估位移）
        this->dynamicMinDispFrames_ = this->getParamHelper<int>("dynamic_min_disp_frames", 6);
        cout << this->hint_ << ": Dynamic min disp frames is set to: " << this->dynamicMinDispFrames_ << endl;

        // LiDAR/odom 同步逾時（秒）：超過即清空動態輸出，避免發布過期偵測
        this->staleTimeout_ = this->getParamHelper<double>("detection_stale_timeout", 0.5);
        cout << this->hint_ << ": Detection stale timeout is set to: " << this->staleTimeout_ << endl;

        // min num of points for a voxel to be occupied in voxel filter
        this->voxelOccThresh_ = this->getParamHelper<double>("voxel_occupied_thresh", 10.0);
        cout << this->hint_ << ": Voxel occupied threshold is set to: " << this->voxelOccThresh_ << endl;

        // minimum number of points in each cluster
        this->dbMinPointsCluster_ = this->getParamHelper<int>("dbscan_min_points_cluster", 18);
        cout << this->hint_ << ": DBSCAN min points per cluster is set to: " << this->dbMinPointsCluster_ << endl;

        // search range
        this->dbEpsilon_ = this->getParamHelper<double>("dbscan_search_range_epsilon", 0.3);
        cout << this->hint_ << ": DBSCAN epsilon is set to: " << this->dbEpsilon_ << endl;

        // lidar dbscan min points
        this->lidarDBMinPoints_ = this->getParamHelper<int>("lidar_DBSCAN_min_points", 10);
        cout << this->hint_ << ": Lidar DBSCAN min points per cluster is set to: " << this->lidarDBMinPoints_ << endl;

        // lidar dbscan search range
        this->lidarDBEpsilon_ = this->getParamHelper<double>("lidar_DBSCAN_epsilon", 0.2);
        cout << this->hint_ << ": Lidar DBSCAN epsilon is set to: " << this->lidarDBEpsilon_ << endl;

        // lidar points downsample threshold
        this->downSampleThresh_ = this->getParamHelper<int>("downsample_threshold", 4000);
        cout << this->hint_ << ": Downsample threshold is set to: " << this->downSampleThresh_ << endl;

        // gaussian downsample rate
        this->gaussianDownSampleRate_ = this->getParamHelper<int>("gaussian_downsample_rate", 2);
        cout << this->hint_ << ": Gaussian downsample rate is set to: " << this->gaussianDownSampleRate_ << endl;

        // IOU threshold
        this->boxIOUThresh_ = this->getParamHelper<double>("filtering_BBox_IOU_threshold", 0.5);
        cout << this->hint_ << ": Threshold for bounding box IOU filtering is set to: " << this->boxIOUThresh_ << endl;

        // maximum match range
        this->maxMatchRange_ = this->getParamHelper<double>("max_match_range", 0.5);
        cout << this->hint_ << ": Max match range is set to: " << this->maxMatchRange_ << "m." << endl;

        // maximum size difference for matching
        this->maxMatchSizeRange_ = this->getParamHelper<double>("max_size_diff_range", 0.5);
        cout << this->hint_ << ": Max size difference range for matching is set to: " << this->maxMatchSizeRange_ << "m." << endl;

        // feature weight
        std::vector<double> tempWeights = this->getParamHelper<std::vector<double>>("feature_weight", std::vector<double>{3.0, 3.0, 0.1, 0.5, 0.5, 0.05, 0, 0, 0});
        this->featureWeights_ = Eigen::Map<Eigen::VectorXd>(tempWeights.data(), tempWeights.size());
        cout << this->hint_ << ": Feature weights size: " << tempWeights.size() << endl;

        // tracking history size
        this->histSize_ = this->getParamHelper<int>("history_size", 5);
        cout << this->hint_ << ": History for tracking is set to: " << this->histSize_ << endl;

        // history threshold for fixing box size
        this->fixSizeHistThresh_ = this->getParamHelper<int>("fix_size_history_threshold", 10);
        cout << this->hint_ << ": History threshold for fixing size is set to: " << this->fixSizeHistThresh_ << endl;

        // dimension threshold for fixing box size
        this->fixSizeDimThresh_ = this->getParamHelper<double>("fix_size_dimension_threshold", 0.4);
        cout << this->hint_ << ": Dimension threshold for fixing size is set to: " << this->fixSizeDimThresh_ << endl;

        // kalman filter parameters
        std::vector<double> kalmanFilterParams = this->getParamHelper<std::vector<double>>("kalman_filter_param", std::vector<double>{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});
        if (kalmanFilterParams.size() >= 7){
            this->eP_ = kalmanFilterParams[0];
            this->eQPos_ = kalmanFilterParams[1];
            this->eQVel_ = kalmanFilterParams[2];
            this->eQAcc_ = kalmanFilterParams[3];
            this->eRPos_ = kalmanFilterParams[4];
            this->eRVel_ = kalmanFilterParams[5];
            this->eRAcc_ = kalmanFilterParams[6];
        }
        else{
            this->eP_ = this->eQPos_ = this->eQVel_ = this->eQAcc_ = this->eRPos_ = this->eRVel_ = this->eRAcc_ = 0.5;
        }
        cout << this->hint_ << ": Kalman filter parameters loaded." << endl;

        // num of frames used in KF for observation
        this->kfAvgFrames_ = this->getParamHelper<int>("kalman_filter_averaging_frames", 10);
        cout << this->hint_ << ": Number of frames used in KF for observation is set to: " << this->kfAvgFrames_ << endl;

        // skip frame for classification
        this->skipFrame_ = this->getParamHelper<int>("frame_skip", 5);
        cout << this->hint_ << ": Frames skipped in classification is set to: " << this->skipFrame_ << endl;

        // velocity threshold for dynamic classification
        this->dynaVelThresh_ = this->getParamHelper<double>("dynamic_velocity_threshold", 0.35);
        cout << this->hint_ << ": Velocity threshold for dynamic classification is set to: " << this->dynaVelThresh_ << endl;

        // voting threshold for dynamic classification
        this->dynaVoteThresh_ = this->getParamHelper<double>("dynamic_voting_threshold", 0.8);
        cout << this->hint_ << ": Voting threshold for dynamic classification is set to: " << this->dynaVoteThresh_ << endl;

        // frames to force dynamic
        this->forceDynaFrames_ = this->getParamHelper<int>("frames_force_dynamic", 20);
        cout << this->hint_ << ": Range of searching dynamic obstacles in box history is set to: " << this->forceDynaFrames_ << endl;

        this->forceDynaCheckRange_ = this->getParamHelper<int>("frames_force_dynamic_check_range", 30);
        cout << this->hint_ << ": Threshold for forcing dynamic obstacles is set to: " << this->forceDynaCheckRange_ << endl;

        // dynamic consistency check
        this->dynamicConsistThresh_ = this->getParamHelper<int>("dynamic_consistency_threshold", 3);
        cout << this->hint_ << ": Threshold for dynamic consistency check is set to: " << this->dynamicConsistThresh_ << endl;

        if (this->histSize_ < this->forceDynaCheckRange_+1){
            RCLCPP_ERROR(this->get_logger(), "history length is too short to perform force-dynamic");
        }

        // constrain target object size
        this->constrainSize_ = this->getParamHelper<bool>("target_constrain_size", false);
        cout << this->hint_ << ": Target object constrain is set to: " << this->constrainSize_ << endl;

        // target object sizes
        std::vector<double> targetObjectSizeTemp = this->getParamHelper<std::vector<double>>("target_object_size", std::vector<double>{});
        for (size_t i=0; i+2<targetObjectSizeTemp.size(); i+=3){
            Eigen::Vector3d targetSize (targetObjectSizeTemp[i+0], targetObjectSizeTemp[i+1], targetObjectSizeTemp[i+2]);
            this->targetObjectSize_.push_back(targetSize);
            cout << this->hint_ << ": target object size is set to: [" << targetObjectSizeTemp[i+0] << ", "
                 << targetObjectSizeTemp[i+1] << ", " << targetObjectSizeTemp[i+2] << "]." << endl;
        }

        // max object size
        std::vector<double> maxObjectSizeTemp = this->getParamHelper<std::vector<double>>("max_object_size", std::vector<double>{2.0, 2.0, 2.0});
        if (maxObjectSizeTemp.size() >= 3){
            this->maxObjectSize_(0) = maxObjectSizeTemp[0];
            this->maxObjectSize_(1) = maxObjectSizeTemp[1];
            this->maxObjectSize_(2) = maxObjectSizeTemp[2];
        }
        else{
            this->maxObjectSize_ = Eigen::Vector3d (2.0, 2.0, 2.0);
        }
        cout << this->hint_ << ": Max object size threshold loaded." << endl;
    }

    void dynamicDetector::registerPub(){
        // uv detector depth map pub
        this->uvDepthMapPub_ = this->create_publisher<ImageMsg>(this->ns_ + "/detected_depth_map", 10);

        // uv detector u depth map pub
        this->uDepthMapPub_ = this->create_publisher<ImageMsg>(this->ns_ + "/detected_u_depth_map", 10);

        // uv detector bird view pub
        this->uvBirdViewPub_ = this->create_publisher<ImageMsg>(this->ns_ + "/u_depth_bird_view", 10);

        // color 2D bounding boxes pub
        this->detectedColorImgPub_ = this->create_publisher<ImageMsg>(this->ns_ + "/detected_color_image", 10);

        // uv detector bounding box pub
        this->uvBBoxesPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/uv_bboxes", 10);

        // DBSCAN bounding box pub
        this->dbBBoxesPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/dbscan_bboxes", 10);

        // visual bboxes pub
        this->visualBBoxesPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/visual_bboxes", 10);

        // lidar bbox pub
        this->lidarBBoxesPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/lidar_bboxes", 10);

        // filtered bounding box before YOLO pub
        this->filteredBBoxesBeforeYoloPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/filtered_before_yolo_bboxes", 10);

        // filtered bounding box pub
        this->filteredBBoxesPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/filtered_bboxes", 10);

        // tracked bounding box pub
        this->trackedBBoxesPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/tracked_bboxes", 10);

        // dynamic bounding box pub
        this->dynamicBBoxesPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/dynamic_bboxes", 10);

        // filtered depth pointcloud pub
        this->filteredDepthPointsPub_ = this->create_publisher<PointCloud2Msg>(this->ns_ + "/filtered_depth_cloud", 10);

        // lidar cluster pub
        this->lidarClustersPub_ = this->create_publisher<PointCloud2Msg>(this->ns_ + "/lidar_clusters", 10);

        // filtered pointcloud pub
        this->filteredPointsPub_ = this->create_publisher<PointCloud2Msg>(this->ns_ + "/filtered_point_cloud", 10);

        // dynamic pointcloud pub
        this->dynamicPointsPub_ = this->create_publisher<PointCloud2Msg>(this->ns_ + "/dynamic_point_cloud", 10);

        // raw dynamic pointcloud pub
        this->rawDynamicPointsPub_ = this->create_publisher<PointCloud2Msg>(this->ns_ + "/raw_dynamic_point_cloud", 10);

        // downsample points visualization pub
        this->downSamplePointsPub_ = this->create_publisher<PointCloud2Msg>(this->ns_ + "/downsampled_point_cloud", 10);

        // raw LiDAR points visualization pub
        this->rawLidarPointsPub_ = this->create_publisher<PointCloud2Msg>(this->ns_ + "/raw_lidar_point_cloud", 10);

        // history trajectory pub
        this->historyTrajPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/history_trajectories", 10);

        // velocity visualization pub
        this->velVisPub_ = this->create_publisher<MarkerArrayMsg>(this->ns_ + "/velocity_visualizaton", 10);
    }

    void dynamicDetector::registerCallback(){
        // depth and lidar message_filters subscribers
        this->depthSub_.reset(new message_filters::Subscriber<ImageMsg>(this, this->depthTopicName_));
        this->lidarCloudSub_.reset(new message_filters::Subscriber<PointCloud2Msg>(this, this->lidarTopicName_));
        if (this->localizationMode_ == 0){
            this->poseSub_.reset(new message_filters::Subscriber<PoseStampedMsg>(this, this->poseTopicName_));
            this->depthPoseSync_.reset(new message_filters::Synchronizer<depthPoseSync>(depthPoseSync(100), *this->depthSub_, *this->poseSub_));
            this->depthPoseSync_->registerCallback(std::bind(&dynamicDetector::depthPoseCB, this, std::placeholders::_1, std::placeholders::_2));
            this->lidarPoseSync_.reset(new message_filters::Synchronizer<lidarPoseSync>(lidarPoseSync(100), *this->lidarCloudSub_, *this->poseSub_));
            this->lidarPoseSync_->registerCallback(std::bind(&dynamicDetector::lidarPoseCB, this, std::placeholders::_1, std::placeholders::_2));
        }
        else if (this->localizationMode_ == 1){
            this->odomSub_.reset(new message_filters::Subscriber<OdometryMsg>(this, this->odomTopicName_));
            this->depthOdomSync_.reset(new message_filters::Synchronizer<depthOdomSync>(depthOdomSync(100), *this->depthSub_, *this->odomSub_));
            this->depthOdomSync_->registerCallback(std::bind(&dynamicDetector::depthOdomCB, this, std::placeholders::_1, std::placeholders::_2));
            this->lidarOdomSync_.reset(new message_filters::Synchronizer<lidarOdomSync>(lidarOdomSync(100), *this->lidarCloudSub_, *this->odomSub_));
            this->lidarOdomSync_->registerCallback(std::bind(&dynamicDetector::lidarOdomCB, this, std::placeholders::_1, std::placeholders::_2));
        }
        else{
            RCLCPP_ERROR(this->get_logger(), "[dynamicDetector]: Invalid localization mode!");
            exit(0);
        }

        // color image subscriber
        this->colorImgSub_ = this->create_subscription<ImageMsg>(
            this->colorImgTopicName_, 10, std::bind(&dynamicDetector::colorImgCB, this, std::placeholders::_1));

        // yolo detection results subscriber
        this->yoloDetectionSub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            "yolo_detector/detected_bounding_boxes", 10, std::bind(&dynamicDetector::yoloDetectionCB, this, std::placeholders::_1));

        // timers
        auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(this->dt_));
        this->detectionTimer_ = this->create_wall_timer(period, std::bind(&dynamicDetector::detectionCB, this));
        this->lidarDetectionTimer_ = this->create_wall_timer(period, std::bind(&dynamicDetector::lidarDetectionCB, this));
        this->trackingTimer_ = this->create_wall_timer(period, std::bind(&dynamicDetector::trackingCB, this));
        this->classificationTimer_ = this->create_wall_timer(period, std::bind(&dynamicDetector::classificationCB, this));
        this->visTimer_ = this->create_wall_timer(period, std::bind(&dynamicDetector::visCB, this));

		// get dynamic obstacle service (lambda to disambiguate the overloaded name)
		this->getDynamicObstacleServer_ = this->create_service<onboard_detector::srv::GetDynamicObstacles>(
            "onboard_detector/get_dynamic_obstacles",
            [this](const std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Request> req,
                   std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Response> res){
                this->getDynamicObstacles(req, res);
            });
    }

    void dynamicDetector::getDynamicObstacles(const std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Request> req,
                                              std::shared_ptr<onboard_detector::srv::GetDynamicObstacles::Response> res) {
        // Get the current robot position
        Eigen::Vector3d currPos = Eigen::Vector3d (req->current_position.x, req->current_position.y, req->current_position.z);

        // Vector to store obstacles along with their distances
        std::vector<std::pair<double, onboardDetector::box3D>> obstaclesWithDistances;

        // Go through all obstacles and calculate distances
        for (const onboardDetector::box3D& bbox : this->dynamicBBoxes_) {
            Eigen::Vector3d obsPos(bbox.x, bbox.y, bbox.z);
            Eigen::Vector3d diff = currPos - obsPos;
            diff(2) = 0.;
            double distance = diff.norm();
            if (distance <= req->range) {
                obstaclesWithDistances.push_back(std::make_pair(distance, bbox));
            }
        }

        // Sort obstacles by distance in ascending order
        std::sort(obstaclesWithDistances.begin(), obstaclesWithDistances.end(), 
                [](const std::pair<double, onboardDetector::box3D>& a, const std::pair<double, onboardDetector::box3D>& b) {
                    return a.first < b.first;
                });

        // Push sorted obstacles into the response
        for (const auto& item : obstaclesWithDistances) {
            const onboardDetector::box3D& bbox = item.second;

            geometry_msgs::msg::Vector3 pos;
            geometry_msgs::msg::Vector3 vel;
            geometry_msgs::msg::Vector3 size;

            pos.x = bbox.x;
            pos.y = bbox.y;
            pos.z = bbox.z;

            vel.x = bbox.Vx;
            vel.y = bbox.Vy;
            vel.z = 0.;

            size.x = bbox.x_width;
            size.y = bbox.y_width;
            size.z = bbox.z_width;

            res->position.push_back(pos);
            res->velocity.push_back(vel);
            res->size.push_back(size);
        }

        return;
    }

    void dynamicDetector::depthPoseCB(const ImageMsg::ConstSharedPtr& img, const PoseStampedMsg::ConstSharedPtr& pose){
        // store current depth image
        cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
        if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1){
            (imgPtr->image).convertTo(imgPtr->image, CV_16UC1, this->depthScale_);
        }
        imgPtr->image.copyTo(this->depthImage_);

        // store current position and orientation (camera)
        Eigen::Matrix4d camPoseDepthMatrix, camPoseColorMatrix;
        this->getCameraPose(pose, camPoseDepthMatrix, camPoseColorMatrix);

        this->position_(0) = pose->pose.position.x;
        this->position_(1) = pose->pose.position.y;
        this->position_(2) = pose->pose.position.z;
        Eigen::Quaterniond quat;
        quat = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        this->orientation_ = rot;

        this->positionDepth_(0) = camPoseDepthMatrix(0, 3);
        this->positionDepth_(1) = camPoseDepthMatrix(1, 3);
        this->positionDepth_(2) = camPoseDepthMatrix(2, 3);
        this->orientationDepth_ = camPoseDepthMatrix.block<3, 3>(0, 0);

        this->positionColor_(0) = camPoseColorMatrix(0, 3);
        this->positionColor_(1) = camPoseColorMatrix(1, 3);
        this->positionColor_(2) = camPoseColorMatrix(2, 3);
        this->orientationColor_ = camPoseColorMatrix.block<3, 3>(0, 0);
    }

    void dynamicDetector::depthOdomCB(const ImageMsg::ConstSharedPtr& img, const OdometryMsg::ConstSharedPtr& odom){
        // store current depth image
        cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
        if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1){
            (imgPtr->image).convertTo(imgPtr->image, CV_16UC1, this->depthScale_);
        }
        imgPtr->image.copyTo(this->depthImage_);

        // store current position and orientation (camera)
        Eigen::Matrix4d camPoseDepthMatrix, camPoseColorMatrix;
        this->getCameraPose(odom, camPoseDepthMatrix, camPoseColorMatrix);

        this->position_(0) = odom->pose.pose.position.x;
        this->position_(1) = odom->pose.pose.position.y;
        this->position_(2) = odom->pose.pose.position.z;
        Eigen::Quaterniond quat;
        quat = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        this->orientation_ = rot;

        this->positionDepth_(0) = camPoseDepthMatrix(0, 3);
        this->positionDepth_(1) = camPoseDepthMatrix(1, 3);
        this->positionDepth_(2) = camPoseDepthMatrix(2, 3);
        this->orientationDepth_ = camPoseDepthMatrix.block<3, 3>(0, 0);

        this->positionColor_(0) = camPoseColorMatrix(0, 3);
        this->positionColor_(1) = camPoseColorMatrix(1, 3);
        this->positionColor_(2) = camPoseColorMatrix(2, 3);
        this->orientationColor_ = camPoseColorMatrix.block<3, 3>(0, 0);
    }

    void dynamicDetector::lidarPoseCB(const PointCloud2Msg::ConstSharedPtr& cloudMsg, const PoseStampedMsg::ConstSharedPtr& pose){
        this->lastSyncTime_ = std::chrono::steady_clock::now();
        // for visualization
        this->latestCloud_ = cloudMsg;

        // local cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloudMsg, *tempCloud);

        // filter and downsample pointcloud
        // Create a filtered cloud pointer to store intermediate results
        pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud (new pcl::PointCloud<pcl::PointXYZ>());

        // Apply a pass-through filter to limit points to the local sensor range in X, Y, and Z axes
        pcl::PassThrough<pcl::PointXYZ> pass;

        // Filter for X axis
        pass.setInputCloud(tempCloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-this->localLidarRange_.x(), this->localLidarRange_.x());
        pass.filter(*filteredCloud);

        // Filter for Y axis
        pass.setInputCloud(filteredCloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-this->localLidarRange_.y(), this->localLidarRange_.y());
        pass.filter(*filteredCloud);

        int sigma = this->gaussianDownSampleRate_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr preTransformCloud(new pcl::PointCloud<pcl::PointXYZ>());
        preTransformCloud->reserve(filteredCloud->size());

        for (pcl::PointXYZ &pt : filteredCloud->points) {
            double dist = pow(pow(pt.x, 2) + pow(pt.y, 2), 0.5);
            if (dist < this->lidarMinRange_) continue; // 車身/盲區點（VLP-16 近場 + 車自身結構）
            double p = std::exp(-(dist * dist) / (2 * sigma * sigma));

            double r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (r < p) {
                preTransformCloud->push_back(pt);
            }
        }

        // transform
        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.linear() = this->orientationLidar_;
        transform.translation() = this->positionLidar_;

        // map cloud
        // Create an empty point cloud to store the transformed data
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud (new pcl::PointCloud<pcl::PointXYZ>());

        // Apply the transformation
        pcl::transformPointCloud(*preTransformCloud, *transformedCloud, transform);

        // filter roof and ground 
        pcl::PointCloud<pcl::PointXYZ>::Ptr groundRoofFilterCloud (new pcl::PointCloud<pcl::PointXYZ>());
        pass.setInputCloud(transformedCloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(this->groundHeight_, this->roofHeight_);
        pass.filter(*groundRoofFilterCloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledCloud = groundRoofFilterCloud;
        // Create the VoxelGrid filter object
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        // sor.setInputCloud(filteredCloud);
        sor.setInputCloud(groundRoofFilterCloud);

        // Set the leaf size (adjust to control the downsampling)
        sor.setLeafSize(0.1f, 0.1f, 0.1f); // Try different values based on your point cloud density

        // If the downsampled cloud has more than certain points, further increase the leaf size
        while (int(downsampledCloud->size()) > this->downSampleThresh_) {
            double leafSize = sor.getLeafSize().x() * 1.1f; // Increase the leaf size to reduce point count
            sor.setLeafSize(leafSize, leafSize, leafSize);
            sor.filter(*downsampledCloud);
        }

        this->lidarCloud_ = downsampledCloud;
        sensor_msgs::msg::PointCloud2 outputCloud;
        pcl::toROSMsg(*this->lidarCloud_, outputCloud); // Convert to ROS message
        outputCloud.header.frame_id = this->visFrame_;
        this->downSamplePointsPub_->publish(outputCloud);

        // store current position and orientation
        Eigen::Matrix4d lidarPoseMatrix;
        this->getLidarPose(pose, lidarPoseMatrix);

        this->position_(0) = pose->pose.position.x;
        this->position_(1) = pose->pose.position.y;
        this->position_(2) = pose->pose.position.z;
        Eigen::Quaterniond quat;
        quat = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        this->orientation_ = rot;

        this->positionLidar_(0) = lidarPoseMatrix(0, 3);
        this->positionLidar_(1) = lidarPoseMatrix(1, 3);
        this->positionLidar_(2) = lidarPoseMatrix(2, 3);
        this->orientationLidar_ = lidarPoseMatrix.block<3, 3>(0, 0);
    }

    void dynamicDetector::lidarOdomCB(const PointCloud2Msg::ConstSharedPtr& cloudMsg, const OdometryMsg::ConstSharedPtr& odom){
        this->lastSyncTime_ = std::chrono::steady_clock::now();
        // for visualization
        this->latestCloud_ = cloudMsg;

        // local cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloudMsg, *tempCloud);

        // filter and downsample pointcloud
        // Create a filtered cloud pointer to store intermediate results
        pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud (new pcl::PointCloud<pcl::PointXYZ>());

        // Apply a pass-through filter to limit points to the local sensor range in X, Y, and Z axes
        pcl::PassThrough<pcl::PointXYZ> pass;

        // Filter for X axis
        pass.setInputCloud(tempCloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-this->localLidarRange_.x(), this->localLidarRange_.x());
        pass.filter(*filteredCloud);

        // Filter for Y axis
        pass.setInputCloud(filteredCloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-this->localLidarRange_.y(), this->localLidarRange_.y());
        pass.filter(*filteredCloud);

        int sigma = this->gaussianDownSampleRate_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr preTransformCloud(new pcl::PointCloud<pcl::PointXYZ>());
        preTransformCloud->reserve(filteredCloud->size());

        for (pcl::PointXYZ &pt : filteredCloud->points) {
            double dist = pow(pow(pt.x, 2) + pow(pt.y, 2), 0.5);
            if (dist < this->lidarMinRange_) continue; // 車身/盲區點（VLP-16 近場 + 車自身結構）
            double p = std::exp(-(dist * dist) / (2 * sigma * sigma));

            double r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (r < p) {
                preTransformCloud->push_back(pt);
            }
        }

        // transform
        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.linear() = this->orientationLidar_;
        transform.translation() = this->positionLidar_;

        // map cloud
        // Create an empty point cloud to store the transformed data
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud (new pcl::PointCloud<pcl::PointXYZ>());

        // Apply the transformation
        pcl::transformPointCloud(*preTransformCloud, *transformedCloud, transform);

        // filter roof and ground 
        pcl::PointCloud<pcl::PointXYZ>::Ptr groundRoofFilterCloud (new pcl::PointCloud<pcl::PointXYZ>());
        pass.setInputCloud(transformedCloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(this->groundHeight_, this->roofHeight_);
        pass.filter(*groundRoofFilterCloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledCloud = groundRoofFilterCloud;
        // Create the VoxelGrid filter object
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        // sor.setInputCloud(filteredCloud);
        sor.setInputCloud(groundRoofFilterCloud);

        // Set the leaf size (adjust to control the downsampling)
        sor.setLeafSize(0.1f, 0.1f, 0.1f); // Try different values based on your point cloud density

        // If the downsampled cloud has more than certain points, further increase the leaf size
        while (int(downsampledCloud->size()) > this->downSampleThresh_) {
            double leafSize = sor.getLeafSize().x() * 1.1f; // Increase the leaf size to reduce point count
            sor.setLeafSize(leafSize, leafSize, leafSize);
            sor.filter(*downsampledCloud);
        }

        this->lidarCloud_ = downsampledCloud;
        sensor_msgs::msg::PointCloud2 outputCloud;
        pcl::toROSMsg(*this->lidarCloud_, outputCloud); // Convert to ROS message
        outputCloud.header.frame_id = this->visFrame_;
        this->downSamplePointsPub_->publish(outputCloud);
        
        // store current position and orientation
        Eigen::Matrix4d lidarPoseMatrix;
        this->getLidarPose(odom, lidarPoseMatrix);

        this->position_(0) = odom->pose.pose.position.x;
        this->position_(1) = odom->pose.pose.position.y;
        this->position_(2) = odom->pose.pose.position.z;
        Eigen::Quaterniond quat;
        quat = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        this->orientation_ = rot;

        this->positionLidar_(0) = lidarPoseMatrix(0, 3);
        this->positionLidar_(1) = lidarPoseMatrix(1, 3);
        this->positionLidar_(2) = lidarPoseMatrix(2, 3);
        this->orientationLidar_ = lidarPoseMatrix.block<3, 3>(0, 0);
    }

    void dynamicDetector::colorImgCB(const ImageMsg::ConstSharedPtr& img){
        cv_bridge::CvImagePtr imgPtr = cv_bridge::toCvCopy(img, img->encoding);
        imgPtr->image.copyTo(this->detectedColorImage_);
    }

    void dynamicDetector::yoloDetectionCB(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& detections){
        this->yoloDetectionResults_ = *detections;
    }

   
    void dynamicDetector::lidarDetectionCB(){
        this->lidarDetect();
    }

    void dynamicDetector::detectionCB(){
        // detection thread
        this->dbscanDetect();
        this->uvDetect();
        // ros::Time start = this->now();
        this->filterLVBBoxes();
        // ros::Time end = this->now();
        // ROS_INFO("filtering time: %f", (end - start).toSec());
        this->newDetectFlag_ = true; // get a new detection
    }

    void dynamicDetector::trackingCB(){
        // data association thread
        std::vector<int> bestMatch; // for each current detection, which index of previous obstacle match
        this->boxAssociation(bestMatch);
        // kalman filter tracking
        if (bestMatch.size()){
            this->kalmanFilterAndUpdateHist(bestMatch);
        }
        else {
            this->boxHist_.clear();
            this->pcHist_.clear();
            this->pcCenterHist_.clear();
        }
    }

    void dynamicDetector::classificationCB(){
        // Identification thread
        std::vector<onboardDetector::box3D> dynamicBBoxesTemp;

        // Iterate through all pointcloud/bounding boxes history (note that yolo's pointclouds are dummy pointcloud (empty))
        // NOTE: There are 3 cases which we don't need to perform dynamic obstacle identification.
        for (size_t i=0; i<this->pcHist_.size() ; ++i){
            // ===================================================================================
            // CASE I: yolo recognized as dynamic dynamic obstacle
            if (this->boxHist_[i][0].is_human){
                dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);
                continue;
            }
            // ===================================================================================


            // ===================================================================================
            // CASE II: history length is not enough to run classification
            int curFrameGap;
            if (int(this->pcHist_[i].size()) < this->skipFrame_+1){
                curFrameGap = this->pcHist_[i].size() - 1;
            }
            else{
                curFrameGap = this->skipFrame_;
            }
            // ===================================================================================


            // ==================================================================================
            // CASE III: Force Dynamic (if the obstacle is classifed as dynamic for several time steps)
            int dynaFrames = 0;
            if (int(this->boxHist_[i].size()) > this->forceDynaCheckRange_){
                for (int j=1 ; j<this->forceDynaCheckRange_+1 ; ++j){
                    if (this->boxHist_[i][j].is_dynamic){
                        ++dynaFrames;
                    }
                }
            }

            if (dynaFrames >= this->forceDynaFrames_){
                this->boxHist_[i][0].is_dynamic = true;
                dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);
                continue;
            }
            // ===================================================================================

            std::vector<Eigen::Vector3d> currPc = this->pcHist_[i][0];
            std::vector<Eigen::Vector3d> prevPc = this->pcHist_[i][curFrameGap];
            Eigen::Vector3d Vcur(0.,0.,0.); // single point velocity 
            Eigen::Vector3d Vbox(0.,0.,0.); // bounding box velocity 
            Eigen::Vector3d Vkf(0.,0.,0.);  // velocity estimated from kalman filter
            int numPoints = currPc.size(); // it changes within loop
            int votes = 0;

            Vbox(0) = (this->boxHist_[i][0].x - this->boxHist_[i][curFrameGap].x)/(this->dt_*curFrameGap);
            Vbox(1) = (this->boxHist_[i][0].y - this->boxHist_[i][curFrameGap].y)/(this->dt_*curFrameGap);
            Vbox(2) = (this->boxHist_[i][0].z - this->boxHist_[i][curFrameGap].z)/(this->dt_*curFrameGap);
            Vkf(0) = this->boxHist_[i][0].Vx;
            Vkf(1) = this->boxHist_[i][0].Vy;

            // find nearest neighbor
            for (size_t j=0 ; j<currPc.size() ; ++j){
                double minDist = 2;
                Eigen::Vector3d nearestVect;
                for (size_t k=0 ; k<prevPc.size() ; k++){ // find the nearest point in the previous pointcloud
                    double dist = (currPc[j]-prevPc[k]).norm();
                    if (abs(dist) < minDist){
                        minDist = dist;
                        nearestVect = currPc[j]-prevPc[k];
                    }
                }
                Vcur = nearestVect/(this->dt_*curFrameGap); Vcur(2) = 0;
                double velSim = Vcur.dot(Vbox)/(Vcur.norm()*Vbox.norm());

                if (velSim < 0){
                    --numPoints;
                }
                else{
                    if (Vcur.norm()>this->dynaVelThresh_){
                        ++votes;
                    }
                }
            }
            
            
            // update dynamic boxes
            double voteRatio = (numPoints>0)?double(votes)/double(numPoints):0;
            double velNorm = Vkf.norm();

            // voting and velocity threshold
            // 1. point cloud voting ratio.
            // 2. velocity (from kalman filter) 
            // 淨位移檢查：人走過時被擾動的靜態簇（遮擋→露出）有速度抖動但沒有淨位移，
            // 會在人離開後冒出「尾流」假動態框；真動態物在窗內有實際位移
            bool hasNetDisplacement = true;
            if (this->dynamicMinDisp_ > 0){
                // A：淨位移門檻隨「可用歷史」線性縮放，不再硬等滿窗 → 短 track（VLP-16 稀疏點下
                // 撐不到滿窗的常態）也能判動態。抖動在任何窗長都~0 淨位移，仍被擋。
                // 滿窗(dispWindow 幀)要求 dynamicMinDisp_；可用 W 幀則要求 dynamicMinDisp_·W/dispWindow。
                int dispWindow = (this->dynamicDispWindow_ > 0) ? this->dynamicDispWindow_ : 2*this->dynamicConsistThresh_;
                int availWindow = std::min((int)this->boxHist_[i].size(), dispWindow);
                if (availWindow >= this->dynamicMinDispFrames_){
                    double netDx = this->boxHist_[i][0].x - this->boxHist_[i][availWindow-1].x;
                    double netDy = this->boxHist_[i][0].y - this->boxHist_[i][availWindow-1].y;
                    double required = this->dynamicMinDisp_ * (double)availWindow / (double)dispWindow;
                    hasNetDisplacement = std::sqrt(netDx*netDx + netDy*netDy) >= required;
                }
                else{
                    hasNetDisplacement = false; // track 還太短(< min_disp_frames)，位移無從證明
                }
            }

            if (voteRatio>=this->dynaVoteThresh_ && velNorm>=this->dynaVelThresh_ && hasNetDisplacement){
                this->boxHist_[i][0].is_dynamic_candidate = true;
                // dynamic-consistency check
                int dynaConsistCount = 0;
                if (int(this->boxHist_[i].size()) >= this->dynamicConsistThresh_){
                    for (int j=0 ; j<this->dynamicConsistThresh_; ++j){
                        if (this->boxHist_[i][j].is_dynamic_candidate or this->boxHist_[i][j].is_human or this->boxHist_[i][j].is_dynamic){
                            ++dynaConsistCount;
                        }
                    }
                }            
                // 原為全有或全無（== threshold，斷 1 幀歸零）：VLP-16 稀疏點下行人 track
                // 偶有斷幀，永遠湊不滿 → LiDAR-only 抓不到動態。改 80% 容忍 2/10 幀斷線
                if (dynaConsistCount >= std::max(1, (int)std::ceil(0.8 * this->dynamicConsistThresh_))){
                    // set as dynamic and push into history
                    this->boxHist_[i][0].is_dynamic = true;
                    dynamicBBoxesTemp.push_back(this->boxHist_[i][0]);    
                }
            }
        }

        // filter the dynamic obstacles based on the target sizes
        if (this->constrainSize_){
            std::vector<onboardDetector::box3D> dynamicBBoxesBeforeConstrain = dynamicBBoxesTemp;
            dynamicBBoxesTemp.clear();

            for (onboardDetector::box3D ob : dynamicBBoxesBeforeConstrain){
                // YOLO 已確認是人(is_human)→ 直接放行，不再用尺寸濾。
                // VLP-16 稀疏下人物框 z_width 常 <0.5(只掃到上半身)，套人形尺寸約束會誤殺已確認的人
                // (實測前方 P(動態|YOLO人) 從 4% 掉，關此約束則 78%)。尺寸約束只該濾 LiDAR-only 候選的家具。
                if (ob.is_human){
                    dynamicBBoxesTemp.push_back(ob);
                    continue;
                }
                bool findMatch = false;
                for (Eigen::Vector3d targetSize : this->targetObjectSize_){
                    double xdiff = std::abs(ob.x_width - targetSize(0));
                    double ydiff = std::abs(ob.y_width - targetSize(1));
                    double zdiff = std::abs(ob.z_width - targetSize(2)); 
                    if (xdiff < 0.8 and ydiff < 0.8 and zdiff < 1.0){
                        findMatch = true;
                    }
                }

                if (findMatch){
                    dynamicBBoxesTemp.push_back(ob);
                }
            }
        }

        this->dynamicBBoxes_ = dynamicBBoxesTemp;
    }

    void dynamicDetector::visCB(){
        // odom/LiDAR 同步斷流時偵測管線凍結，但 timer 照發過期內容 → 下游看到「停在
        // 幾秒前位置的動態框」。逾時就清空動態輸出 + 警告，誠實表達「目前沒有有效偵測」
        double syncAge = std::chrono::duration<double>(std::chrono::steady_clock::now() - this->lastSyncTime_).count();
        if (syncAge > this->staleTimeout_){
            this->dynamicBBoxes_.clear();
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[onboardDetector]: LiDAR/odom 同步中斷 %.1fs（odom 斷流？）— 動態偵測已凍結，清空輸出", syncAge);
        }

        this->publishUVImages();
        this->publishColorImages();

        this->publish3dBox(this->uvBBoxes_, this->uvBBoxesPub_, 0, 1, 0);
        this->publish3dBox(this->dbBBoxes_, this->dbBBoxesPub_, 1, 0, 0);
        this->publish3dBox(this->visualBBoxes_, this->visualBBoxesPub_, 0.3, 0.8, 1.0);
        this->publish3dBox(this->lidarBBoxes_, this->lidarBBoxesPub_, 0.5, 0.5, 0.5); // raw lidar cluster bounding boxes
        this->publish3dBox(this->filteredBBoxesBeforeYolo_, this->filteredBBoxesBeforeYoloPub_, 0, 1, 0.5);
        this->publish3dBox(this->filteredBBoxes_, this->filteredBBoxesPub_, 0, 1, 1);
        this->publish3dBox(this->trackedBBoxes_, this->trackedBBoxesPub_, 1, 1, 0);
        this->publish3dBox(this->dynamicBBoxes_, this->dynamicBBoxesPub_, 0, 0, 1);

        this->publishLidarClusters(); // colored clusters
        this->publishFilteredPoints();
        std::vector<Eigen::Vector3d> dynamicPoints;
        this->getDynamicPc(dynamicPoints);
        this->publishPoints(dynamicPoints, this->dynamicPointsPub_);
        this->publishPoints(this->filteredDepthPoints_, this->filteredDepthPointsPub_);
        this->publishRawDynamicPoints();

        this->publishHistoryTraj();
        this->publishVelVis();

        // 終端輸出偵測到的動態障礙詳細資訊（~1Hz；loop 20Hz → 每 20 幀印一次）
        static int dynLogCounter = 0;
        if (++dynLogCounter >= 20){
            dynLogCounter = 0;
            if (this->dynamicBBoxes_.empty()){
                RCLCPP_INFO(this->get_logger(), "[動態障礙] 0 個");
            }
            else{
                std::ostringstream oss;
                oss << "[動態障礙] " << this->dynamicBBoxes_.size() << " 個:";
                for (size_t i = 0; i < this->dynamicBBoxes_.size(); ++i){
                    const onboardDetector::box3D& b = this->dynamicBBoxes_[i];
                    double spd = std::sqrt(b.Vx*b.Vx + b.Vy*b.Vy);
                    oss << std::fixed << std::setprecision(2)
                        << "  #" << i
                        << " pos(" << b.x << "," << b.y << ")"
                        << " v(" << b.Vx << "," << b.Vy << ")|" << spd << "m/s"
                        << " 尺寸(" << b.x_width << "x" << b.y_width << "x" << b.z_width << ")"
                        << (b.is_human ? " [YOLO人]" : " [LiDAR]");  // 來源：YOLO(相機)確認 vs LiDAR 速度分類
                }
                RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
            }
        }
    }

    void dynamicDetector::uvDetect(){
        // initialization
        if (this->uvDetector_ == NULL){
            this->uvDetector_.reset(new UVdetector ());
            this->uvDetector_->fx = this->fx_;
            this->uvDetector_->fy = this->fy_;
            this->uvDetector_->px = this->cx_;
            this->uvDetector_->py = this->cy_;
            this->uvDetector_->depthScale_ = this->depthScale_; 
            this->uvDetector_->max_dist = this->raycastMaxLength_ * 1000;
        }

        // detect from depth mapcalBox
        if (not this->depthImage_.empty()){
            this->uvDetector_->depth = this->depthImage_;
            this->uvDetector_->detect();
            this->uvDetector_->extract_3Dbox();

            this->uvDetector_->display_U_map();
            this->uvDetector_->display_bird_view();
            this->uvDetector_->display_depth();

            // transform to the world frame (recalculate the boudning boxes)
            std::vector<onboardDetector::box3D> uvBBoxes;
            this->transformUVBBoxes(uvBBoxes);
            this->uvBBoxes_ = uvBBoxes;
        }
    }

    void dynamicDetector::dbscanDetect(){
        // 1. get pointcloud
        this->projectDepthImage();

        // 2. filter points
        this->filterPoints(this->projPoints_, this->filteredDepthPoints_);

        // 3. cluster points and get bounding boxes
        this->clusterPointsAndBBoxes(this->filteredDepthPoints_, this->dbBBoxes_, this->pcClustersVisual_, 
                                     this->pcClusterCentersVisual_, this->pcClusterStdsVisual_);
    }


    void dynamicDetector::lidarDetect(){
        if (this->lidarDetector_ == NULL){
            this->lidarDetector_.reset(new lidarDetector());
            this->lidarDetector_->setParams(this->lidarDBEpsilon_, this->lidarDBMinPoints_);
        }

        if (this->lidarCloud_ != NULL){
            this->lidarDetector_->getPointcloud(this->lidarCloud_);
            this->lidarDetector_->lidarDBSCAN();

            std::vector<onboardDetector::Cluster> lidarClustersRaw = this->lidarDetector_->getClusters();
            std::vector<onboardDetector::Cluster> lidarClustersFiltered;
            std::vector<onboardDetector::box3D> lidarBBoxesRaw = this->lidarDetector_->getBBoxes();
            std::vector<onboardDetector::box3D> lidarBBoxesFiltered;
            for (int i=0; i<int(lidarBBoxesRaw.size()); ++i){
                onboardDetector::box3D lidarBBox = lidarBBoxesRaw[i];
                // filter out lidar bounding boxes that are too large
                if(lidarBBox.x_width > this->maxObjectSize_(0) || lidarBBox.y_width > this->maxObjectSize_(1) || lidarBBox.z_width > this->maxObjectSize_(2)){
                    continue;
                }
                lidarBBoxesFiltered.push_back(lidarBBox);
                lidarClustersFiltered.push_back(lidarClustersRaw[i]);            
            }
            this->lidarBBoxes_ = lidarBBoxesFiltered;
            this->lidarClusters_ = lidarClustersFiltered;
        }
    }

    void dynamicDetector::filterLVBBoxes(){
        std::vector<onboardDetector::box3D> filteredBBoxesTemp;
        std::vector<std::vector<Eigen::Vector3d>> filteredPcClustersTemp;
        std::vector<Eigen::Vector3d> filteredPcClusterCentersTemp;
        std::vector<Eigen::Vector3d> filteredPcClusterStdsTemp; 

        std::vector<onboardDetector::box3D> visualBBoxesTemp;
        std::vector<std::vector<Eigen::Vector3d>> visualPcClustersTemp;
        std::vector<Eigen::Vector3d> visualPcClusterCentersTemp;
        std::vector<Eigen::Vector3d> visualPcClusterStdsTemp; // store visual output

        std::vector<onboardDetector::box3D> lidarBBoxesTemp;
        std::vector<std::vector<Eigen::Vector3d>> lidarPcClustersTemp;
        std::vector<Eigen::Vector3d> lidarPcClusterCentersTemp;
        std::vector<Eigen::Vector3d> lidarPcClusterStdsTemp; // store lidar output

        // STEP 1: Get visual bboxes by fusing visual bounding boxes
        // find best IOU match for both uv and dbscan. If they are best for each other, then add to filtered bbox and fuse.
        for (size_t i=0 ; i<this->uvBBoxes_.size(); ++i){
            onboardDetector::box3D uvBBox = this->uvBBoxes_[i];
            double bestIOUForUVBBox, bestIOUForDBBBox;
            int bestMatchForUVBBox = this->getBestOverlapBBox(uvBBox, this->dbBBoxes_, bestIOUForUVBBox);
            if (bestMatchForUVBBox == -1) continue; // no match at all
            onboardDetector::box3D matchedDBBBox = this->dbBBoxes_[bestMatchForUVBBox]; 
            std::vector<Eigen::Vector3d> matchedPcCluster = this->pcClustersVisual_[bestMatchForUVBBox];
            Eigen::Vector3d matchedPcClusterCenter = this->pcClusterCentersVisual_[bestMatchForUVBBox];
            Eigen::Vector3d matchedPcClusterStd = this->pcClusterStdsVisual_[bestMatchForUVBBox];
            int bestMatchForDBBBox = this->getBestOverlapBBox(matchedDBBBox, this->uvBBoxes_, bestIOUForDBBBox);

            // if best match is each other and both the IOU is greater than the threshold
            if (bestMatchForDBBBox == int(i) and bestIOUForUVBBox > this->boxIOUThresh_ and bestIOUForDBBBox > this->boxIOUThresh_){
                onboardDetector::box3D bbox;
                
                // take concervative strategy
                double xmax = std::max(uvBBox.x+uvBBox.x_width/2, matchedDBBBox.x+matchedDBBBox.x_width/2);
                double xmin = std::min(uvBBox.x-uvBBox.x_width/2, matchedDBBBox.x-matchedDBBBox.x_width/2);
                double ymax = std::max(uvBBox.y+uvBBox.y_width/2, matchedDBBBox.y+matchedDBBBox.y_width/2);
                double ymin = std::min(uvBBox.y-uvBBox.y_width/2, matchedDBBBox.y-matchedDBBBox.y_width/2);
                double zmax = std::max(uvBBox.z+uvBBox.z_width/2, matchedDBBBox.z+matchedDBBBox.z_width/2);
                double zmin = std::min(uvBBox.z-uvBBox.z_width/2, matchedDBBBox.z-matchedDBBBox.z_width/2);
                bbox.x = (xmin+xmax)/2;
                bbox.y = (ymin+ymax)/2;
                bbox.z = (zmin+zmax)/2;
                bbox.x_width = xmax-xmin;
                bbox.y_width = ymax-ymin;
                bbox.z_width = zmax-zmin;
                bbox.Vx = 0;
                bbox.Vy = 0;

                visualBBoxesTemp.push_back(bbox);
                visualPcClustersTemp.push_back(matchedPcCluster);
                visualPcClusterCentersTemp.push_back(matchedPcClusterCenter);
                visualPcClusterStdsTemp.push_back(matchedPcClusterStd);
            }
        }
        this->visualBBoxes_ = visualBBoxesTemp; // for visualization

        // STEP 2: Get lidar bboxes and its corresponding clusters and features
        // lidar bbox filter
        for (size_t i = 0; i < this->lidarBBoxes_.size(); ++i) {
            onboardDetector::box3D lidarBBox = this->lidarBBoxes_[i];
            
            // get corresponding point cloud cluster
            onboardDetector::Cluster cluster = this->lidarClusters_[i];

            std::vector<Eigen::Vector3d> pcCluster;
            for (const pcl::PointXYZ& point : cluster.points->points) {
                pcCluster.emplace_back(point.x, point.y, point.z);
            }

            // extract the cluster center
            Eigen::Vector3d clusterCenter(cluster.centroid[0], cluster.centroid[1], cluster.centroid[2]);

            // compute std
            Eigen::Vector3d clusterStd = cluster.eigen_values.cwiseSqrt().cast<double>();

            lidarBBoxesTemp.push_back(lidarBBox);
            lidarPcClustersTemp.push_back(pcCluster);
            lidarPcClusterCentersTemp.push_back(clusterCenter);
            lidarPcClusterStdsTemp.push_back(clusterStd);
        }

        // STEP 3: Fuse LiDAR and visual bounding boxes
        // init processed flags
        std::vector<bool> processedLidarBBoxes(lidarBBoxesTemp.size(), false);
        std::vector<bool> processedVisualBBoxes(visualBBoxesTemp.size(), false);
        for (size_t i = 0; i < visualBBoxesTemp.size(); ++i) {
            if (processedVisualBBoxes[i]) continue; // skip processed visual boxes
            onboardDetector::box3D visualBBox = visualBBoxesTemp[i];
            std::vector<int> overlappingLidarBBoxes;
            std::vector<int> overlappingVisualBBoxes;

            // loop through all LiDAR boxes
            for (size_t j = 0; j < lidarBBoxesTemp.size(); ++j) {
                if (processedLidarBBoxes[j]) continue; // skip processed LiDAR boxes
                onboardDetector::box3D lidarBBox = lidarBBoxesTemp[j];

                // find IOU between the visual bboxes and lidar bboxes
                double lvIOU = this->calBoxIOU(visualBBox, lidarBBox, true);
                if (lvIOU > this->boxIOUThresh_){
                    overlappingLidarBBoxes.push_back(j);
                    for (size_t k=0; k<visualBBoxesTemp.size(); ++k){
                        if (processedVisualBBoxes[i] or i==k) continue;
                        onboardDetector::box3D visualBBoxPotentialMatch = visualBBoxesTemp[k];
                        double lvIOUPotentialMatch = this->calBoxIOU(visualBBoxPotentialMatch, lidarBBox, true);
                        if (lvIOUPotentialMatch > this->boxIOUThresh_){
                            overlappingVisualBBoxes.push_back(k);
                        }
                    }
                }
            }

            // **Case 1: no overlapping LiDAR boxes
            if (overlappingLidarBBoxes.empty()) {
                // no overlapping LiDAR boxes, keep the visual box
                filteredBBoxesTemp.push_back(visualBBox);
                filteredPcClustersTemp.push_back(visualPcClustersTemp[i]);
                filteredPcClusterCentersTemp.push_back(visualPcClusterCentersTemp[i]);
                filteredPcClusterStdsTemp.push_back(visualPcClusterStdsTemp[i]);
                processedVisualBBoxes[i] = true; // mark the visual box as processed
            // **Case 2: multiple bounding boxes have overlapping
            }else{
                std::vector<onboardDetector::box3D> overlappingBBoxes {visualBBox};
                std::vector<Eigen::Vector3d> overlappingPcCluster = visualPcClustersTemp[i];
                // update size of fused bounding boxes
                double xmax = visualBBox.x + visualBBox.x_width / 2;
                double xmin = visualBBox.x - visualBBox.x_width / 2;
                double ymax = visualBBox.y + visualBBox.y_width / 2;
                double ymin = visualBBox.y - visualBBox.y_width / 2;
                double zmax = visualBBox.z + visualBBox.z_width / 2;
                double zmin = visualBBox.z - visualBBox.z_width / 2;

                // get all potential bounding boxes that can merge
                for (int lidarIdx : overlappingLidarBBoxes){
                    overlappingBBoxes.push_back(lidarBBoxesTemp[lidarIdx]);
                    xmax = std::max(xmax, lidarBBoxesTemp[lidarIdx].x + lidarBBoxesTemp[lidarIdx].x_width / 2);
                    xmin = std::min(xmin, lidarBBoxesTemp[lidarIdx].x - lidarBBoxesTemp[lidarIdx].x_width / 2);
                    ymax = std::max(ymax, lidarBBoxesTemp[lidarIdx].y + lidarBBoxesTemp[lidarIdx].y_width / 2);
                    ymin = std::min(ymin, lidarBBoxesTemp[lidarIdx].y - lidarBBoxesTemp[lidarIdx].y_width / 2);
                    zmax = std::max(zmax, lidarBBoxesTemp[lidarIdx].z + lidarBBoxesTemp[lidarIdx].z_width / 2);
                    zmin = std::min(zmin, lidarBBoxesTemp[lidarIdx].z - lidarBBoxesTemp[lidarIdx].z_width / 2);
                    for (Eigen::Vector3d lidarPoints : lidarPcClustersTemp[lidarIdx]){
                        overlappingPcCluster.push_back(lidarPoints);
                    }
                    processedLidarBBoxes[lidarIdx] = true;
                }
                for (int visualIdx : overlappingVisualBBoxes){
                    overlappingBBoxes.push_back(visualBBoxesTemp[visualIdx]);
                    xmax = std::max(xmax, visualBBoxesTemp[visualIdx].x + visualBBoxesTemp[visualIdx].x_width / 2);
                    xmin = std::min(xmin, visualBBoxesTemp[visualIdx].x - visualBBoxesTemp[visualIdx].x_width / 2);
                    ymax = std::max(ymax, visualBBoxesTemp[visualIdx].y + visualBBoxesTemp[visualIdx].y_width / 2);
                    ymin = std::min(ymin, visualBBoxesTemp[visualIdx].y - visualBBoxesTemp[visualIdx].y_width / 2);
                    zmax = std::max(zmax, visualBBoxesTemp[visualIdx].z + visualBBoxesTemp[visualIdx].z_width / 2);
                    zmin = std::min(zmin, visualBBoxesTemp[visualIdx].z - visualBBoxesTemp[visualIdx].z_width / 2);
                    for (Eigen::Vector3d visualPoints : visualPcClustersTemp[visualIdx]){
                        overlappingPcCluster.push_back(visualPoints);
                    }
                    processedVisualBBoxes[visualIdx] = true;
                }

                std::vector<Eigen::Vector3d>& fusedPcCluster = overlappingPcCluster;
                Eigen::Vector3d fusedPcClusterCenter, fusedPcClusterStd;
                this->calcPcFeat(fusedPcCluster, fusedPcClusterCenter, fusedPcClusterStd);

                onboardDetector::box3D fusedBBox;
                fusedBBox.x = (xmin + xmax) / 2;
                fusedBBox.y = (ymin + ymax) / 2;
                fusedBBox.z = (zmin + zmax) / 2;
                fusedBBox.x_width = xmax - xmin;
                fusedBBox.y_width = ymax - ymin;
                fusedBBox.z_width = zmax - zmin;
                fusedBBox.Vx = 0;
                fusedBBox.Vy = 0;

                filteredBBoxesTemp.push_back(fusedBBox);
                filteredPcClustersTemp.push_back(fusedPcCluster); 
                filteredPcClusterCentersTemp.push_back(fusedPcClusterCenter);
                filteredPcClusterStdsTemp.push_back(fusedPcClusterStd);
                processedVisualBBoxes[i] = true;
            }
        }

        // STEP 4: Add rest of LiDAR detection 
        for (size_t i = 0; i < lidarBBoxesTemp.size(); ++i) {
            if (processedLidarBBoxes[i]) continue; // skip processed LiDAR boxes
            onboardDetector::box3D lidarBBox = lidarBBoxesTemp[i];

            // put the rest lidar bbox into the filtered bboxes
            filteredBBoxesTemp.push_back(lidarBBox);
            filteredPcClustersTemp.push_back(lidarPcClustersTemp[i]);
            filteredPcClusterCentersTemp.push_back(lidarPcClusterCentersTemp[i]);
            filteredPcClusterStdsTemp.push_back(lidarPcClusterStdsTemp[i]);
            processedLidarBBoxes[i] = true;
        }
        this->filteredBBoxesBeforeYolo_ = filteredBBoxesTemp; // for visualization


        // STEP 5: If YOLO detection results are available, improve the classification and splitting potential incorrect bboxes
        if (this->yoloDetectionResults_.detections.size() != 0){
            std::vector<int> best3DBBoxForYOLO(this->yoloDetectionResults_.detections.size(), -1);

            // Project 2D bbox in color image plane from 3D
            vision_msgs::msg::Detection2DArray filteredDetectionResults;
            for (int j=0; j<int(filteredBBoxesTemp.size()); ++j){
                onboardDetector::box3D bbox = filteredBBoxesTemp[j];

                // 1. transform the bounding boxes into the camera frame
                Eigen::Vector3d centerWorld (bbox.x, bbox.y, bbox.z);
                Eigen::Vector3d sizeWorld (bbox.x_width, bbox.y_width, bbox.z_width);
                Eigen::Vector3d centerCam, sizeCam;
                this->transformBBox(centerWorld, sizeWorld, -this->orientationColor_.inverse() * this->positionColor_, this->orientationColor_.inverse(), centerCam, sizeCam);

                // 2. find the top left and bottom right corner 3D position of the transformed bbox
                Eigen::Vector3d topleft (centerCam(0)-sizeCam(0)/2, centerCam(1)-sizeCam(1)/2, centerCam(2));
                Eigen::Vector3d bottomright (centerCam(0)+sizeCam(0)/2, centerCam(1)+sizeCam(1)/2, centerCam(2));

                // 3. project those two points into the camera image plane
                int tlX = (this->fxC_ * topleft(0) + this->cxC_ * topleft(2)) / topleft(2);
                int tlY = (this->fyC_ * topleft(1) + this->cyC_ * topleft(2)) / topleft(2);
                int brX = (this->fxC_ * bottomright(0) + this->cxC_ * bottomright(2)) / bottomright(2);
                int brY = (this->fyC_ * bottomright(1) + this->cyC_ * bottomright(2)) / bottomright(2);

                vision_msgs::msg::Detection2D result;
                result.bbox.center.position.x = tlX;
                result.bbox.center.position.y = tlY;
                result.bbox.size_x = brX - tlX;
                result.bbox.size_y = brY - tlY;
                filteredDetectionResults.detections.push_back(result);

                // cv::Rect bboxVis;
                // bboxVis.x = tlX;
                // bboxVis.y = tlY;
                // bboxVis.height = brY - tlY;
                // bboxVis.width = brX - tlX;
                // cv::rectangle(this->detectedColorImage_, bboxVis, cv::Scalar(0, 255, 0), 5, 8, 0);
            }

            for (int i=0; i<int(this->yoloDetectionResults_.detections.size()); ++i){
                int tlXTarget = int(this->yoloDetectionResults_.detections[i].bbox.center.position.x);
                int tlYTarget = int(this->yoloDetectionResults_.detections[i].bbox.center.position.y);
                int brXTarget = tlXTarget + int(this->yoloDetectionResults_.detections[i].bbox.size_x);
                int brYTarget = tlYTarget + int(this->yoloDetectionResults_.detections[i].bbox.size_y);

                cv::Rect bboxVis;
                bboxVis.x = tlXTarget;
                bboxVis.y = tlYTarget;
                bboxVis.height = brYTarget - tlYTarget;
                bboxVis.width = brXTarget - tlXTarget;
                cv::rectangle(this->detectedColorImage_, bboxVis, cv::Scalar(255, 0, 0), 5, 8, 0);

                // Define the text to be added
                std::string text = "dynamic";

                // Define the position for the text (above the bounding box)
                int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                double fontScale = 1.0;
                int thickness = 2;
                int baseline;
                cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
                cv::Point textOrg(bboxVis.x, bboxVis.y - 10);  // 10 pixels above the bounding box

                // Add the text to the image
                cv::putText(this->detectedColorImage_, text, textOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness, 8);

                double bestIOU = 0.0;
                int bestIdx = -1;
                for (int j = 0; j < int(filteredBBoxesTemp.size()); ++j) {
                    int tlX = int(filteredDetectionResults.detections[j].bbox.center.position.x);
                    int tlY = int(filteredDetectionResults.detections[j].bbox.center.position.y);
                    int brX = tlX + int(filteredDetectionResults.detections[j].bbox.size_x);
                    int brY = tlY + int(filteredDetectionResults.detections[j].bbox.size_y);

                    // check the IOU between yolo and projected bbox
                    double xOverlap = double(std::max(0, std::min(brX, brXTarget) - std::max(tlX, tlXTarget)));
                    double yOverlap = double(std::max(0, std::min(brY, brYTarget) - std::max(tlY, tlYTarget)));
                    double intersection = xOverlap * yOverlap;

                    // Calculate union area
                    double areaBox = double((brX - tlX) * (brY - tlY));
                    double areaBoxTarget = double((brXTarget - tlXTarget) * (brYTarget - tlYTarget));
                    double unionArea = areaBox + areaBoxTarget - intersection;

                    double IOU = (unionArea == 0) ? 0 : intersection / unionArea;
                    if (IOU > bestIOU){
                        bestIOU = IOU;
                        bestIdx = j;
                    }
                }

                if (bestIOU > 0.0){
                    best3DBBoxForYOLO[i] = bestIdx;
                }
            }

            std::map<int, std::vector<int>> box3DToYolo;
            for (int i = 0; i < int(best3DBBoxForYOLO.size()); ++i) {
                int idx3D = best3DBBoxForYOLO[i];
                if (idx3D >= 0 && idx3D < int(filteredBBoxesTemp.size())){
                    box3DToYolo[idx3D].push_back(i);
                }
            }

            std::vector<onboardDetector::box3D> newFilteredBBoxes;
            std::vector<std::vector<Eigen::Vector3d>> newFilteredPcClusters;
            std::vector<Eigen::Vector3d> newFilteredPcClusterCenters;
            std::vector<Eigen::Vector3d> newFilteredPcClusterStds;
            
            for (int idx3D = 0; idx3D < int(filteredBBoxesTemp.size()); ++idx3D) {
                auto it = box3DToYolo.find(idx3D);
                // *Case 1: No corresponding yolo box
                if (it == box3DToYolo.end()) {
                    newFilteredBBoxes.push_back(filteredBBoxesTemp[idx3D]);
                    newFilteredPcClusters.push_back(filteredPcClustersTemp[idx3D]);
                    newFilteredPcClusterCenters.push_back(filteredPcClusterCentersTemp[idx3D]);
                    newFilteredPcClusterStds.push_back(filteredPcClusterStdsTemp[idx3D]);
                    continue;
                }

                std::vector<int> yoloIndices = it->second;
                // *Case 2: one yolo box corresponds to one 3D box
                if (yoloIndices.size() == 1) {
                    filteredBBoxesTemp[idx3D].is_dynamic = true;
                    filteredBBoxesTemp[idx3D].is_human = true;
                    newFilteredBBoxes.push_back(filteredBBoxesTemp[idx3D]);
                    newFilteredPcClusters.push_back(filteredPcClustersTemp[idx3D]);
                    newFilteredPcClusterCenters.push_back(filteredPcClusterCentersTemp[idx3D]);
                    newFilteredPcClusterStds.push_back(filteredPcClusterStdsTemp[idx3D]);
                // *Case 3: multiple yolo boxes correspond to one 3D box
                } else {
                    std::vector<Eigen::Vector3d> cloudCluster = filteredPcClustersTemp[idx3D];

                    // iterate to assign all points
                    int allowMargin = 0; // pixel 
                    std::vector<int> assignment(cloudCluster.size(), -1);
                    for (size_t i = 0; i < cloudCluster.size(); ++i){
                        Eigen::Vector3d ptWorld = cloudCluster[i];
                        Eigen::Vector3d ptCam = this->orientationColor_.inverse() * (ptWorld - this->positionColor_);

                        int u = (this->fxC_ * ptCam(0) + this->cxC_ * ptCam(2)) / ptCam(2);
                        int v = (this->fyC_ * ptCam(1) + this->cyC_ * ptCam(2)) / ptCam(2);

                        int closestDist = std::numeric_limits<int>::max();
                        for (int yidx : yoloIndices){
                            int XTarget = int(this->yoloDetectionResults_.detections[yidx].bbox.center.position.x);
                            int YTarget = int(this->yoloDetectionResults_.detections[yidx].bbox.center.position.y);
                            int XTargetWid = int(this->yoloDetectionResults_.detections[yidx].bbox.size_x);
                            int YTargetWid = int(this->yoloDetectionResults_.detections[yidx].bbox.size_y);
                            int xMin = XTarget;
                            int xMax = XTarget + XTargetWid;
                            int yMin = YTarget;
                            int yMax = YTarget + YTargetWid;

                            if (u >= xMin-allowMargin && u <= xMax+allowMargin && v >= yMin-allowMargin && v <= yMax+allowMargin) {
                                // Horizontal signed distance
                                int horizontalDistance = 0;
                                if (u < xMin) {
                                    horizontalDistance = xMin - u; // Outside on the left
                                } else if (u > xMax) {
                                    horizontalDistance = u - xMax; // Outside on the right
                                } else {
                                    horizontalDistance = std::max(xMin - u, u - xMax); // Inside horizontally
                                }

                                // Compute signed distance to the closest edge
                                int signedDistance;
                                if (u < xMin || u > xMax || v < yMin || v > yMax) {
                                    // Outside: Take the larger of horizontal or vertical distance
                                    signedDistance = horizontalDistance;
                                } else {
                                    // Inside: Take the negative of the minimum distance to any edge
                                    signedDistance = horizontalDistance;
                                }
          
                                int distance = signedDistance;
                                if (distance < closestDist){
                                    assignment[i] = yidx;
                                    closestDist = distance;
                                }
                            }
                        }
                    }

                    std::vector<bool> flag(cloudCluster.size(), false);
                    for (int yidx : yoloIndices){
                        std::vector<Eigen::Vector3d> subCloud;
                        for (size_t i = 0; i < cloudCluster.size(); ++i){
                            if (flag[i]){
                                continue;
                            }

                            if (assignment[i] == yidx){
                                subCloud.push_back(cloudCluster[i]);
                                flag[i] = true;
                            }
                        }
                        if (subCloud.size() != 0){
                            onboardDetector::box3D newBox;
                            Eigen::Vector3d center, stddev;
                            center = computeCenter(subCloud);

                            double xMin = std::numeric_limits<double>::max(), xMax = std::numeric_limits<double>::lowest();
                            double yMin = std::numeric_limits<double>::max(), yMax = std::numeric_limits<double>::lowest();
                            double zMin = std::numeric_limits<double>::max(), zMax = std::numeric_limits<double>::lowest();

                            for (const auto &pt : subCloud) {
                                xMin = std::min(xMin, pt.x());
                                xMax = std::max(xMax, pt.x());
                                yMin = std::min(yMin, pt.y());
                                yMax = std::max(yMax, pt.y());
                                zMin = std::min(zMin, pt.z());
                                zMax = std::max(zMax, pt.z());
                            }
                            // create a new bounding box
                            newBox.x = (xMin + xMax) / 2.;
                            newBox.y = (yMin + yMax) / 2.;
                            newBox.z = (zMin + zMax) / 2.;
                            newBox.x_width = xMax - xMin;
                            newBox.y_width = yMax - yMin;
                            newBox.z_width = zMax - zMin;
                            if (newBox.x_width <= 0 or newBox.y_width <= 0 or newBox.x_width <= 0){
                                continue;
                            }

                            newBox.is_dynamic = true;
                            newBox.is_human = true;

                            stddev = computeStd(subCloud, center);
                            newFilteredBBoxes.push_back(newBox);
                            newFilteredPcClusters.push_back(subCloud);
                            newFilteredPcClusterCenters.push_back(center);
                            newFilteredPcClusterStds.push_back(stddev);
                        }
                    }
                }
            }
            filteredBBoxesTemp = newFilteredBBoxes;
            filteredPcClustersTemp = newFilteredPcClusters;
            filteredPcClusterCentersTemp = newFilteredPcClusterCenters;
            filteredPcClusterStdsTemp = newFilteredPcClusterStds;
        }
        this->filteredBBoxes_ = filteredBBoxesTemp;
        this->filteredPcClusters_ = filteredPcClustersTemp;
        this->filteredPcClusterCenters_ = filteredPcClusterCentersTemp;
        this->filteredPcClusterStds_ = filteredPcClusterStdsTemp;
    }

    void dynamicDetector::transformUVBBoxes(std::vector<onboardDetector::box3D>& bboxes){
        bboxes.clear();
        for(size_t i = 0; i < this->uvDetector_->box3Ds.size(); ++i){
            onboardDetector::box3D bbox;
            double x = this->uvDetector_->box3Ds[i].x; 
            double y = this->uvDetector_->box3Ds[i].y;
            double z = this->uvDetector_->box3Ds[i].z;
            double xWidth = this->uvDetector_->box3Ds[i].x_width;
            double yWidth = this->uvDetector_->box3Ds[i].y_width;
            double zWidth = this->uvDetector_->box3Ds[i].z_width;

            Eigen::Vector3d center (x, y, z);
            Eigen::Vector3d size (xWidth, yWidth, zWidth);
            Eigen::Vector3d newCenter, newSize;

            this->transformBBox(center, size, this->positionDepth_, this->orientationDepth_, newCenter, newSize);

            // assign values to bounding boxes in the map frame
            bbox.x = newCenter(0);
            bbox.y = newCenter(1);
            bbox.z = newCenter(2);
            bbox.x_width = newSize(0);
            bbox.y_width = newSize(1);
            bbox.z_width = newSize(2);
            bboxes.push_back(bbox);            
        }        
    }

    void dynamicDetector::projectDepthImage(){
        this->projPointsNum_ = 0;

        int cols = this->depthImage_.cols;
        int rows = this->depthImage_.rows;
        uint16_t* rowPtr;

        Eigen::Vector3d currPointCam, currPointMap;
        double depth;
        const double inv_factor = 1.0 / this->depthScale_;
        const double inv_fx = 1.0 / this->fx_;
        const double inv_fy = 1.0 / this->fy_;

        // iterate through each pixel in the depth image
        for (int v=this->depthFilterMargin_; v<rows-this->depthFilterMargin_; v=v+this->skipPixel_){ // row
            rowPtr = this->depthImage_.ptr<uint16_t>(v) + this->depthFilterMargin_;
            for (int u=this->depthFilterMargin_; u<cols-this->depthFilterMargin_; u=u+this->skipPixel_){ // column
                depth = (*rowPtr) * inv_factor;
                
                if (*rowPtr == 0) {
                    depth = this->raycastMaxLength_ + 0.1;
                } else if (depth < this->depthMinValue_) {
                    continue;
                } else if (depth > this->depthMaxValue_) {
                    depth = this->raycastMaxLength_ + 0.1;
                }
                rowPtr =  rowPtr + this->skipPixel_;

                // get 3D point in camera frame
                currPointCam(0) = (u - this->cx_) * depth * inv_fx;
                currPointCam(1) = (v - this->cy_) * depth * inv_fy;
                currPointCam(2) = depth;
                currPointMap = this->orientationDepth_ * currPointCam + this->positionDepth_; // transform to map coordinate

                this->projPoints_[this->projPointsNum_] = currPointMap;
                this->pointsDepth_[this->projPointsNum_] = depth;
                this->projPointsNum_ = this->projPointsNum_ + 1;
            }
        } 
    }

    void dynamicDetector::filterPoints(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints){
        // currently there is only one filtered (might include more in the future)
        std::vector<Eigen::Vector3d> voxelFilteredPoints;
        this->voxelFilter(points, voxelFilteredPoints);

        filteredPoints.clear();
        for (const auto& point : voxelFilteredPoints){
            if (point.z() <= this->roofHeight_ && point.z() >= this->groundHeight_){
                filteredPoints.push_back(point);
            }
        }
    }


    void dynamicDetector::clusterPointsAndBBoxes(const std::vector<Eigen::Vector3d>& points, std::vector<onboardDetector::box3D>& bboxes, std::vector<std::vector<Eigen::Vector3d>>& pcClusters, std::vector<Eigen::Vector3d>& pcClusterCenters, std::vector<Eigen::Vector3d>& pcClusterStds){
        std::vector<onboardDetector::Point> pointsDB;
        this->eigenToDBPointVec(points, pointsDB, points.size());

        this->dbCluster_.reset(new DBSCAN (this->dbMinPointsCluster_, this->dbEpsilon_, pointsDB));

        // DBSCAN clustering
        this->dbCluster_->run();
        // get the cluster data with bounding boxes
        // iterate through all the clustered points and find number of clusters
        int clusterNum = 0;
        for (size_t i=0; i<this->dbCluster_->m_points.size(); ++i){
            onboardDetector::Point pDB = this->dbCluster_->m_points[i];
            if (pDB.clusterID > clusterNum){
                clusterNum = pDB.clusterID;
            }
        }

        
        // pcClusters.resize(clusterNum);
        std::vector<std::vector<Eigen::Vector3d>> pcClustersTemp;
        pcClustersTemp.resize(clusterNum);
        for (size_t i=0; i<this->dbCluster_->m_points.size(); ++i){
            onboardDetector::Point pDB = this->dbCluster_->m_points[i];
            if (pDB.clusterID > 0){
                Eigen::Vector3d p = this->dbPointToEigen(pDB);
                pcClustersTemp[pDB.clusterID-1].push_back(p);
            }            
        }

        // calculate the bounding boxes and clusters
        pcClusters.clear();
        bboxes.clear();
        // bboxes.resize(clusterNum);
        for (size_t i=0; i<pcClustersTemp.size(); ++i){
            onboardDetector::box3D box;

            double xmin = pcClustersTemp[i][0](0);
            double ymin = pcClustersTemp[i][0](1);
            double zmin = pcClustersTemp[i][0](2);
            double xmax = pcClustersTemp[i][0](0);
            double ymax = pcClustersTemp[i][0](1);
            double zmax = pcClustersTemp[i][0](2);
            for (size_t j=0; j<pcClustersTemp[i].size(); ++j){
                xmin = (pcClustersTemp[i][j](0)<xmin)?pcClustersTemp[i][j](0):xmin;
                ymin = (pcClustersTemp[i][j](1)<ymin)?pcClustersTemp[i][j](1):ymin;
                zmin = (pcClustersTemp[i][j](2)<zmin)?pcClustersTemp[i][j](2):zmin;
                xmax = (pcClustersTemp[i][j](0)>xmax)?pcClustersTemp[i][j](0):xmax;
                ymax = (pcClustersTemp[i][j](1)>ymax)?pcClustersTemp[i][j](1):ymax;
                zmax = (pcClustersTemp[i][j](2)>zmax)?pcClustersTemp[i][j](2):zmax;
            }
            box.id = i;

            box.x = (xmax + xmin)/2.0;
            box.y = (ymax + ymin)/2.0;
            box.z = (zmax + zmin)/2.0;
            box.x_width = (xmax - xmin)>0.1?(xmax-xmin):0.1;
            box.y_width = (ymax - ymin)>0.1?(ymax-ymin):0.1;
            box.z_width = (zmax - zmin);

            // filter out bounding boxes that are too large
            if(box.x_width > this->maxObjectSize_(0) || box.y_width > this->maxObjectSize_(1) || box.z_width > this->maxObjectSize_(2)){
                continue;
            }
            bboxes.push_back(box);
            pcClusters.push_back(pcClustersTemp[i]);
        }

        for (size_t i=0 ; i<pcClusters.size() ; ++i){
            Eigen::Vector3d pcClusterCenter(0.,0.,0.);
            Eigen::Vector3d pcClusterStd(0.,0.,0.);
            this->calcPcFeat(pcClusters[i], pcClusterCenter, pcClusterStd);
            pcClusterCenters.push_back(pcClusterCenter);
            pcClusterStds.push_back(pcClusterStd);
        }

    }

    void dynamicDetector::voxelFilter(const std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector3d>& filteredPoints){
        const double res = 0.1; // resolution of voxel
        int xVoxels = ceil(2*this->localSensorRange_(0)/res); int yVoxels = ceil(2*this->localSensorRange_(1)/res); int zVoxels = ceil(2*this->localSensorRange_(2)/res);
        int totalVoxels = xVoxels * yVoxels * zVoxels;
        // std::vector<bool> voxelOccupancyVec (totalVoxels, false);
        std::vector<int> voxelOccupancyVec (totalVoxels, 0);

        // Iterate through each points in the cloud
        filteredPoints.clear();
        
        for (int i=0; i<this->projPointsNum_; ++i){
            Eigen::Vector3d p = points[i];

            if (this->isInFilterRange(p) and p(2) >= this->groundHeight_ and this->pointsDepth_[i] <= this->raycastMaxLength_){
                // find the corresponding voxel id in the vector and check whether it is occupied
                int pID = this->posToAddress(p, res);

                // add one point
                voxelOccupancyVec[pID] +=1;

                // add only if thresh points are found
                if (voxelOccupancyVec[pID] == this->voxelOccThresh_){
                    filteredPoints.push_back(p);
                }
            }
        } 
    }

    void dynamicDetector::calcPcFeat(const std::vector<Eigen::Vector3d>& pcCluster, Eigen::Vector3d& pcClusterCenter, Eigen::Vector3d& pcClusterStd){
        int numPoints = pcCluster.size();
        
        // center
        for (int i=0 ; i<numPoints ; i++){
            pcClusterCenter(0) += pcCluster[i](0)/numPoints;
            pcClusterCenter(1) += pcCluster[i](1)/numPoints;
            pcClusterCenter(2) += pcCluster[i](2)/numPoints;
        }

        // std
        for (int i=0 ; i<numPoints ; i++){
            pcClusterStd(0) += std::pow(pcCluster[i](0) - pcClusterCenter(0),2);
            pcClusterStd(1) += std::pow(pcCluster[i](1) - pcClusterCenter(1),2);
            pcClusterStd(2) += std::pow(pcCluster[i](2) - pcClusterCenter(2),2);
        }        

        // take square root
        pcClusterStd(0) = std::sqrt(pcClusterStd(0)/numPoints);
        pcClusterStd(1) = std::sqrt(pcClusterStd(1)/numPoints);
        pcClusterStd(2) = std::sqrt(pcClusterStd(2)/numPoints);
    }


    double dynamicDetector::calBoxIOU(const onboardDetector::box3D& box1, const onboardDetector::box3D& box2, bool ignoreZmin){
        double box1Volume = box1.x_width * box1.y_width * box1.z_width;
        double box2Volume = box2.x_width * box2.y_width * box2.z_width;

        double l1Y = box1.y+box1.y_width/2.-(box2.y-box2.y_width/2.);
        double l2Y = box2.y+box2.y_width/2.-(box1.y-box1.y_width/2.);
        double l1X = box1.x+box1.x_width/2.-(box2.x-box2.x_width/2.);
        double l2X = box2.x+box2.x_width/2.-(box1.x-box1.x_width/2.);
        double l1Z = box1.z+box1.z_width/2.-(box2.z-box2.z_width/2.);
        double l2Z = box2.z+box2.z_width/2.-(box1.z-box1.z_width/2.);
        
        if (ignoreZmin){
            // modify box1 and box2 volumn based on the maximum lower z of two
            double zmin = std::max(box1.z - box1.z_width/2., box2.z - box2.z_width/2.);
            double zWidth1 = box1.z_width/2. + (box1.z - zmin);
            double zWidth2 = box2.z_width/2. + (box2.z - zmin);
            box1Volume = box1.x_width * box1.y_width * zWidth1;
            box2Volume = box2.x_width * box2.y_width * zWidth2;

            l1Z = box1.z+box1.z_width/2. - zmin;
            l2Z = box2.z+box2.z_width/2. - zmin;
        }
        
        double overlapX = std::min( l1X , l2X );
        double overlapY = std::min( l1Y , l2Y );
        double overlapZ = std::min( l1Z , l2Z );
       
        if (std::max(l1X, l2X)<=std::max(box1.x_width,box2.x_width)){ 
            overlapX = std::min(box1.x_width, box2.x_width);
        }
        if (std::max(l1Y, l2Y)<=std::max(box1.y_width,box2.y_width)){ 
            overlapY = std::min(box1.y_width, box2.y_width);
        }
        if (std::max(l1Z, l2Z)<=std::max(box1.z_width,box2.z_width)){ 
            overlapZ = std::min(box1.z_width, box2.z_width);
        }


        double overlapVolume = overlapX * overlapY *  overlapZ;
        double IOU = overlapVolume / (box1Volume+box2Volume-overlapVolume);
        
        // D-IOU
        if (overlapX<=0 || overlapY<=0 ||overlapZ<=0){
            IOU = 0;
        }
        return IOU;
    }

    void dynamicDetector::boxAssociation(std::vector<int>& bestMatch){
        int numObjs = int(this->filteredBBoxes_.size()); // current detected bboxes
        if (this->boxHist_.size() == 0){ // initialize new bounding box history if no history exists
            this->boxHist_.resize(numObjs);
            this->pcHist_.resize(numObjs);
            this->pcCenterHist_.resize(numObjs);
            bestMatch.resize(this->filteredBBoxes_.size(), -1); // first detection no match
            for (int i=0 ; i<numObjs ; ++i){
                // initialize history for bbox, pc and KF
                this->boxHist_[i].push_back(this->filteredBBoxes_[i]);
                this->pcHist_[i].push_back(this->filteredPcClusters_[i]);
                this->pcCenterHist_[i].push_back(this->filteredPcClusterCenters_[i]);
                MatrixXd states, A, B, H, P, Q, R;       
                this->kalmanFilterMatrixAcc(this->filteredBBoxes_[i], states, A, B, H, P, Q, R);
                onboardDetector::kalman_filter newFilter;
                newFilter.setup(states, A, B, H, P, Q, R);
                this->filters_.push_back(newFilter);
            }
        }
        else{
            // start association only if a new detection is available
            if (this->newDetectFlag_){
                this->boxAssociationHelper(bestMatch);
            }
        }

        this->newDetectFlag_ = false; // the most recent detection has been associated
    }

    void dynamicDetector::boxAssociationHelper(std::vector<int>& bestMatch){
        int numObjs = int(this->filteredBBoxes_.size());
        std::vector<onboardDetector::box3D> prevBBoxes;
        std::vector<Eigen::Vector3d> prevPcCenters;
        std::vector<Eigen::VectorXd> prevBBoxesFeat;
        std::vector<onboardDetector::box3D> propedBBoxes;
        std::vector<Eigen::Vector3d> propedPcCenters;
        std::vector<Eigen::VectorXd> propedBBoxesFeat;
        std::vector<Eigen::VectorXd> currBBoxesFeat;
        currBBoxesFeat.resize(numObjs);
        bestMatch.resize(numObjs);

        // Features for current detected bboxes
        this->genFeatHelper(this->filteredBBoxes_, this->filteredPcClusterCenters_, currBBoxesFeat);

        // Features for previous time step bboxes
        this->getPrevBBoxes(prevBBoxes, prevPcCenters);
        this->genFeatHelper(prevBBoxes, prevPcCenters, prevBBoxesFeat);

        // Features for propogated bboxes
        this->linearProp(propedBBoxes, propedPcCenters);
        this->genFeatHelper(propedBBoxes, propedPcCenters, propedBBoxesFeat);

        // calculate association: find best match
        this->findBestMatch(prevBBoxes, prevBBoxesFeat, propedBBoxes, propedBBoxesFeat, currBBoxesFeat, bestMatch);      
    }

    void dynamicDetector::genFeatHelper( 
        const std::vector<onboardDetector::box3D>& boxes,
        const std::vector<Eigen::Vector3d>& pcCenters,
        std::vector<Eigen::VectorXd>& features){ 
        Eigen::VectorXd featureWeights = Eigen::VectorXd::Zero(9); // 3 pos + 3 size + 3 pc centers
        featureWeights = this->featureWeights_;
        features.resize(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            Eigen::VectorXd feature = Eigen::VectorXd::Zero(10);
            feature(0) = (boxes[i].x - this->position_(0)) * featureWeights(0);
            feature(1) = (boxes[i].y - this->position_(1)) * featureWeights(1);
            feature(2) = (boxes[i].z - this->position_(2)) * featureWeights(2);
            feature(3) = boxes[i].x_width * featureWeights(3);
            feature(4) = boxes[i].y_width * featureWeights(4);
            feature(5) = boxes[i].z_width * featureWeights(5);
            feature(6) = pcCenters[i](0) * featureWeights(6);
            feature(7) = pcCenters[i](1) * featureWeights(7);
            feature(8) = pcCenters[i](2) * featureWeights(8);

            // fix nan problem
            for(int j = 0; j < feature.size(); ++j) {
                if (std::isnan(feature(j)) || std::isinf(feature(j))) {
                    feature(j) = 0;
                }
            }
            features[i] = feature;
        }
    }

    void dynamicDetector::getPrevBBoxes(std::vector<onboardDetector::box3D>& prevBoxes, std::vector<Eigen::Vector3d>& prevPcCenters){
        onboardDetector::box3D prevBox;
        for (size_t i=0 ; i<this->boxHist_.size() ; i++){
            prevBox = this->boxHist_[i][0];
            prevBoxes.push_back(prevBox);

            Eigen::Vector3d prevPcCenter = this->pcCenterHist_[i][0];
            prevPcCenters.push_back(prevPcCenter);
        }
    }
      
    void dynamicDetector::linearProp(std::vector<onboardDetector::box3D>& propedBBoxes, std::vector<Eigen::Vector3d>& propedPcCenters){
        onboardDetector::box3D propedBBox;
        for (size_t i=0 ; i<this->boxHist_.size() ; i++){
            propedBBox = this->boxHist_[i][0];
            propedBBox.x += propedBBox.Vx*this->dt_;
            propedBBox.y += propedBBox.Vy*this->dt_;
            propedBBoxes.push_back(propedBBox);

            Eigen::Vector3d propedPcCenter = this->pcCenterHist_[i][0];
            propedPcCenter(0) += propedBBox.Vx*this->dt_;
            propedPcCenter(1) += propedBBox.Vy*this->dt_;
            propedPcCenters.push_back(propedPcCenter);
        }
    }

    void dynamicDetector::findBestMatch(const std::vector<onboardDetector::box3D>& prevBBoxes, const std::vector<Eigen::VectorXd>& prevBBoxesFeat, 
                                        const std::vector<onboardDetector::box3D>& propedBBoxes, const std::vector<Eigen::VectorXd>& propedBBoxesFeat, 
                                        const std::vector<Eigen::VectorXd>& currBBoxesFeat, std::vector<int>& bestMatch){
        int numObjs = this->filteredBBoxes_.size();
        std::vector<double> bestSims; // best similarity
        bestSims.resize(numObjs, 0);

        for (int i=0 ; i<numObjs ; i++){
            double bestSim = -1.;
            int bestMatchInd = -1;
            onboardDetector::box3D currBBox = this->filteredBBoxes_[i];
            
            for (size_t j=0 ; j<propedBBoxes.size() ; j++){
                onboardDetector::box3D propedBBox = propedBBoxes[j];
                double propedWidth = std::max(propedBBox.x_width, propedBBox.y_width);
                double currWidth = std::max(currBBox.x_width, currBBox.y_width);
                if (std::abs(propedWidth - currWidth) < this->maxMatchSizeRange_){
                    if (pow(pow(propedBBox.x - currBBox.x, 2) + pow(propedBBox.y - currBBox.y, 2), 0.5) < this->maxMatchRange_){
                        // calculate the velocity feature based on propedBBox and currBBox
                        double simPrev = prevBBoxesFeat[j].dot(currBBoxesFeat[i])/(prevBBoxesFeat[j].norm()*currBBoxesFeat[i].norm());
                        double simProped = propedBBoxesFeat[j].dot(currBBoxesFeat[i])/(propedBBoxesFeat[j].norm()*currBBoxesFeat[i].norm());
                        double sim = simPrev + simProped;
                        if (sim > bestSim){
                            bestSim = sim;
                            bestMatchInd = j;
                        }
                    }

                }
            }
            bestSims[i] = bestSim;
            bestMatch[i] = bestMatchInd;
        }
    }

    void dynamicDetector::kalmanFilterAndUpdateHist(const std::vector<int>& bestMatch){
        std::vector<std::deque<onboardDetector::box3D>> boxHistTemp; 
        std::vector<std::deque<std::vector<Eigen::Vector3d>>> pcHistTemp;
        std::vector<std::deque<Eigen::Vector3d>> pcCenterHistTemp;
        std::vector<onboardDetector::kalman_filter> filtersTemp;
        std::deque<onboardDetector::box3D> newSingleBoxHist;
        std::deque<std::vector<Eigen::Vector3d>> newSinglePcHist; 
        std::deque<Eigen::Vector3d> newSinglePcCenterHist; 
        onboardDetector::kalman_filter newFilter;
        std::vector<onboardDetector::box3D> trackedBBoxesTemp;

        newSingleBoxHist.resize(0);
        newSinglePcHist.resize(0);
        newSinglePcCenterHist.resize(0);
        int numObjs = this->filteredBBoxes_.size();

        for (int i=0 ; i<numObjs ; i++){
            onboardDetector::box3D newEstimatedBBox; // from kalman filter

            // inheret history. push history one by one
            if (bestMatch[i]>=0){
                boxHistTemp.push_back(this->boxHist_[bestMatch[i]]);
                pcHistTemp.push_back(this->pcHist_[bestMatch[i]]);
                pcCenterHistTemp.push_back(this->pcCenterHist_[bestMatch[i]]);
                filtersTemp.push_back(this->filters_[bestMatch[i]]);

                // kalman filter to get new state estimation
                onboardDetector::box3D currDetectedBBox = this->filteredBBoxes_[i];

                Eigen::MatrixXd Z;
                this->getKalmanObservationAcc(currDetectedBBox, bestMatch[i], Z);
                filtersTemp.back().estimate(Z, MatrixXd::Zero(6,1));
                
                
                newEstimatedBBox.x = filtersTemp.back().output(0);
                newEstimatedBBox.y = filtersTemp.back().output(1);
                newEstimatedBBox.z = currDetectedBBox.z;
                newEstimatedBBox.Vx = filtersTemp.back().output(2);
                newEstimatedBBox.Vy = filtersTemp.back().output(3);
                newEstimatedBBox.Ax = filtersTemp.back().output(4);
                newEstimatedBBox.Ay = filtersTemp.back().output(5);   
                          

                newEstimatedBBox.x_width = currDetectedBBox.x_width;
                newEstimatedBBox.y_width = currDetectedBBox.y_width;
                newEstimatedBBox.z_width = currDetectedBBox.z_width;
                newEstimatedBBox.is_dynamic = currDetectedBBox.is_dynamic;
                newEstimatedBBox.is_human = currDetectedBBox.is_human;
            }
            else{
                boxHistTemp.push_back(newSingleBoxHist);
                pcHistTemp.push_back(newSinglePcHist);
                pcCenterHistTemp.push_back(newSinglePcCenterHist);

                // create new kalman filter for this object
                onboardDetector::box3D currDetectedBBox = this->filteredBBoxes_[i];
                MatrixXd states, A, B, H, P, Q, R;    
                this->kalmanFilterMatrixAcc(currDetectedBBox, states, A, B, H, P, Q, R);
                
                newFilter.setup(states, A, B, H, P, Q, R);
                filtersTemp.push_back(newFilter);
                newEstimatedBBox = currDetectedBBox;
                
            }

            // pop old data if len of hist > size limit
            if (int(boxHistTemp[i].size()) == this->histSize_){
                boxHistTemp[i].pop_back();
                pcHistTemp[i].pop_back();
                pcCenterHistTemp[i].pop_back();
            }

            // push new data into history
            boxHistTemp[i].push_front(newEstimatedBBox); 
            pcHistTemp[i].push_front(this->filteredPcClusters_[i]);
            pcCenterHistTemp[i].push_front(this->filteredPcClusterCenters_[i]);

            // update new tracked bounding boxes
            trackedBBoxesTemp.push_back(newEstimatedBBox);
        }
  

        if (boxHistTemp.size()){
            for (size_t i=0; i<trackedBBoxesTemp.size(); ++i){ 
                if (int(boxHistTemp[i].size()) >= this->fixSizeHistThresh_){
                    if ((abs(trackedBBoxesTemp[i].x_width-boxHistTemp[i][1].x_width)/boxHistTemp[i][1].x_width) <= this->fixSizeDimThresh_ &&
                        (abs(trackedBBoxesTemp[i].y_width-boxHistTemp[i][1].y_width)/boxHistTemp[i][1].y_width) <= this->fixSizeDimThresh_&&
                        (abs(trackedBBoxesTemp[i].z_width-boxHistTemp[i][1].z_width)/boxHistTemp[i][1].z_width) <= this->fixSizeDimThresh_){
                        trackedBBoxesTemp[i].x_width = boxHistTemp[i][1].x_width;
                        trackedBBoxesTemp[i].y_width = boxHistTemp[i][1].y_width;
                        trackedBBoxesTemp[i].z_width = boxHistTemp[i][1].z_width;
                        boxHistTemp[i][0].x_width = trackedBBoxesTemp[i].x_width;
                        boxHistTemp[i][0].y_width = trackedBBoxesTemp[i].y_width;
                        boxHistTemp[i][0].z_width = trackedBBoxesTemp[i].z_width;
                    }

                }
            }
        }
        
        // update history member variable
        this->boxHist_ = boxHistTemp;
        this->pcHist_ = pcHistTemp;
        this->pcCenterHist_ = pcCenterHistTemp;
        this->filters_ = filtersTemp;

        // update tracked bounding boxes
        this->trackedBBoxes_=  trackedBBoxesTemp;
    }

    void dynamicDetector::kalmanFilterMatrixVel(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R){
        states.resize(4,1);
        states(0) = currDetectedBBox.x;
        states(1) = currDetectedBBox.y;
        // init vel and acc to zeros
        states(2) = 0.;
        states(3) = 0.;

        MatrixXd ATemp;
        ATemp.resize(4, 4);
        ATemp <<  0, 0, 1, 0,
                  0, 0, 0, 1,
                  0, 0, 0, 0,
                  0 ,0, 0, 0;
        A = MatrixXd::Identity(4,4) + this->dt_*ATemp;
        B = MatrixXd::Zero(4, 4);
        H = MatrixXd::Identity(4, 4);
        P = MatrixXd::Identity(4, 4) * this->eP_;
        Q = MatrixXd::Identity(4, 4);
        Q(0,0) *= this->eQPos_; Q(1,1) *= this->eQPos_; Q(2,2) *= this->eQVel_; Q(3,3) *= this->eQVel_; 
        R = MatrixXd::Identity(4, 4);
        R(0,0) *= this->eRPos_; R(1,1) *= this->eRPos_; R(2,2) *= this->eRVel_; R(3,3) *= this->eRVel_;

    }

    void dynamicDetector::kalmanFilterMatrixAcc(const onboardDetector::box3D& currDetectedBBox, MatrixXd& states, MatrixXd& A, MatrixXd& B, MatrixXd& H, MatrixXd& P, MatrixXd& Q, MatrixXd& R){
        states.resize(6,1);
        states(0) = currDetectedBBox.x;
        states(1) = currDetectedBBox.y;
        // init vel and acc to zeros
        states(2) = 0.;
        states(3) = 0.;
        states(4) = 0.;
        states(5) = 0.;

        MatrixXd ATemp;
        ATemp.resize(6, 6);

        ATemp <<  1, 0, this->dt_, 0, 0.5*pow(this->dt_, 2), 0,
                  0, 1, 0, this->dt_, 0, 0.5*pow(this->dt_, 2),
                  0, 0, 1, 0, this->dt_, 0,
                  0 ,0, 0, 1, 0, this->dt_,
                  0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 1;
        A = ATemp;
        B = MatrixXd::Zero(6, 6);
        H = MatrixXd::Identity(6, 6);
        P = MatrixXd::Identity(6, 6) * this->eP_;
        Q = MatrixXd::Identity(6, 6);
        Q(0,0) *= this->eQPos_; Q(1,1) *= this->eQPos_; Q(2,2) *= this->eQVel_; Q(3,3) *= this->eQVel_; Q(4,4) *= this->eQAcc_; Q(5,5) *= this->eQAcc_;
        R = MatrixXd::Identity(6, 6);
        R(0,0) *= this->eRPos_; R(1,1) *= this->eRPos_; R(2,2) *= this->eRVel_; R(3,3) *= this->eRVel_; R(4,4) *= this->eRAcc_; R(5,5) *= this->eRAcc_;
    }

    void dynamicDetector::getKalmanObservationVel(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z){
        Z.resize(4,1);
        Z(0) = currDetectedBBox.x; 
        Z(1) = currDetectedBBox.y;

        // use previous k frame for velocity estimation
        int k = this->kfAvgFrames_;
        int historySize = this->boxHist_[bestMatchIdx].size();
        if (historySize < k){
            k = historySize;
        }
        onboardDetector::box3D prevMatchBBox = this->boxHist_[bestMatchIdx][k-1];

        Z(2) = (currDetectedBBox.x-prevMatchBBox.x)/(this->dt_*k);
        Z(3) = (currDetectedBBox.y-prevMatchBBox.y)/(this->dt_*k);
    }

    void dynamicDetector::getKalmanObservationAcc(const onboardDetector::box3D& currDetectedBBox, int bestMatchIdx, MatrixXd& Z){
        Z.resize(6, 1);
        Z(0) = currDetectedBBox.x;
        Z(1) = currDetectedBBox.y;

        // use previous k frame for velocity estimation
        int k = this->kfAvgFrames_;
        int historySize = this->boxHist_[bestMatchIdx].size();
        if (historySize < k){
            k = historySize;
        }
        onboardDetector::box3D prevMatchBBox = this->boxHist_[bestMatchIdx][k-1];

        Z(2) = (currDetectedBBox.x - prevMatchBBox.x)/(this->dt_*k);
        Z(3) = (currDetectedBBox.y - prevMatchBBox.y)/(this->dt_*k);
        Z(4) = (Z(2) - prevMatchBBox.Vx)/(this->dt_*k);
        Z(5) = (Z(3) - prevMatchBBox.Vy)/(this->dt_*k);
    }
 
    void dynamicDetector::getDynamicPc(std::vector<Eigen::Vector3d>& dynamicPc){
        Eigen::Vector3d curPoint;
        for (size_t i=0; i<this->filteredPcClusters_.size(); ++i){
            for (size_t j=0; j<this->filteredPcClusters_[i].size(); ++j){
                curPoint = this->filteredPcClusters_[i][j];
                for (size_t k=0; k<this->dynamicBBoxes_.size(); ++k){
                    if (abs(curPoint(0)-this->dynamicBBoxes_[k].x)<=this->dynamicBBoxes_[k].x_width/2 and 
                        abs(curPoint(1)-this->dynamicBBoxes_[k].y)<=this->dynamicBBoxes_[k].y_width/2 and 
                        abs(curPoint(2)-this->dynamicBBoxes_[k].z)<=this->dynamicBBoxes_[k].z_width/2) {
                        dynamicPc.push_back(curPoint);
                        break;
                    }
                }
            }
        }
    } 
    
    void dynamicDetector::publishUVImages(){
        if (this->uvDetector_ != NULL){
            sensor_msgs::msg::Image::SharedPtr depthBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->depth_show).toImageMsg();
            sensor_msgs::msg::Image::SharedPtr UmapBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->U_map_show).toImageMsg();
            sensor_msgs::msg::Image::SharedPtr birdBoxMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", this->uvDetector_->bird_view).toImageMsg();  
            this->uvDepthMapPub_->publish(*depthBoxMsg);
            this->uDepthMapPub_->publish(*UmapBoxMsg);
            this->uvBirdViewPub_->publish(*birdBoxMsg);
        }     
    }


    void dynamicDetector::publishColorImages(){
        sensor_msgs::msg::Image::SharedPtr detectedColorImgMsg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", this->detectedColorImage_).toImageMsg();
        this->detectedColorImgPub_->publish(*detectedColorImgMsg);
    }

    void dynamicDetector::publishPoints(const std::vector<Eigen::Vector3d>& points, const rclcpp::Publisher<PointCloud2Msg>::SharedPtr& publisher){
        pcl::PointXYZ pt;
        pcl::PointCloud<pcl::PointXYZ> cloud;        
        for (size_t i=0; i<points.size(); ++i){
            pt.x = points[i](0);
            pt.y = points[i](1);
            pt.z = points[i](2);
            cloud.push_back(pt);
        }    
        cloud.width = cloud.points.size();
        cloud.height = 1;
        cloud.is_dense = true;
        cloud.header.frame_id = this->visFrame_;

        sensor_msgs::msg::PointCloud2 cloudMsg;
        pcl::toROSMsg(cloud, cloudMsg);
        publisher->publish(cloudMsg);
    }


    void dynamicDetector::publish3dBox(const std::vector<box3D>& boxes,
                                   const rclcpp::Publisher<MarkerArrayMsg>::SharedPtr& publisher,
                                   double r, double g, double b){
        visualization_msgs::msg::MarkerArray markers;

        for (size_t i = 0; i < boxes.size(); i++)
        {
            visualization_msgs::msg::Marker line;
            line.header.frame_id = this->visFrame_;
            line.ns = "box3D";
            line.id = i;
            line.type = visualization_msgs::msg::Marker::LINE_LIST;
            line.action = visualization_msgs::msg::Marker::ADD;
            line.scale.x = 0.06;
            line.color.r = r;
            line.color.g = g;
            line.color.b = b;
            line.color.a = 1.0;
            // 0.05s < 實際發布間隔(~0.09s@11Hz) → 框在兩次發布間過期，RViz 看起來閃爍/延遲
            line.lifetime = rclcpp::Duration::from_seconds(0.25);
            line.pose.orientation.x = 0.0;
            line.pose.orientation.y = 0.0;
            line.pose.orientation.z = 0.0;
            line.pose.orientation.w = 1.0;
            line.pose.position.x = boxes[i].x;
            line.pose.position.y = boxes[i].y;
            double x_width = boxes[i].x_width;
            double y_width = boxes[i].y_width;

            double top = boxes[i].z + boxes[i].z_width / 2.0;
            double z_width = top / 2.0;
            line.pose.position.z = z_width; 

            geometry_msgs::msg::Point corner[8];
            corner[0].x = -x_width / 2.0; corner[0].y = -y_width / 2.0; corner[0].z = -z_width;
            corner[1].x = -x_width / 2.0; corner[1].y =  y_width / 2.0; corner[1].z = -z_width;
            corner[2].x =  x_width / 2.0; corner[2].y =  y_width / 2.0; corner[2].z = -z_width;
            corner[3].x =  x_width / 2.0; corner[3].y = -y_width / 2.0; corner[3].z = -z_width;

            corner[4].x = -x_width / 2.0; corner[4].y = -y_width / 2.0; corner[4].z =  z_width;
            corner[5].x = -x_width / 2.0; corner[5].y =  y_width / 2.0; corner[5].z =  z_width;
            corner[6].x =  x_width / 2.0; corner[6].y =  y_width / 2.0; corner[6].z =  z_width;
            corner[7].x =  x_width / 2.0; corner[7].y = -y_width / 2.0; corner[7].z =  z_width;

            int edgeIdx[12][2] = {
                {0,1}, {1,2}, {2,3}, {3,0},  
                {4,5}, {5,6}, {6,7}, {7,4},  
                {0,4}, {1,5}, {2,6}, {3,7}   
            };

            for (int e = 0; e < 12; e++)
            {
                line.points.push_back(corner[edgeIdx[e][0]]);
                line.points.push_back(corner[edgeIdx[e][1]]);
            }

            markers.markers.push_back(line);
        }

        publisher->publish(markers);
    }


    void dynamicDetector::publishHistoryTraj(){
        visualization_msgs::msg::MarkerArray trajMsg;
        int countMarker = 0;
        for (size_t i=0; i<this->boxHist_.size(); ++i){
            if (this->boxHist_[i].size() > 1){
                visualization_msgs::msg::Marker traj;
                traj.header.frame_id = this->visFrame_;
                traj.header.stamp = this->now();
                traj.ns = "dynamic_detector";
                traj.id = countMarker;
                traj.type = visualization_msgs::msg::Marker::LINE_LIST;
                traj.scale.x = 0.03;
                traj.scale.y = 0.03;
                traj.scale.z = 0.03;
                traj.color.a = 1.0; // Don't forget to set the alpha!
                traj.color.r = 0.0;
                traj.color.g = 1.0;
                traj.color.b = 0.0;
                traj.pose.orientation.w = 1.0;
                traj.pose.orientation.x = 0.0;
                traj.pose.orientation.y = 0.0;
                traj.pose.orientation.z = 0.0;
                for (size_t j=0; j<this->boxHist_[i].size()-1; ++j){
                    geometry_msgs::msg::Point p1, p2;
                    onboardDetector::box3D box1 = this->boxHist_[i][j];
                    onboardDetector::box3D box2 = this->boxHist_[i][j+1];
                    p1.x = box1.x; p1.y = box1.y; p1.z = box1.z;
                    p2.x = box2.x; p2.y = box2.y; p2.z = box2.z;
                    traj.points.push_back(p1);
                    traj.points.push_back(p2);
                }

                ++countMarker;
                trajMsg.markers.push_back(traj);
            }
        }
        this->historyTrajPub_->publish(trajMsg);
    }

    void dynamicDetector::publishVelVis(){ // publish velocities for all tracked objects
        visualization_msgs::msg::MarkerArray velVisMsg;
        int countMarker = 0;
        for (size_t i=0; i<this->trackedBBoxes_.size(); ++i){
            // 靜態物的 KF 速度是簇心量測噪音（量小、變化快），畫出來干擾判讀；
            // 只顯示超過動態門檻的速度（低於門檻者本來也不會被判動態）
            double vNormCheck = sqrt(pow(this->trackedBBoxes_[i].Vx, 2) + pow(this->trackedBBoxes_[i].Vy, 2));
            if (vNormCheck < this->dynaVelThresh_){
                continue;
            }
            visualization_msgs::msg::Marker velMarker;
            velMarker.header.frame_id = this->visFrame_;
            velMarker.header.stamp = this->now();
            velMarker.ns = "dynamic_detector";
            velMarker.id =  countMarker;
            velMarker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            velMarker.pose.position.x = this->trackedBBoxes_[i].x;
            velMarker.pose.position.y = this->trackedBBoxes_[i].y;
            velMarker.pose.position.z = this->trackedBBoxes_[i].z + this->trackedBBoxes_[i].z_width/2. + 0.3;
            velMarker.scale.x = 0.15;
            velMarker.scale.y = 0.15;
            velMarker.scale.z = 0.15;
            velMarker.color.a = 1.0;
            velMarker.color.r = 1.0;
            velMarker.color.g = 0.0;
            velMarker.color.b = 0.0;
            velMarker.lifetime = rclcpp::Duration::from_seconds(0.1);
            double vx = this->trackedBBoxes_[i].Vx;
            double vy = this->trackedBBoxes_[i].Vy;
            double vNorm = sqrt(vx*vx+vy*vy);
            std::string velText = "Vx=" + std::to_string(vx) + ", Vy=" + std::to_string(vy) + ", |V|=" + std::to_string(vNorm);
            velMarker.text = velText;
            velVisMsg.markers.push_back(velMarker);
            ++countMarker;
        }
        this->velVisPub_->publish(velVisMsg);
    }

    void dynamicDetector::publishLidarClusters(){
        sensor_msgs::msg::PointCloud2 lidarClustersMsg;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (size_t i=0; i<this->lidarClusters_.size(); ++i){
            onboardDetector::Cluster & cluster = this->lidarClusters_[i];

            std_msgs::msg::ColorRGBA color;
            srand(cluster.cluster_id);
            color.r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            color.g = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            color.b = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            // color.r = 0.5;
            // color.g = 0.5;
            // color.b = 0.5;
            // color.a = 1.0;

            for (size_t j=0; j<cluster.points->size(); ++j){
                pcl::PointXYZRGB point;
                const pcl::PointXYZ & pt = cluster.points->at(j);
                point.x = pt.x;
                point.y = pt.y;
                point.z = pt.z;
                point.r = color.r * 255;
                point.g = color.g * 255;
                point.b = color.b * 255;
                colored_cloud->push_back(point);
            }
        }
        pcl::toROSMsg(*colored_cloud, lidarClustersMsg);
        lidarClustersMsg.header.frame_id = this->visFrame_;
        lidarClustersMsg.header.stamp = this->now();
        this->lidarClustersPub_->publish(lidarClustersMsg);
    }

    void dynamicDetector::publishFilteredPoints(){
        sensor_msgs::msg::PointCloud2 filteredPointsMsg;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (size_t i=0; i<this->filteredPcClusters_.size(); ++i){
            std_msgs::msg::ColorRGBA color;
            color.r = 0.5;
            color.g = 0.5;
            color.b = 0.5;
            color.a = 1.0;

            for (size_t j=0; j<this->filteredPcClusters_[i].size(); ++j){
                pcl::PointXYZRGB point;
                point.x = this->filteredPcClusters_[i][j](0);
                point.y = this->filteredPcClusters_[i][j](1);
                point.z = this->filteredPcClusters_[i][j](2);
                point.r = color.r * 255;
                point.g = color.g * 255;
                point.b = color.b * 255;
                colored_cloud->push_back(point);
            }
        }
        pcl::toROSMsg(*colored_cloud, filteredPointsMsg);
        filteredPointsMsg.header.frame_id = this->visFrame_;
        filteredPointsMsg.header.stamp = this->now();
        this->filteredPointsPub_->publish(filteredPointsMsg);
    }

    void dynamicDetector::publishRawDynamicPoints(){
        if (not this->latestCloud_){
            return;
        }
        try {
            pcl::PointCloud<pcl::PointXYZ>::Ptr globalCloud(new pcl::PointCloud<pcl::PointXYZ>);
            if (this->hasSensorPose_) {
                pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::fromROSMsg(*this->latestCloud_, *tempCloud);
                
                Eigen::Affine3d transform = Eigen::Affine3d::Identity();
                transform.linear() = this->orientationLidar_;
                transform.translation() = this->positionLidar_;
                
                pcl::transformPointCloud(*tempCloud, *globalCloud, transform);
                sensor_msgs::msg::PointCloud2 cloudMsg;
                pcl::toROSMsg(*globalCloud, cloudMsg);
                cloudMsg.header.frame_id = this->visFrame_;
                cloudMsg.header.stamp = this->now();
                this->rawLidarPointsPub_->publish(cloudMsg);
            }
            else {
                pcl::fromROSMsg(*this->latestCloud_, *globalCloud);
            }
            
            std::vector<Eigen::Vector3d> dynamicEigenPoints;
            
            for (const auto& box : this->dynamicBBoxes_) {
                if (!box.is_dynamic)
                    continue;
                
                double xmin = box.x - box.x_width / 2.0;
                double xmax = box.x + box.x_width / 2.0;
                double ymin = box.y - box.y_width / 2.0;
                double ymax = box.y + box.y_width / 2.0;
                double zmin = box.z - box.z_width / 2.0;
                double zmax = box.z + box.z_width / 2.0;
                
                for (const auto& point : globalCloud->points) {
                    if (point.x >= xmin && point.x <= xmax &&
                        point.y >= ymin && point.y <= ymax &&
                        point.z >= zmin && point.z <= zmax)
                    {
                        dynamicEigenPoints.push_back(Eigen::Vector3d(point.x, point.y, point.z));
                    }
                }
            }
            
            if (dynamicEigenPoints.empty()) {
                return;
            }
            
            this->publishPoints(dynamicEigenPoints, this->rawDynamicPointsPub_);
        }
        catch (const pcl::PCLException& e) {
            RCLCPP_ERROR(this->get_logger(), "PCL Exception during dynamic point extraction: %s", e.what());
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Standard Exception during dynamic point extraction: %s", e.what());
        }
        catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown error during dynamic point extraction.");
        }
    }

    void dynamicDetector::transformBBox(const Eigen::Vector3d& center, const Eigen::Vector3d& size, const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation,
                                               Eigen::Vector3d& newCenter, Eigen::Vector3d& newSize){
        double x = center(0); 
        double y = center(1);
        double z = center(2);
        double xWidth = size(0);
        double yWidth = size(1);
        double zWidth = size(2);

        // get 8 bouding boxes coordinates in the camera frame
        Eigen::Vector3d p1 (x+xWidth/2.0, y+yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p2 (x+xWidth/2.0, y+yWidth/2.0, z-zWidth/2.0);
        Eigen::Vector3d p3 (x+xWidth/2.0, y-yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p4 (x+xWidth/2.0, y-yWidth/2.0, z-zWidth/2.0);
        Eigen::Vector3d p5 (x-xWidth/2.0, y+yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p6 (x-xWidth/2.0, y+yWidth/2.0, z-zWidth/2.0);
        Eigen::Vector3d p7 (x-xWidth/2.0, y-yWidth/2.0, z+zWidth/2.0);
        Eigen::Vector3d p8 (x-xWidth/2.0, y-yWidth/2.0, z-zWidth/2.0);

        // transform 8 points to the map coordinate frame
        Eigen::Vector3d p1m = orientation * p1 + position;
        Eigen::Vector3d p2m = orientation * p2 + position;
        Eigen::Vector3d p3m = orientation * p3 + position;
        Eigen::Vector3d p4m = orientation * p4 + position;
        Eigen::Vector3d p5m = orientation * p5 + position;
        Eigen::Vector3d p6m = orientation * p6 + position;
        Eigen::Vector3d p7m = orientation * p7 + position;
        Eigen::Vector3d p8m = orientation * p8 + position;
        std::vector<Eigen::Vector3d> pointsMap {p1m, p2m, p3m, p4m, p5m, p6m, p7m, p8m};

        // find max min in x, y, z directions
        double xmin=p1m(0); double xmax=p1m(0); 
        double ymin=p1m(1); double ymax=p1m(1);
        double zmin=p1m(2); double zmax=p1m(2);
        for (Eigen::Vector3d pm : pointsMap){
            if (pm(0) < xmin){xmin = pm(0);}
            if (pm(0) > xmax){xmax = pm(0);}
            if (pm(1) < ymin){ymin = pm(1);}
            if (pm(1) > ymax){ymax = pm(1);}
            if (pm(2) < zmin){zmin = pm(2);}
            if (pm(2) > zmax){zmax = pm(2);}
        }
        newCenter(0) = (xmin + xmax)/2.0;
        newCenter(1) = (ymin + ymax)/2.0;
        newCenter(2) = (zmin + zmax)/2.0;
        newSize(0) = xmax - xmin;
        newSize(1) = ymax - ymin;
        newSize(2) = zmax - zmin;
    }

    int dynamicDetector::getBestOverlapBBox(const onboardDetector::box3D& currBBox, const std::vector<onboardDetector::box3D>& targetBBoxes, double& bestIOU){
        bestIOU = 0.0;
        int bestIOUIdx = -1; // no match
        for (size_t i=0; i<targetBBoxes.size(); ++i){
            onboardDetector::box3D targetBBox = targetBBoxes[i];
            double IOU = this->calBoxIOU(currBBox, targetBBox);
            if (IOU > bestIOU){
                bestIOU = IOU;
                bestIOUIdx = i;
            }
        }
        return bestIOUIdx;
    }

    // user functions
    void dynamicDetector::getDynamicObstacles(std::vector<onboardDetector::box3D>& incomeDynamicBBoxes, const Eigen::Vector3d &robotSize){
        incomeDynamicBBoxes.clear();
        for (int i=0; i<int(this->dynamicBBoxes_.size()); i++){
            onboardDetector::box3D box = this->dynamicBBoxes_[i];
            box.x_width += robotSize(0);
            box.y_width += robotSize(1);
            box.z_width += robotSize(2);
            incomeDynamicBBoxes.push_back(box);
        }
    }

    void dynamicDetector::getDynamicObstaclesHist(std::vector<std::vector<Eigen::Vector3d>>& posHist, std::vector<std::vector<Eigen::Vector3d>>& velHist, std::vector<std::vector<Eigen::Vector3d>>& sizeHist, const Eigen::Vector3d &robotSize){
		posHist.clear();
        velHist.clear();
        sizeHist.clear();

        if (this->boxHist_.size()){
            for (size_t i=0 ; i<this->boxHist_.size() ; ++i){
                if (this->boxHist_[i][0].is_dynamic or this->boxHist_[i][0].is_human){   
                    bool findMatch = false;     
                    if (this->constrainSize_){
                        for (Eigen::Vector3d targetSize : this->targetObjectSize_){
                            double xdiff = std::abs(this->boxHist_[i][0].x_width - targetSize(0));
                            double ydiff = std::abs(this->boxHist_[i][0].y_width - targetSize(1));
                            double zdiff = std::abs(this->boxHist_[i][0].z_width - targetSize(2)); 
                            if (xdiff < 0.8 and ydiff < 0.8 and zdiff < 1.0){
                                findMatch = true;
                            }
                        }
                    }
                    else{
                        findMatch = true;
                    }
                    if (findMatch){
                        std::vector<Eigen::Vector3d> obPosHist, obVelHist, obSizeHist;
                        for (size_t j=0; j<this->boxHist_[i].size() ; ++j){
                            Eigen::Vector3d pos(this->boxHist_[i][j].x, this->boxHist_[i][j].y, this->boxHist_[i][j].z);
                            Eigen::Vector3d vel(this->boxHist_[i][j].Vx, this->boxHist_[i][j].Vy, 0);
                            Eigen::Vector3d size(this->boxHist_[i][j].x_width, this->boxHist_[i][j].y_width, this->boxHist_[i][j].z_width);
                            size += robotSize;
                            obPosHist.push_back(pos);
                            obVelHist.push_back(vel);
                            obSizeHist.push_back(size);
                        }
                        posHist.push_back(obPosHist);
                        velHist.push_back(obVelHist);
                        sizeHist.push_back(obSizeHist);
                    }
                }
            }
        }
	}
}
