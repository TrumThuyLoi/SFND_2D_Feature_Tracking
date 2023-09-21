/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>
// #include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;
namespace std_fs = std::experimental::filesystem;

void processImgs(std::string& imgBasePath, std::string& imgPrefix, std::string& imgFileType, std::string& detectorType,
                std::string& descriptorType, std::string& matcherType, std::string& descriptorCategory, std::string& selectorType,
                std_fs::path& result_dir, int imgStartIndex=0, int imgEndIndex=9, int imgFillWidth=4)
{
    std::cout << "Combinations of detectorType:"<< detectorType <<  " and descriptorType:" << descriptorType << "\n";
    // misc
    int dataBufferSize = 2;             // no. of images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer;   // list of data frames which are held in memory at the same time
    bool bVis = false;                  // visualize results
    bool is_logging = true;
    double time_it_takes = 0.0;

    std::string log_time = detectorType + " + " + descriptorType + ",";
    std::string log_detected_keypoints = detectorType + " + " + descriptorType + ",";
    std::string log_matched_keypoints = detectorType + " + " + descriptorType + ",";
    std::string log_keypoints_inside_rect = detectorType + " + " + descriptorType + ",";
    std::string log_info = "";

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        std::cout << "Process img: " << imgFullFilename << "\n";

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if(dataBuffer.size() >= dataBufferSize)
        {
            dataBuffer.pop_front();
        }
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            time_it_takes += detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            time_it_takes += detKeypointsHarris(keypoints, imgGray, false);
        }
        else
        {
            time_it_takes += detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        //// EOF STUDENT ASSIGNMENT

        log_detected_keypoints += std::to_string(keypoints.size()) + ",";

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            std::vector<cv::KeyPoint> inside_box;
            std::vector<float> neighbor_sizes;
            inside_box.reserve(keypoints.size());
            for(cv::KeyPoint& kp : keypoints)
            {
                if(vehicleRect.contains(kp.pt) == false)
                    continue;
                
                inside_box.emplace_back(kp);
                neighbor_sizes.emplace_back(kp.size);
            }
            keypoints = std::move(inside_box);
            
            log_keypoints_inside_rect += std::to_string(keypoints.size()) + ",";

            std::fstream fsKeypoints;
            if(is_logging)
            {
                fsKeypoints.open(result_dir.string() + "/" + detectorType + "-" + descriptorType + "-keypoints.txt", std::ios::app);
                log_info += "Neighborbood Sizes: \n";
                for(float& neighbor_size : neighbor_sizes)
                {
                    log_info += std::to_string(neighbor_size) + " ";
                }
                log_info += "\n";
                fsKeypoints.write(log_info.c_str(), log_info.size());
                fsKeypoints.close();
            }
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = true;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            std::cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer.back().keypoints = keypoints;
        std::cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        time_it_takes += descKeypoints(dataBuffer.back().keypoints, dataBuffer.back().cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        dataBuffer.back().descriptors = descriptors;

        //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, dataBuffer.back().keypoints,
                             (dataBuffer.end() - 2)->descriptors, dataBuffer.back().descriptors,
                             matches, descriptorCategory, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            log_matched_keypoints += std::to_string(matches.size()) + ",";

            // store matches in current data frame
            dataBuffer.back().kptMatches = matches;

            std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(500); // wait for key to be pressed
            }
            bVis = false;
        }
    } // eof loop over all images

    log_time += std::to_string(time_it_takes) + "\n";
    log_detected_keypoints += "\n";
    log_keypoints_inside_rect += "\n";
    log_matched_keypoints += "\n";

    std::fstream sheet_time;
    std::fstream sheet_detected_keypoints;
    std::fstream sheet_keypoints_inside_rect;
    std::fstream sheet_matched_keypoints;
    sheet_time.open(result_dir.string() + "/sheet_time.csv", std::ios::app);
    sheet_detected_keypoints.open(result_dir.string() + "/sheet_detected_keypoints.csv", std::ios::app);
    sheet_keypoints_inside_rect.open(result_dir.string() + "/sheet_keypoints_inside_rect.csv", std::ios::app);
    sheet_matched_keypoints.open(result_dir.string() + "/sheet_matched_keypoints.csv", std::ios::app);
    if(sheet_time.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_time.csv" << "\n";
    }
    if(sheet_detected_keypoints.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_detected_keypoints.csv" << "\n";
    }
    if(sheet_keypoints_inside_rect.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_keypoints_inside_rect.csv" << "\n";
    }
    if(sheet_matched_keypoints.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_matched_keypoints.csv" << "\n";
    }

    sheet_time.write(log_time.c_str(), log_time.size());
    sheet_detected_keypoints.write(log_detected_keypoints.c_str(), log_detected_keypoints.size());
    sheet_keypoints_inside_rect.write(log_keypoints_inside_rect.c_str(), log_keypoints_inside_rect.size());
    sheet_matched_keypoints.write(log_matched_keypoints.c_str(), log_matched_keypoints.size());

    sheet_time.close();
    sheet_detected_keypoints.close();
    sheet_keypoints_inside_rect.close();
    sheet_matched_keypoints.close();
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    std::vector<std::string> detectorTypes = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    std::vector<std::string> descriptorTypes = {"BRISK", "ORB", "FREAK", "AKAZE", "SIFT"};

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    string matcherType = "MAT_FLANN";           // MAT_BF, MAT_FLANN
    string descriptorCategory = "DES_BINARY";   // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";            // SEL_NN, SEL_KNN

    std_fs::path result_dir("../results");
    std::uintmax_t num_deleted = std_fs::remove_all(result_dir);
    if(num_deleted > 0)
    {
        std::cout << "Deleted " << num_deleted << " files or directories\n";
    }
    else std::cout << "Cannot delete file or directory\n";

    std_fs::create_directory(result_dir);

    std::string text;
    text.reserve(128);

    std::fstream sheet_time;
    std::fstream sheet_detected_keypoints;
    std::fstream sheet_keypoints_inside_rect;
    std::fstream sheet_matched_keypoints;
    sheet_time.open(result_dir.string() + "/sheet_time.csv", std::ios::app);
    sheet_detected_keypoints.open(result_dir.string() + "/sheet_detected_keypoints.csv", std::ios::app);
    sheet_keypoints_inside_rect.open(result_dir.string() + "/sheet_keypoints_inside_rect.csv", std::ios::app);
    sheet_matched_keypoints.open(result_dir.string() + "/sheet_matched_keypoints.csv", std::ios::app);
    if(sheet_time.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_time.csv" << "\n";
    }
    if(sheet_detected_keypoints.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_detected_keypoints.csv" << "\n";
    }
    if(sheet_keypoints_inside_rect.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_keypoints_inside_rect.csv" << "\n";
    }
    if(sheet_matched_keypoints.is_open() == false)
    {
        std::cout << "Cannot open " << result_dir << "/sheet_matched_keypoints.csv" << "\n";
    }

    text = "Combination,Time_it_take(ms)\n";
    sheet_time.write(text.c_str(), text.size());
    
    std_fs::path imgs_dir("../images/KITTI/2011_09_26/image_00/data");
    std::vector<std::string> list_dirs;
    for(auto it_dir : std_fs::directory_iterator(imgs_dir))
    {
        list_dirs.emplace_back(it_dir.path().filename().string());
    }

    text = "Combination,";
    for(auto it_dir_name = list_dirs.begin(); it_dir_name != list_dirs.end()-1; it_dir_name++)
    {
        auto it_next = std::next(it_dir_name);
        text += *it_dir_name + "->" + *it_next;
    }
    text += "\n";

    sheet_detected_keypoints.write(text.c_str(), text.size());
    sheet_keypoints_inside_rect.write(text.c_str(), text.size());
    sheet_matched_keypoints.write(text.c_str(), text.size());

    sheet_time.close();
    sheet_detected_keypoints.close();
    sheet_keypoints_inside_rect.close();
    sheet_matched_keypoints.close();

    for(std::string& detectorType : detectorTypes)
    {
        for(std::string& descriptorType : descriptorTypes)
        {
            text = "";
            try
            {
                processImgs(imgBasePath, imgPrefix, imgFileType, detectorType, descriptorType, matcherType, 
                            descriptorCategory, selectorType, result_dir, imgStartIndex, imgEndIndex, imgFillWidth);
            }
            catch(...)
            {
                std::cout << "Default Exception!\n";

                text += detectorType + " + " + descriptorType + ",Not combinable\n";
                
                sheet_time.open(result_dir.string() + "/sheet_time.csv", std::ios::app);
                sheet_detected_keypoints.open(result_dir.string() + "/sheet_detected_keypoints.csv", std::ios::app);
                sheet_keypoints_inside_rect.open(result_dir.string() + "/sheet_keypoints_inside_rect.csv", std::ios::app);
                sheet_matched_keypoints.open(result_dir.string() + "/sheet_matched_keypoints.csv", std::ios::app);
                
                sheet_time.write(text.c_str(), text.size());
                sheet_detected_keypoints.write(text.c_str(), text.size());
                sheet_keypoints_inside_rect.write(text.c_str(), text.size());
                sheet_matched_keypoints.write(text.c_str(), text.size());

                sheet_time.close();
                sheet_detected_keypoints.close();
                sheet_keypoints_inside_rect.close();
                sheet_matched_keypoints.close();
            }
        }
    }

    return 0;
}