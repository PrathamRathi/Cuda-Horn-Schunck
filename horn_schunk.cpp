#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

// Compute spatial derivatives
void computeDerivatives(const cv::Mat& frame1, const cv::Mat& frame2, 
                         cv::Mat& Ix, cv::Mat& Iy, cv::Mat& It) {
    cv::Mat gray1, gray2;
    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
    
    // Compute image derivatives
    cv::Sobel(gray1, Ix, CV_64F, 1, 0, 3);
    cv::Sobel(gray1, Iy, CV_64F, 0, 1, 3);
    
    // Compute temporal derivative
    It = gray2 - gray1;
}

// Compute average flow from neighboring pixels using direct indexing
void computeNeighborAverage(const cv::Mat& u, const cv::Mat& v, 
                             cv::Mat& uAvg, cv::Mat& vAvg) {
    uAvg = cv::Mat::zeros(u.size(), CV_64F);
    vAvg = cv::Mat::zeros(v.size(), CV_64F);

    for (int y = 1; y < u.rows - 1; ++y) {
        for (int x = 1; x < u.cols - 1; ++x) {
            // Directly compute average from 8 neighboring pixels
            uAvg.at<double>(y, x) = (
                u.at<double>(y-1, x-1) + u.at<double>(y-1, x) + u.at<double>(y-1, x+1) +
                u.at<double>(y, x-1) + u.at<double>(y, x+1) +
                u.at<double>(y+1, x-1) + u.at<double>(y+1, x) + u.at<double>(y+1, x+1)
            ) / 8.0;

            vAvg.at<double>(y, x) = (
                v.at<double>(y-1, x-1) + v.at<double>(y-1, x) + v.at<double>(y-1, x+1) +
                v.at<double>(y, x-1) + v.at<double>(y, x+1) +
                v.at<double>(y+1, x-1) + v.at<double>(y+1, x) + v.at<double>(y+1, x+1)
            ) / 8.0;
        }
    }
}

// Compute optical flow using Horn-Schunck method
void computeOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2, 
                         cv::Mat& flowX, cv::Mat& flowY, 
                         double alpha = 1.0, int iterations = 100) {
    // Compute image derivatives
    cv::Mat Ix, Iy, It;
    computeDerivatives(frame1, frame2, Ix, Iy, It);

    // Initialize flow fields
    flowX = cv::Mat::zeros(frame1.size(), CV_64F);
    flowY = cv::Mat::zeros(frame1.size(), CV_64F);

    // Temporary matrices for averaging
    cv::Mat uAvg, vAvg;

    // Iterative optimization
    for (int iter = 0; iter < iterations; ++iter) {
        // Compute average flow from neighboring pixels
        computeNeighborAverage(flowX, flowY, uAvg, vAvg);

        // Update flow for each pixel
        for (int y = 1; y < frame1.rows - 1; ++y) {
            for (int x = 1; x < frame1.cols - 1; ++x) {
                // Compute local variables
                double ix = Ix.at<double>(y, x);
                double iy = Iy.at<double>(y, x);
                double it = It.at<double>(y, x);

                // Compute denominator
                double denom = alpha * alpha + ix * ix + iy * iy;

                // Update flow
                double u = uAvg.at<double>(y, x) - 
                    (ix * (ix * uAvg.at<double>(y, x) + 
                           iy * vAvg.at<double>(y, x) + it)) / denom;
                
                double v = vAvg.at<double>(y, x) - 
                    (iy * (ix * uAvg.at<double>(y, x) + 
                           iy * vAvg.at<double>(y, x) + it)) / denom;

                // Update flow fields
                flowX.at<double>(y, x) = u;
                flowY.at<double>(y, x) = v;
            }
        }
    }
}

// Visualize optical flow
cv::Mat visualizeFlow(const cv::Mat& flowX, const cv::Mat& flowY) {
    cv::Mat flowImage(flowX.size(), CV_8UC3);
    
    for (int y = 0; y < flowX.rows; ++y) {
        for (int x = 0; x < flowX.cols; ++x) {
            // Compute flow magnitude and angle
            double fx = flowX.at<double>(y, x);
            double fy = flowY.at<double>(y, x);
            
            // Map flow to HSV color representation
            double magnitude = std::sqrt(fx * fx + fy * fy);
            double angle = std::atan2(fy, fx) * 180.0 / CV_PI;
            
            // Normalize angle to [0, 180]
            angle = std::max(0.0, std::min(angle + 180.0, 179.0));
            
            // Create HSV representation
            flowImage.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(angle),
                255,
                static_cast<uchar>(std::min(magnitude * 10, 255.0))
            );
        }
    }
    
    // Convert HSV to BGR for visualization
    cv::Mat bgrFlow;
    cv::cvtColor(flowImage, bgrFlow, cv::COLOR_HSV2BGR);
    return bgrFlow;
}

// Write flow to a CSV file
void writeFlowToCSV(const cv::Mat& flowX, const cv::Mat& flowY, const std::string& filename) {
    std::ofstream csvFile(filename);
    
    if (!csvFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header
    csvFile << "x,y,flow_x,flow_y,magnitude,angle\n";

    for (int y = 0; y < flowX.rows; ++y) {
        for (int x = 0; x < flowX.cols; ++x) {
            // Compute flow components
            double fx = flowX.at<double>(y, x);
            double fy = flowY.at<double>(y, x);
            
            // Compute magnitude and angle
            double magnitude = std::sqrt(fx * fx + fy * fy);
            double angle = std::atan2(fy, fx) * 180.0 / CV_PI;
            
            // Write to CSV
            csvFile << x << "," 
                    << y << "," 
                    << fx << "," 
                    << fy << "," 
                    << magnitude << "," 
                    << angle << "\n";
        }
    }

    csvFile.close();
    std::cout << "Flow data written to " << filename << std::endl;
}

// Main function demonstrating usage
int main() {
    std::cout<<"running"<<std::endl;
    // Load two consecutive frames
    cv::Mat frame1 = cv::imread("car1.jpg");
    cv::Mat frame2 = cv::imread("car2.jpg");
    
    if (frame1.empty() || frame2.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }
    
    // Compute optical flow
    cv::Mat flowX, flowY;
    computeOpticalFlow(frame1, frame2, flowX, flowY);
    
    // Visualize flow
    cv::Mat flowVisualization = visualizeFlow(flowX, flowY);
    
    // Write flow data to CSV
    writeFlowToCSV(flowX, flowY, "optical_flow.csv");
    
    // Display results
    cv::imshow("Original Frame 1", frame1);
    cv::imshow("Original Frame 2", frame2);
    cv::imshow("Optical Flow", flowVisualization);
    
    // Save visualization
    cv::imwrite("optical_flow_visualization.png", flowVisualization);
    
    cv::waitKey(0);
    
    return 0;
}