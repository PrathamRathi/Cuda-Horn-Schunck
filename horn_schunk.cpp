#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace std;

// Write flow to a CSV file
void writeFlowToCSV(const cv::Mat& flowX, const cv::Mat& flowY, const std::string& filename, const cv::Mat& Ix) {
    std::ofstream csvFile(filename);
   
    if (!csvFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header
    csvFile << "x,y,flow_x,flow_y,magnitude,angle, ix\n";

    for (int y = 0; y < flowX.rows; ++y) {
        for (int x = 0; x < flowX.cols; ++x) {
            // Compute flow components
            double fx = flowX.at<double>(y, x);
            double fy = flowY.at<double>(y, x);
           
            // Compute magnitude and angle
            double magnitude = std::sqrt(fx * fx + fy * fy);
            double angle = std::atan2(fy, fx) * 180.0 / CV_PI;
            double ix = Ix.at<double>(y, x);
            // Write to CSV
            csvFile << x << ","
                    << y << ","
                    << fx << ","
                    << fy << ","
                    << magnitude << ","
                    << angle << ","
                    << ix << "\n";
        }
    }

    csvFile.close();
    std::cout << "Flow data written to " << filename << std::endl;
}


void computeDerivatives(const Mat& im1, const Mat& im2, Mat& ix, Mat& iy, Mat& it) {
    // Define kernels for calculating derivatives
    Mat kernelX = (Mat_<double>(2, 2) << 0.25, -0.25, 0.25, -0.25); // Kernel for dx
    Mat kernelY = (Mat_<double>(2, 2) << 0.25, 0.25, -.25, -.25); // Kernel for dy
    Mat kernelT = (Mat_<double>(2, 2) << 0.25, 0.25, 0.25, 0.25);   // Kernel for dt

    // Convert images to double precision
    Mat im1_d, im2_d;
    im1.convertTo(im1_d, CV_64FC1);
    im2.convertTo(im2_d, CV_64FC1);

    // Compute derivatives
    Mat fx1, fx2, fy1, fy2, ft1, ft2;
    filter2D(im1_d, fx1, -1, kernelX);
    filter2D(im2_d, fx2, -1, kernelX);
    ix = fx1 + fx2;

    filter2D(im1_d, fy1, -1, kernelY);
    filter2D(im2_d, fy2, -1, kernelY);
    iy = fy1 + fy2;

    filter2D(im2_d, ft1, -1, -kernelT);
    filter2D(im1_d, ft2, -1, kernelT);
    it = ft1 + ft2;
}

// Compute average flow from neighboring pixels using direct indexing
// void computeNeighborAverage(const cv::Mat& u, const cv::Mat& v,
//                              cv::Mat& uAvg, cv::Mat& vAvg) {
//     uAvg = cv::Mat::zeros(u.size(), CV_64F);
//     vAvg = cv::Mat::zeros(v.size(), CV_64F);

//     for (int y = 1; y < u.rows - 1; ++y) {
//         for (int x = 1; x < u.cols - 1; ++x) {
//             // Directly compute average from 8 neighboring pixels
//             uAvg.at<double>(y, x) = (
//                 u.at<double>(y-1, x-1)/12 + u.at<double>(y-1, x)/6 + u.at<double>(y-1, x+1)/12 +
//                 u.at<double>(y, x-1)/6 + u.at<double>(y, x+1)/6 +
//                 u.at<double>(y+1, x-1)/12 + u.at<double>(y+1, x)/6 + u.at<double>(y+1, x+1)/12
//             );

//             vAvg.at<double>(y, x) = (
//                 v.at<double>(y-1, x-1)/12 + v.at<double>(y-1, x)/6 + v.at<double>(y-1, x+1)/12 +
//                 v.at<double>(y, x-1)/6 + v.at<double>(y, x+1)/6 +
//                 v.at<double>(y+1, x-1)/12 + v.at<double>(y+1, x)/6 + v.at<double>(y+1, x+1)/12
//             );
//         }
//     }
// }
void computeNeighborAverage(const Mat& u, const Mat& v, Mat& uAvg, Mat& vAvg) {
    // Define the kernel for the weighted average
    Mat kernel = (Mat_<double>(3, 3) << 
        1.0 / 12, 1.0 / 6, 1.0 / 12,
        1.0 / 6,  0.0,      1.0 / 6,
        1.0 / 12, 1.0 / 6, 1.0 / 12);

    // Apply convolution using filter2D
    filter2D(u, uAvg, -1, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
    filter2D(v, vAvg, -1, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
}
// Compute optical flow using Horn-Schunck method
void computeOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2,
                         cv::Mat& flowX, cv::Mat& flowY,
                         double alpha = 0.01, int iterations = 20) {
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
        cout << "iter " << iter << endl;
        // Update flow for each pixel
        for (int y = 0; y < frame1.rows; ++y) {
            for (int x = 0; x < frame1.cols; ++x) {
                double ix = Ix.at<double>(y, x);
                double iy = Iy.at<double>(y, x);
                double it = It.at<double>(y, x);

                // Add small epsilon to prevent division by zero
                double denom = alpha * alpha + ix * ix + iy * iy;

                double uAvgVal = uAvg.at<double>(y,x);
                double vAvgVal = vAvg.at<double>(y,x);
                double p = (ix * uAvgVal + iy * vAvgVal + it);
                double u = uAvgVal - ix * (p/denom);
                double v = vAvgVal - iy * (p/denom);
                flowX.at<double>(y,x) = u;
                flowY.at<double>(y,x) = v;
            }
        }
    }
    writeFlowToCSV(flowX, flowY, "optical_flow.csv", Ix); //
}

// Visualize optical flow
void drawOpticalFlow(const Mat& flowU, const Mat& flowV, Mat& image, int scale = 3, int step = 16) {
    for (int y = 0; y < image.rows; y += step) {
        for (int x = 0; x < image.cols; x += step) {
            cout << flowU.at<double>(y, x) << " " << flowV.at<double>(y, x) << endl;
            Point2f flow(flowU.at<double>(y, x), flowV.at<double>(y, x));
            Point start(x, y);
            Point end(cvRound(x + flow.x * scale), cvRound(y + flow.y * scale));
            arrowedLine(image, start, end, Scalar(0, 255, 0), 1, LINE_AA, 0, 0.2);
        }
    }
}

// Main function demonstrating usage
int main() {
    std::cout << "Running Horn-Schunck optical flow..." << std::endl;
   
    std::string filename1 = "frame1.png";
    std::string filename2 = "frame2.png";

    // std::string filename1 = "car1.jpg";
    // std::string filename2 = "car2.jpg";

    // Load two consecutive frames
    cv::Mat frame1 = cv::imread(filename1, 0);
    cv::Mat frame2 = cv::imread(filename2, 0);
   
    if (frame1.empty() || frame2.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        std::cerr << "Make sure car1.jpg and car2.jpg exist in: " << std::filesystem::current_path() << std::endl;
        return -1;
    }
   
    std::cout << "Loaded images - Frame1: " << frame1.size() << " Frame2: " << frame2.size() << std::endl;
   
    // Compute optical flow
    cv::Mat flowX, flowY;
    computeOpticalFlow(frame1, frame2, flowX, flowY, 1, 100);

     // Visualize optical flow
    Mat img_color;
    cvtColor(frame1, img_color, COLOR_GRAY2BGR);
    
    drawOpticalFlow(flowX, flowY, img_color);
    
    // Write flow data to CSV
    

    imshow("Optical Flow", img_color);
   cv::waitKey(0);
    // Save visualization
    cv::imwrite("optical_flow_visualization.png", img_color);
   
    cv::waitKey(0);
   
    return 0;
}