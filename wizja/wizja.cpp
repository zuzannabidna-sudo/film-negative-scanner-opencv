#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Functions declarations 
cv::Rect findTemplate(const cv::Mat& src, cv::RotatedRect& outExternalRect);
void initialCropAndGeometry(cv::Mat& film, const cv::RotatedRect& externalRect, const cv::Rect& firstBoundingBox, std::vector<int>& detectedVerticalX, std::vector<int>& detectedHorizontalY);
cv::Rect findInterframeCrop(const cv::Mat& film, const std::vector<int>& detectedVerticalX, const std::vector<int>& detectedHorizontalY);
cv::Mat convertToPositive(const cv::Mat& croppedFilm);

//////////////////////////////////////////////////////////////

int main() {
    std::vector<int> detectedVerticalX;
    std::vector<int> detectedHorizontalY;

    // Image path to be replaced manually later: bw1.jpg, bw2.jpg, bw3.jpg etc.
    std::string imagePath = "bw_negatyw\\bw1.jpg";
    cv::Mat film = cv::imread(imagePath);

    if (film.empty())
    {
        std::cerr << "Error: Could not load the image" << std::endl;
        return -1;
    }

    // STEP 1: Locate the external black frame/holder
    cv::RotatedRect externalRect;
    cv::Rect firstBoundingBox = findTemplate(film, externalRect);

    if (firstBoundingBox.width == 0 || firstBoundingBox.height == 0)
    {
        std::cerr << "Error: Template/frame not found" << std::endl;
        return -1;
    }

    // STEP 2: Initial crop and geometry alignment (rotation correction)
    initialCropAndGeometry(film, externalRect, firstBoundingBox, detectedVerticalX, detectedHorizontalY);

    cv::namedWindow("Step 1 and 2: Aligned film strip", cv::WINDOW_NORMAL);
    cv::imshow("Step 1 and 2: Aligned film strip", film);

    // STEP 3: Analyze strip brightness + determine 3:2 frame window using Hough lines
    cv::Rect finalPhotoBox = findInterframeCrop(film, detectedVerticalX, detectedHorizontalY);

    // Preview the cropped frame before inversion
    cv::Mat cropped = film(finalPhotoBox).clone();

    // STEP 4: Convert negative to positive and apply filtration
    cv::Mat finalPositive = convertToPositive(cropped);

    // Display the final image
    cv::namedWindow("Step 4: Grande Finale - Positive", cv::WINDOW_NORMAL);
    cv::imshow("Step 4: Grande Finale - Positive", finalPositive);

    // STEP 5: Automatic file save
    // Update the output name according to the loaded image
    std::string outputPath = "bw_negatyw\\bw1_pozytyw.jpg";
    if (cv::imwrite(outputPath, finalPositive))
    {
        std::cout << "Successfully saved developed photo: " << outputPath << std::endl;
    }
    else
    {
        std::cerr << "Failed to save the output file" << std::endl;
    }

    cv::waitKey(0);
    return 0;
}
///////////////////////////////////////////////////////

// Implemented functions 

cv::Mat convertToPositive(const cv::Mat& croppedFilm)
{
    const int medianBlurKernelSize = 3;
    const double claheClipLimit = 5.0;

    cv::Mat positiveGray;
    cv::cvtColor(croppedFilm, positiveGray, cv::COLOR_BGR2GRAY);    // Convert to grayscale

    cv::Mat inverted;
    cv::bitwise_not(positiveGray, inverted);    // Invert pixels

    cv::Mat normalized;
    cv::normalize(inverted, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);    // Normalize dynamic range

    cv::Mat cleaned;
    cv::medianBlur(normalized, cleaned, medianBlurKernelSize);     // Median blur to reduce noise and dust

    cv::Mat finalPositive;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();   // CLAHE 
    // Clip limit prevents over-amplification of noise and grain in dark areas
    clahe->setClipLimit(claheClipLimit);   // Modifying this affects the visual depth of the image
    clahe->apply(cleaned, finalPositive);

    return finalPositive;
}

cv::Rect findTemplate(const cv::Mat& src, cv::RotatedRect& outExternalRect)
{
    const int gaussianBlurSize = 5;
    const double binaryThresholdValue = 40.0;
    const double maxBinaryValue = 255.0;

    cv::Mat gray, thresh;
    // Noise reduction and binarization
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(gaussianBlurSize, gaussianBlurSize), 0);
    cv::threshold(gray, thresh, binaryThresholdValue, maxBinaryValue, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    // Find external contours, ignore everything inside the film area (RETR_EXTERNAL)
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Search for the largest contour which represents the film holder/frame
    double maxArea = 0;
    int maxIdx = -1;
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) { maxArea = area; maxIdx = i; }
    }

    if (maxIdx == -1) return cv::Rect();
    // Determine frame geometry
    outExternalRect = cv::minAreaRect(contours[maxIdx]);
    cv::Rect boundingBox = cv::boundingRect(contours[maxIdx]);

    return boundingBox & cv::Rect(0, 0, src.cols, src.rows);
}

void initialCropAndGeometry(cv::Mat& film, const cv::RotatedRect& externalRect, const cv::Rect& firstBoundingBox, std::vector<int>& detectedVerticalX, std::vector<int>& detectedHorizontalY)
{
    const double cannyThresh1 = 50.0;
    const double cannyThresh2 = 150.0;
    const int cannyAperture = 3;

    const int edgeOffset = 50;
    const double topBottomMarginRatio = 0.12;
    const double leftRightMarginRatio = 0.15;

    const double houghRho = 1.0;
    const double houghTheta = CV_PI / 180.0;
    const int houghThreshold = 40;
    const double houghMinLineLength = 50.0;
    const double houghMaxLineGap = 20.0;

    const double horizontalAngleThreshold = 15.0;
    const double horizontalAngleComplement = 165.0;
    const double verticalAngleMin = 75.0;
    const double verticalAngleMax = 105.0;

    const float angleMaxRotationLimit = 45.0f;
    const float angleRotationCorrection = 90.0f;

    film = film(firstBoundingBox).clone();

    cv::Mat gray, edges;
    cv::cvtColor(film, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, cannyThresh1, cannyThresh2, cannyAperture);

    // Cut off extreme outer edges from the previous crop step
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);

    int topBottomMargin = edges.rows * topBottomMarginRatio;
    int leftRightMargin = edges.cols * leftRightMarginRatio;

    // Define ROI zone masks
    cv::rectangle(mask, cv::Rect(edgeOffset, edgeOffset, edges.cols - (2 * edgeOffset), topBottomMargin - edgeOffset), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Rect(edgeOffset, edges.rows - topBottomMargin, edges.cols - (2 * edgeOffset), topBottomMargin - edgeOffset), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Rect(edgeOffset, edgeOffset, leftRightMargin - edgeOffset, edges.rows - (2 * edgeOffset)), cv::Scalar(255), -1);
    cv::rectangle(mask, cv::Rect(edges.cols - leftRightMargin, edgeOffset, leftRightMargin - edgeOffset, edges.rows - (2 * edgeOffset)), cv::Scalar(255), -1);

    cv::bitwise_and(edges, mask, edges);


    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, houghRho, houghTheta, houghThreshold, houghMinLineLength, houghMaxLineGap);

    double totalAngle = 0.0;
    int validLinesCount = 0;

    detectedVerticalX.clear();
    detectedHorizontalY.clear();

    cv::Mat houghDebug = film.clone();

    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = lines[i];
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];

        double angleRad = atan2(dy, dx);
        double angleDeg = angleRad * 180.0 / CV_PI;

        // Red lines -> top and bottom perforation edges
        if (abs(angleDeg) < horizontalAngleThreshold)
        {
            totalAngle += angleDeg;
            validLinesCount++;
            detectedHorizontalY.push_back((l[1] + l[3]) / 2); 
            cv::line(houghDebug, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2);
        }
        else if (abs(angleDeg) > horizontalAngleComplement)
        {
            totalAngle += (angleDeg > 0 ? angleDeg - 180.0 : angleDeg + 180.0);
            validLinesCount++;
            detectedHorizontalY.push_back((l[1] + l[3]) / 2);
            cv::line(houghDebug, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2);
        }

        // Blue lines -> vertical frame edges (left and right)
        if (abs(angleDeg) > verticalAngleMin && abs(angleDeg) < verticalAngleMax)
        {
            detectedVerticalX.push_back((l[0] + l[2]) / 2); 
            cv::line(houghDebug, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255, 0, 0), 2);
        }
    }

    // Display Hough lines detection diagnostics
    cv::namedWindow("Detected Hough Lines", cv::WINDOW_NORMAL);
    cv::imshow("Detected Hough Lines", houghDebug);

    float finalAngle = 0.0f;
    if (validLinesCount > 0)
    {
        finalAngle = static_cast<float>(totalAngle / validLinesCount);
        std::cout << "Precise rotation angle calculated from perforation lines: " << finalAngle << " degrees." << std::endl;
    }
    else
    {
        finalAngle = externalRect.angle;
        if (externalRect.size.width < externalRect.size.height) finalAngle += angleRotationCorrection;
        if (finalAngle > angleMaxRotationLimit) finalAngle -= angleRotationCorrection;
        if (finalAngle < -angleMaxRotationLimit) finalAngle += angleRotationCorrection;
        std::cout << "No Hough lines found on margins. Fallback angle from contour bounding box: " << finalAngle << " degrees." << std::endl;
    }

    cv::Point2f localCenter(externalRect.center.x - firstBoundingBox.x, externalRect.center.y - firstBoundingBox.y);
    cv::Mat rotMat = cv::getRotationMatrix2D(localCenter, finalAngle, 1.0);
    cv::warpAffine(film, film, rotMat, film.size(), cv::INTER_CUBIC);
}

cv::Rect findInterframeCrop(const cv::Mat& film, const std::vector<int>& detectedVerticalX, const std::vector<int>& detectedHorizontalY) 
{
    const double rowScanStartRatio = 0.25;
    const double rowScanEndRatio = 0.75;
    const double leftMarginBoundaryRatio = 0.02;
    const double rightMarginBoundaryRatio = 0.98;
    const double brightnessDropThreshold = 0.85;

    const double estimatedTopRatio = 0.13;
    const double estimatedBottomRatio = 0.87;
    const int tolerance = 40; // Tolerance margin in pixels for line matching

    const int paddingX = 20;       // Left and right expansion margin
    const int paddingTop = 80;     // Top expansion margin
    const int paddingBottom = 80;  // Bottom expansion margin
    const double targetAspectRatio = 1.5; // Constant aspect ratio for 3:2 format

    cv::Mat gray;
    cv::cvtColor(film, gray, cv::COLOR_BGR2GRAY);

    // Brightness profile analysis along the X axis
    // Ignore top and bottom sprocket holes/perforations during vertical summation
    int startRow = film.rows * rowScanStartRatio;
    int endRow = film.rows * rowScanEndRatio;
    std::vector<long long> colBrightness(film.cols, 0);

    // Sum intensity values across columns
    for (int col = 0; col < film.cols; col++)
    {
        for (int row = startRow; row < endRow; row++)
        {
            colBrightness[col] += gray.at<uchar>(row, col);
        }
    }

    // Locate the center of the left frame separator -> overexposed space between frames on the left side
    int leftSeparatorX = 0;
    long long maxLeftVal = 0;
    for (int col = static_cast<int>(film.cols * leftMarginBoundaryRatio); col < film.cols / 2; col++)
    {
        if (colBrightness[col] > maxLeftVal)
        {
            maxLeftVal = colBrightness[col];
            leftSeparatorX = col;
        }
    }

    // Locate the center of the right frame separator
    int rightSeparatorX = film.cols - 1;
    long long maxRightVal = 0;
    for (int col = film.cols / 2; col < static_cast<int>(film.cols * rightMarginBoundaryRatio); col++)
    {
        if (colBrightness[col] > maxRightVal)
        {
            maxRightVal = colBrightness[col];
            rightSeparatorX = col;
        }
    }

    // Analyze brightness drop to 85% -> looking for where the actual exposed frame starts
    int photoLeftEdge = leftSeparatorX;
    long long thresholdLeft = maxLeftVal * brightnessDropThreshold;
    while (photoLeftEdge < film.cols / 2 && colBrightness[photoLeftEdge] > thresholdLeft)
    {
        photoLeftEdge++;
    }

    int photoRightEdge = rightSeparatorX;
    long long thresholdRight = maxRightVal * brightnessDropThreshold;
    while (photoRightEdge > film.cols / 2 && colBrightness[photoRightEdge] > thresholdRight)
    {
        photoRightEdge--;
    }

    // Edge estimation via Hough line data snap
    int optLeft = photoLeftEdge;
    int optRight = photoRightEdge;

    // Estimated position of horizontal boundaries (top/bottom)
    int optTop = static_cast<int>(film.rows * estimatedTopRatio);
    int optBottom = static_cast<int>(film.rows * estimatedBottomRatio);

    // Snap boundaries to previously detected vertical Hough lines (X-axis)
    for (int x : detectedVerticalX)
    {
        if (abs(x - photoLeftEdge) < tolerance)  optLeft = x;
        if (abs(x - photoRightEdge) < tolerance) optRight = x;
    }

    // Snap boundaries to horizontal Hough lines (Y-axis)
    for (int y : detectedHorizontalY)
    {
        if (abs(y - (film.rows * estimatedTopRatio)) < tolerance) optTop = y;
        if (abs(y - (film.rows * estimatedBottomRatio)) < tolerance) optBottom = y;
    }

    // Margin safety padding -> safe buffer expansion to prevent cutting off the original image content
    int finalLeft = optLeft - paddingX;
    int finalRight = optRight + paddingX;
    int finalTop = optTop - paddingTop;
    int finalBottom = optBottom + paddingBottom;

    // Calculate crop width and enforce standard 3:2 aspect ratio geometry
    int calculatedWidth = finalRight - finalLeft;
    int calculatedHeight = static_cast<int>(calculatedWidth / targetAspectRatio);

    // Center the target crop window vertically according to corrected baseline bounds
    int actualCenterY = (finalTop + finalBottom) / 2;
    int cropY = actualCenterY - (calculatedHeight / 2);

    // Define final ROI Box
    cv::Rect dynamicPhotoBox(
        finalLeft,
        cropY,
        calculatedWidth,
        calculatedHeight
    );

    cv::Mat debugImg = film.clone();
    cv::line(debugImg, cv::Point(optLeft, 0), cv::Point(optLeft, film.rows), cv::Scalar(0, 255, 0), 2);
    cv::line(debugImg, cv::Point(optRight, 0), cv::Point(optRight, film.rows), cv::Scalar(0, 255, 0), 2);
    cv::line(debugImg, cv::Point(0, optTop), cv::Point(film.cols, optTop), cv::Scalar(255, 255, 0), 1);
    cv::line(debugImg, cv::Point(0, optBottom), cv::Point(film.cols, optBottom), cv::Scalar(255, 255, 0), 1);
    // Preview detected film strips/boundaries
    cv::namedWindow("Detected Film Strip Edges", cv::WINDOW_NORMAL);
    cv::imshow("Detected Film Strip Edges", debugImg);

    std::cout << "Hough + padding estimation -> Crop Width: " << calculatedWidth << "px, Height: " << calculatedHeight << "px." << std::endl;

    return dynamicPhotoBox & cv::Rect(0, 0, film.cols, film.rows);
}