# 35mm B&W Film Negative Scanner & Processing Pipeline

**Author:** Zuzanna Bidna 

An automatic computer vision pipeline written in **C++** using **OpenCV** designed to locate, deskew, crop, and digitally "develop" 35mm black-and-white film negatives housed in a black scanner film holder.

---

## Project Assumptions & Hardware Setup

This project addresses the challenge of manual, budget-friendly film digitization without the need for expensive dedicated film scanners or precise laboratory copy stands.

### Hardware Configuration:
* **Sensor (Camera):** Pentax K-r (APS-C CMOS 12.9 Mpix sensor, $4288 \times 2848$ px)
* **Light Source:** LCD tablet screen displaying a solid white background
* **Diffuser:** Matte polymer film (applied to the tablet to disperse the light and eliminate Moire patterns)
* **Target:** 35mm black-and-white film strips (standard frame size of $36 \times 24$ mm) placed inside a physical black scanner guide

### Programmatic Error Compensation:
Because the film strips are photographed manually, each shot introduces spatial and geometric inconsistencies. The algorithm was specifically designed to programmatically compensate for these real-world acquisition errors:
1. **Rotation Error** – The film strip lies at a slightly different angle relative to the camera sensor in each shot.
2. **Scale Error** – The distance between the camera and the film varies between shots, meaning the same film strip occupies a different number of pixels each time.
3. **Backlight Inhomogeneity** – The tablet screen introduces non-uniform illumination (brighter center, darker edges).

---

## Algorithmic Pipeline

The computer vision pipeline is divided into four main processing steps:

### 1. Film Strip Location 
* Conversion to grayscale (`cv::cvtColor`) and noise reduction using a $5\times5$ Gaussian blur (`cv::GaussianBlur`).
* Binary thresholding (`cv::threshold`) to separate the dark film holder frame from the bright tablet background.
* External contour analysis (`cv::findContours` with `RETR_EXTERNAL`) to isolate the largest object (the holder) and establish the initial bounding box (`cv::boundingRect`).

### 2. Geometry Correction & Deskewing 
* Edge detection using the Canny operator (`cv::Canny`).
* Zone masking (`cv::bitwise_and`) to exclude outer boundary edges that could distort the Hough transform.
* Line detection via the **Probabilistic Hough Transform** (`cv::HoughLinesP`) followed by line classification (horizontal vs. vertical) using `atan2()`.
* Affine transformation (`cv::getRotationMatrix2D` + `cv::warpAffine` with cubic interpolation `cv::INTER_CUBIC`) to precisely deskew the film strip horizontally. If no lines are detected, a fallback mechanism estimates the angle directly from the external holder geometry (`externalRect.angle`).

### 3. Hybrid 3:2 Frame Cropping 
* **Luminance Profiling:** The unexposed gaps between frames let the most light through, creating local maxima in the projected brightness profile.
* **Rough Edge Detection:** Initial frame boundary estimation based on an 85% luminance drop-off threshold.
* **Hough Line Snapping:** The mathematically calculated boundaries are automatically "snapped" to actual structural lines detected by the Hough transform within a 40-pixel tolerance window.
* **Aspect Ratio Enforcement:** To maintain the native **1.5 (3:2)** aspect ratio, the target height is calculated directly from the adjusted width and centered vertically.
* A subtle, safe margin is preserved around the cropped frame to ensure no critical edge details are lost.

### 4. Digital Development & Tonal Optimization 
* Bitwise inversion (`cv::bitwise_not`) to convert the negative image into a positive.
* Global histogram normalization (`cv::normalize` with `NORM_MINMAX`) to stretch the dynamic range, establishing true black and white points while neutralizing exposure errors.
* Median filtering with a $3\times3$ kernel (`cv::medianBlur`) to suppress minor dust and scratches.
* Local contrast enhancement using **CLAHE** (`cv::createCLAHE` with a clip limit of 3.5) to bring out hidden details in deep shadows and highlights.

---

## Experimental Verification & Performance

* **Theoretical Resolution:** The sensor's maximum sampling density yields ~**3000 DPI** ($4288 \times 2848$ px over a $36 \times 24$ mm frame area).
* **Real-world Resolution:** Due to safe-crop margins and camera-distance fluctuations, the finalized positive images measure between **$1895 \times 1263$ px** and **$2550 \times 1700$ px**.
* This translates to a real scanning resolution of **1300 to 1800 DPI**, which is more than sufficient for high-quality home archiving and print reproductions up to A4 size.

### Limitations & Future Scope:
* **Extreme Overexposure:** On heavily overexposed negatives, the lack of contrast can cause the algorithm to mistake technical film edge markings for the frame border, resulting in an overly tight crop.
* **Deep Learning Integration:** Classical, rule-based thresholding is inherently sensitive to emulsion damage. Integrating Machine Learning (specifically convolutional networks for object detection) to locate frame boundaries would dramatically increase robustness against extreme exposure fluctuations.

---

## How to Run

### System Requirements:
* IDE: **Visual Studio 2022** (or newer)
* Library: **OpenCV** (recommended: `4.12.0`)
* Language Standard: **C++17** (or newer)

### Setup Instructions:
1. Clone the repository to your local machine.
2. Ensure OpenCV is properly installed and its environment variables are configured.
3. Import the `OpencvDebug.props` property sheet inside Visual Studio's Property Manager to automatically link OpenCV includes and libraries.
4. Place your raw test negatives in the `bw_negatyw/` directory within the project folder.
5. Build and run. The processed positive images will be saved in the same directory.

