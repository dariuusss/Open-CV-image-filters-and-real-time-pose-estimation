// Computer Vision Plugin.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#if _MSC_VER // this is defined when compiling with Visual Studio
#define EXPORT_API __declspec(dllexport) // Visual Studio needs annotating exported functions with this
#else
#define EXPORT_API // XCode does not need annotating exported functions, so define is empty
#endif

#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

// ------------------------------------------------------------------------
// Plugin itself
// Link following functions C-style (required for plugins)
extern "C"
{
	struct Vector2
	{
		float x;
		float y;
	};

	struct Vector3
	{
		float x;
		float y;
		float z;
	};

	struct Vector4
	{
		float x;
		float y;
		float z;
		float w;
	};

	struct Matrix
	{
		float fx;
		float fy;
		float px;
		float py;
		float width;
		float height;
	};

	// The functions we will call from Unity.
	//
	void transform(const cv::Point3d& pt, const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3d& ptTrans)
	{
		cv::Mat R;
		cv::Rodrigues(rvec, R);

		cv::Mat matPt = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
		cv::Mat matPtTrans = R * matPt + tvec;
		ptTrans.x = matPtTrans.at<double>(0, 0);
		ptTrans.y = matPtTrans.at<double>(1, 0);
		ptTrans.z = matPtTrans.at<double>(2, 0);
	}

	void recoverPoseFromPnP(const std::vector<cv::Point3d>& objectPoints1, const cv::Mat& rvec1,
		const cv::Mat& tvec1, const std::vector<cv::Point2d>& imagePoints2,
		const cv::Mat& cameraMatrix, cv::Mat& rvec1to2, cv::Mat& tvec1to2)
	{
		cv::Mat R1;
		cv::Rodrigues(rvec1, R1);

		/* transform object points in camera frame */
		std::vector<cv::Point3d> objectPoints1InCam;
		for (size_t i = 0; i < objectPoints1.size(); i++) {
			cv::Point3d ptTrans;
			transform(objectPoints1[i], rvec1, tvec1, ptTrans);
			objectPoints1InCam.push_back(ptTrans);
		}

		cv::solvePnPRansac(objectPoints1InCam, imagePoints2, cameraMatrix, cv::noArray(), rvec1to2, tvec1to2, false, cv::SOLVEPNP_ITERATIVE);
	}

	EXPORT_API void PNP(Vector3 position1, Vector4 rotation1, Matrix cameraIntrinsics, Vector3 points3DUnity[8], Vector2 points2DUnity[8],
		Vector3 position2[1], Vector4 rotation2[1])
	{
		/* Quaternion to rotation matrix */
		/* TODO 2.1.1 Create quaternion from input data */
		cv::Vec4f quaternion1{ -rotation1.x, rotation1.y, -rotation1.z, rotation1.w };

		/* TODO 2.1.2 Convert quaternion to rotation matrix */
		cv::Mat rotationMatrix1(cv::Size(3, 3), CV_64FC1);
		struct Vector4 q1;
		q1.x = quaternion1[0];
		q1.y = quaternion1[1];
		q1.z = quaternion1[2];
		q1.w = quaternion1[3];

		double sqw, sqx, sqy, sqz, invSquareRoot, tmp1, tmp2;
		sqw = q1.w * q1.w;
		sqx = q1.x * q1.x;
		sqy = q1.y * q1.y;
		sqz = q1.z * q1.z;
		invSquareRoot = 1.0 / (sqx + sqy + sqz + sqw);

		rotationMatrix1.at<double>(0,0) = (sqx - sqy - sqz + sqw) * invSquareRoot;
		rotationMatrix1.at<double>(1,1) = (-sqx + sqy - sqz + sqw) * invSquareRoot;
		rotationMatrix1.at<double>(2,2) = (-sqx - sqy + sqz + sqw) * invSquareRoot;

		tmp1 = q1.x * q1.y;
		tmp2 = q1.z * q1.w;
		rotationMatrix1.at<double>(1, 0) = 2.0 * (tmp1 + tmp2) * invSquareRoot;
		rotationMatrix1.at<double>(0, 1) = 2.0 * (tmp1 - tmp2) * invSquareRoot;

		tmp1 = q1.x * q1.z;
		tmp2 = q1.y * q1.w;
		rotationMatrix1.at<double>(2, 0) = 2.0 * (tmp1 - tmp2) * invSquareRoot;
		rotationMatrix1.at<double>(0, 2) = 2.0 * (tmp1 + tmp2) * invSquareRoot;

		tmp1 = q1.y * q1.z;
		tmp2 = q1.x * q1.w;
		rotationMatrix1.at<double>(2, 1) = 2.0 * (tmp1 + tmp2) * invSquareRoot;
		rotationMatrix1.at<double>(1, 2) = 2.0 * (tmp1 - tmp2) * invSquareRoot;

		/* TODO 2.1.3 Convert rotation matrix to rotation vector */
		cv::Mat rotationVector1;
		cv::Rodrigues(rotationMatrix1,rotationVector1);
		/* TODO 2.2 Create translation vector from input data */
		cv::Mat translationVector1 = (cv::Mat_<double>(3, 1) << position1.x, -position1.y, position1.z);

		/* TODO 2.3 Create camera intrinsic matrix from input data */
		cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << cameraIntrinsics.fx, 0.0f, cameraIntrinsics.px,
			0.0f, cameraIntrinsics.fy, cameraIntrinsics.py,
			0.0f, 0.0f, 1.0f);

		/* Convert Unity feature points and their projections to OpenCV format */
		std::vector<cv::Point2d> points2DOpenCV;
		std::vector<cv::Point3d> points3DOpenCV;

		for (int i = 0; i < 8; i++)
		{
			points2DOpenCV.push_back(cv::Point2d(points2DUnity[i].x, cameraIntrinsics.height - points2DUnity[i].y));
			points3DOpenCV.push_back(cv::Point3d(points3DUnity[i].x, -points3DUnity[i].y, points3DUnity[i].z));
		}

		/* PNP */
		cv::Mat R1;
		Rodrigues(rotationVector1, R1);
		R1 = R1.t();
		translationVector1 = -R1 * translationVector1;
		Rodrigues(R1, rotationVector1);

		cv::Mat rotationVector2, translationVector2;
		recoverPoseFromPnP(points3DOpenCV, rotationVector1, translationVector1, points2DOpenCV, cameraMatrix, rotationVector2, translationVector2);

		/* TODO 2.4 Inverse results */
		cv::Mat R2;
		Rodrigues(rotationVector2, R2);
		R2 = R2.t();
		translationVector2 = -R2 * translationVector2;
		Rodrigues(R2, rotationVector2);
		/* TODO 2.5 Save position */
		position2[0].x = translationVector2.at<double>(0,0);
		position2[0].y = translationVector2.at<double>(1,0);
		position2[0].z = translationVector2.at<double>(2,0);

		/* Save orientation */
		/* - TODO 2.6.1 Convert rotation vector to rotation matrix */
		cv::Mat rm2(cv::Size(3, 3), CV_64FC1); // rotation matrix
		Rodrigues(rotationVector2, rm2);
		/* - TODO 2.6.2 Convert rotation matrix to quaternion */
		struct Vector4 q2;
		double matrix_trace = 0;
		for (int i = 0; i < 3; i++) {
			matrix_trace += rm2.at<double>(i, i);
		}
		
		double S;
		if (matrix_trace > 0) {
			S = sqrt(matrix_trace + 1.0) * 2;
			q2.w = S / 4.0;
			q2.x = (rm2.at<double>(2, 1) - rm2.at<double>(1, 2)) / S;
			q2.y = (rm2.at<double>(0, 2) - rm2.at<double>(2, 0)) / S;
			q2.z = (rm2.at<double>(1, 0) - rm2.at<double>(0, 1)) / S;

		}
		else if (rm2.at<double>(0, 0) > rm2.at<double>(1, 1) and rm2.at<double>(0, 0) > rm2.at<double>(2, 2)) {

			S = sqrt(1.0 + rm2.at<double>(0, 0) - rm2.at<double>(1, 1) - rm2.at<double>(2, 2)) * 2;
			q2.w = (rm2.at<double>(2, 1) - rm2.at<double>(1, 2)) / S;
			q2.x = S / 4.0;
			q2.y = (rm2.at<double>(0, 1) + rm2.at<double>(1, 0)) / S;
			q2.z = (rm2.at<double>(0, 2) + rm2.at<double>(2, 0)) / S;

		}
		else if (rm2.at<double>(1, 1) > rm2.at<double>(2, 2)) {

			S = sqrt(1.0 + rm2.at<double>(1, 1) - rm2.at<double>(0, 0) - rm2.at<double>(2, 2)) * 2;
			q2.w = (rm2.at<double>(0, 2) - rm2.at<double>(2, 0)) / S;
			q2.x = (rm2.at<double>(0, 1) + rm2.at<double>(1, 0)) / S;
			q2.y = S / 4.0;
			q2.z = (rm2.at<double>(1, 2) + rm2.at<double>(2, 1)) / S;

		} else {
			S = sqrt(1.0 + rm2.at<double>(2, 2) - rm2.at<double>(0, 0) - rm2.at<double>(1, 1)) * 2;
			q2.w = (rm2.at<double>(1, 0) - rm2.at<double>(0, 1)) / S;
			q2.x = (rm2.at<double>(0, 2) + rm2.at<double>(2, 0)) / S;
			q2.y = (rm2.at<double>(1, 2) + rm2.at<double>(2, 1)) / S;
			q2.z = S / 4.0;
		}



		/* - TODO 2.6.3 Normalize the quaternion */
		cv::Vec4f quaternion2;
		quaternion2[0] = q2.x;
		quaternion2[1] = q2.y;
		quaternion2[2] = q2.z;
		quaternion2[3] = q2.w;
		cv::normalize(quaternion2, quaternion2);
	}
} // end of export C block

/* Detect feature points with SIFT. Match feature points with FLANN */
void DetectorFLANNSift(cv::Mat image1, cv::Mat image2, std::ofstream& measure)
{
	cv::Ptr<cv::Feature2D> detectorSift;
	cv::Ptr<cv::DescriptorMatcher> matcherFlann;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	std::vector<std::vector<cv::DMatch> > knn_matches;
	std::vector<cv::DMatch> good_matches;

	const float ratio_thresh = 0.75f;
	/* SIFT detector */
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	/* TODO 1.2.1 Intialize SIFT detector */
	detectorSift = cv::SIFT::create();
	/* TODO 1.2.2 Detect keypoints and descriptors in both images */
	detectorSift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detectorSift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	/* TODO 1.6.1 Compute execution time of SIFT */
	std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	auto durationSift = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	/* Save execution time and number of feature points */
	measure << "SIFT detectors execution time: " << durationSift.count() << std::endl;
	measure << "SIFT detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	/* Save output image */
	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("SIFT.jpg", output);

	/* FLANNMatcher */
	std::chrono::high_resolution_clock::time_point start_flann = std::chrono::high_resolution_clock::now();
	/* TODO 1.3.1 Initialize FLANN Matcher */
	matcherFlann = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	/* TODO 1.3.2 Use knnMatch to match feature points in the two images */
	matcherFlann->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	/* TODO 1.3.3 Filter the matches using Lowe's ratio test. Save the result in good_matches */
	for (size_t i = 0; i < knn_matches.size(); i++) {
		if (knn_matches[i][0].distance < 0.7 * knn_matches[i][1].distance) {
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	/* TODO 1.6.3 Compute execution time of FLANN */
	std::chrono::high_resolution_clock::time_point stop_flann = std::chrono::high_resolution_clock::now();
	auto durationFlann = std::chrono::duration_cast<std::chrono::milliseconds>(stop_flann - start_flann);

	/* Save execution time and number of matched feature points */
	measure << "FLANN matcher execution time: " << durationFlann.count() << std::endl;
	measure << "FLANN matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	/* Save results */
	cv::Mat image_matches;
	drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("SIFT_Flann.jpg", image_matches);
}

/* Detect feature points with SIFT. Match feature points with BF */
void DetectorBFSift(cv::Mat image1, cv::Mat image2, std::ofstream& measure)
{
	cv::Ptr<cv::Feature2D> detectorSift;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	std::vector<cv::DMatch> matches;
	std::vector<cv::DMatch> good_matches;

	int maximumSize = 20;

	/* SIFT detector */
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	/* TODO 1.2.1 Intialize SIFT detector */
	detectorSift = cv::SIFT::create();
	/* TODO 1.2.2 Detect keypoints and descriptors in both images */
	detectorSift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detectorSift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	/* TODO 1.6.1 Compute execution time of SIFT */
	std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
	auto durationSift = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	/* Save execution time and number of feature points */
	measure << "SIFT detectors execution time: " << durationSift.count() << std::endl;
	measure << "SIFT detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;


	/* BFMatcher */
	std::chrono::high_resolution_clock::time_point start_bf = std::chrono::high_resolution_clock::now();
	/* TODO 1.4.1 Initialize BFMatcher */
	matcherBF = cv::BFMatcher::create();
	/* TODO 1.4.2 Match the feature points in the two images */
	matcherBF->match(descriptors1,descriptors2,matches,cv::noArray());
	/* Save in good_matches the best 20 results */
	sort(begin(matches), end(matches), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	if (maximumSize > matches.size())
	{
		maximumSize = matches.size();
	}

	for (size_t i = 0; i < maximumSize; i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::chrono::high_resolution_clock::time_point final_bf = std::chrono::high_resolution_clock::now();
	/* TODO 1.6.4 Compute execution time of BF */
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_bf - start_bf);

	/* Save execution time and number of matched feature points */
	measure << "BF matcher execution time: " << durationBF.count() << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	/* Save results */
	cv::Mat image_matches;
	drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("SIFT_BF.jpg", image_matches);
}

/* Detect feature points with ORB. Match feature points with BF */
void DetectorBFOrb(cv::Mat image1, cv::Mat image2, std::ofstream& measure)
{
	cv::Ptr<cv::ORB> detectorOrb;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	std::vector<cv::DMatch> matches;
	std::vector<cv::DMatch> good_matches;

	int maximumSize = 20;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	/* ORB detector */
	/* TODO 1.5.1 Intialize ORB detector */
	detectorOrb = cv::ORB::create();
	/* TODO 1.5.2 Detect keypoints and descriptors in both images */
	detectorOrb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detectorOrb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	/* TODO 1.6.2 Compute execution time of ORB */
	std::chrono::high_resolution_clock::time_point final = std::chrono::high_resolution_clock::now();
	auto durationOrb = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);

	/* Save execution time and number of feature points */
	measure << "ORB detectors execution time: " << durationOrb.count() << std::endl;
	measure << "ORB detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	/* Save output image */
	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("ORB.jpg", output);

	/* BFMatcher */
	std::chrono::high_resolution_clock::time_point start_brute_force = std::chrono::high_resolution_clock::now();
	/* TODO 1.4.1 Initialize BFMatcher */
	matcherBF = cv::BFMatcher::create();
	/* TODO 1.4.2 Match the feature points in the two images */
	matcherBF->match(descriptors1, descriptors2, matches, cv::noArray());
	/* Save in good_matches the best 20 results */
	sort(begin(matches), end(matches), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	if (maximumSize > matches.size())
	{
		maximumSize = matches.size();
	}

	for (size_t i = 0; i < maximumSize; i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::chrono::high_resolution_clock::time_point final_brute_force = std::chrono::high_resolution_clock::now();
	/* TODO 1.6.4 Compute execution time of BF */
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);

	/* Save execution time and number of matched feature points */
	measure << "BF matcher execution time: " << durationBF.count() << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	/* Save results */
	cv::Mat image_matches;
	drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("ORB_BF.jpg", image_matches);
}


void DetectorBFSurf(cv::Mat image1, cv::Mat image2, std::ofstream& measure) {

	cv::Ptr<cv::xfeatures2d::SURF> detectorSurf;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	std::vector<cv::DMatch> matches;
	std::vector<cv::DMatch> good_matches;

	
	int maximumSize = 20;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	detectorSurf = cv::xfeatures2d::SURF::create();
	detectorSurf->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detectorSurf->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	std::chrono::high_resolution_clock::time_point final = std::chrono::high_resolution_clock::now();
	auto durationOrb = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);
	measure << "SURF detectors execution time: " << durationOrb.count() << "ms" <<  std::endl;
	measure << "SURF detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("SURF.jpg", output);

	/* BFMatcher */
	std::chrono::high_resolution_clock::time_point start_brute_force = std::chrono::high_resolution_clock::now();
	matcherBF = cv::BFMatcher::create();
	matcherBF->match(descriptors1, descriptors2, matches, cv::noArray());
	sort(begin(matches), end(matches), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });
	if (maximumSize > matches.size())
	{
		maximumSize = matches.size();
	}
	for (size_t i = 0; i < maximumSize; i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::chrono::high_resolution_clock::time_point final_brute_force = std::chrono::high_resolution_clock::now();
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);
	measure << "BF matcher execution time: " << durationBF.count() << "ms" << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	
	cv::Mat image_matches;
	drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1),
		cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("SURF_BF.jpg", image_matches);
}

void DetectorBFFast(cv::Mat image1, cv::Mat image2, std::ofstream& measure) {

	cv::Ptr<cv::FastFeatureDetector> detectorFast;
	cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> descriptorBRIEF;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches, good_matches;
	int maximumSize = 20;

	auto start = std::chrono::high_resolution_clock::now();
	detectorFast = cv::FastFeatureDetector::create();
	detectorFast->detect(image1, keypoints1);
	detectorFast->detect(image2, keypoints2);
	descriptorBRIEF = cv::xfeatures2d::BriefDescriptorExtractor::create();
	descriptorBRIEF->compute(image1, keypoints1, descriptors1);
	descriptorBRIEF->compute(image2, keypoints2, descriptors2);
	auto final = std::chrono::high_resolution_clock::now();
	auto durationFast = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);
	measure << "FAST detectors execution time: " << durationFast.count() << " ms" << std::endl;
	measure << "FAST detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	
	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("FAST.jpg", output);

	
	auto start_brute_force = std::chrono::high_resolution_clock::now();
	matcherBF = cv::BFMatcher::create(cv::NORM_HAMMING);
	matcherBF->match(descriptors1, descriptors2, matches);
	std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	maximumSize = std::min(maximumSize, (int)matches.size());
	good_matches.insert(good_matches.end(), matches.begin(), matches.begin() + maximumSize);

	auto final_brute_force = std::chrono::high_resolution_clock::now();
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);
	measure << "BF matcher execution time: " << durationBF.count() << " ms" << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	cv::Mat image_matches;
	cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("FAST_BF.jpg", image_matches);
}


void DetectorBFStar(cv::Mat image1, cv::Mat image2, std::ofstream& measure) {
	
	cv::Ptr<cv::xfeatures2d::StarDetector> detectorStar;
	cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> descriptorBRIEF;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches, good_matches;
	int maximumSize = 20;

	
	auto start = std::chrono::high_resolution_clock::now();
	detectorStar = cv::xfeatures2d::StarDetector::create();
	detectorStar->detect(image1, keypoints1);
	detectorStar->detect(image2, keypoints2);
	descriptorBRIEF = cv::xfeatures2d::BriefDescriptorExtractor::create();
	descriptorBRIEF->compute(image1, keypoints1, descriptors1);
	descriptorBRIEF->compute(image2, keypoints2, descriptors2);
	auto final = std::chrono::high_resolution_clock::now();
	auto durationStar = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);
	measure << "STAR detectors execution time: " << durationStar.count() << " ms" << std::endl;
	measure << "STAR detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;


	
	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("STAR.jpg", output);

	
	auto start_brute_force = std::chrono::high_resolution_clock::now();
	matcherBF = cv::BFMatcher::create(cv::NORM_HAMMING);
	matcherBF->match(descriptors1, descriptors2, matches);
	std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	maximumSize = std::min(maximumSize, (int)matches.size());
	good_matches.insert(good_matches.end(), matches.begin(), matches.begin() + maximumSize);

	auto final_brute_force = std::chrono::high_resolution_clock::now();
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);
	measure << "BF matcher execution time: " << durationBF.count() << " ms" << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	
	cv::Mat image_matches;
	cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("STAR_BF.jpg", image_matches);
}

void DetectorBFBrisk(cv::Mat image1, cv::Mat image2, std::ofstream& measure) {

	cv::Ptr<cv::BRISK> detectorBRISK;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches, good_matches;
	int maximumSize = 20;

	auto start = std::chrono::high_resolution_clock::now();
	detectorBRISK = cv::BRISK::create();
	detectorBRISK->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detectorBRISK->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	auto final = std::chrono::high_resolution_clock::now();
	auto durationBRISK = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);
	measure << "BRISK detectors execution time: " << durationBRISK.count() << " ms" << std::endl;
	measure << "BRISK detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("BRISK.jpg", output);

	auto start_brute_force = std::chrono::high_resolution_clock::now();
	matcherBF = cv::BFMatcher::create(cv::NORM_HAMMING);
	matcherBF->match(descriptors1, descriptors2, matches);
	std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	maximumSize = std::min(maximumSize, (int)matches.size());
	good_matches.insert(good_matches.end(), matches.begin(), matches.begin() + maximumSize);

	auto final_brute_force = std::chrono::high_resolution_clock::now();
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);
	measure << "BF matcher execution time: " << durationBF.count() << " ms" << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	
	cv::Mat image_matches;
	cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("BRISK_BF.jpg", image_matches);
}


void DetectorBFGFTT(cv::Mat image1, cv::Mat image2, std::ofstream& measure) {

	cv::Ptr<cv::GFTTDetector> detectorGFTT;
	cv::Ptr<cv::ORB> descriptorORB;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches, good_matches;
	int maximumSize = 20;

	auto start = std::chrono::high_resolution_clock::now();
	detectorGFTT = cv::GFTTDetector::create();
	detectorGFTT->detect(image1, keypoints1);
	detectorGFTT->detect(image2, keypoints2);

	descriptorORB = cv::ORB::create();
	descriptorORB->compute(image1, keypoints1, descriptors1);
	descriptorORB->compute(image2, keypoints2, descriptors2);

	auto final = std::chrono::high_resolution_clock::now();
	auto durationGFTT = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);
	measure << "GFTT detectors execution time: " << durationGFTT.count() << " ms" << std::endl;
	measure << "GFTT detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("GFTT.jpg", output);

	auto start_brute_force = std::chrono::high_resolution_clock::now();
	matcherBF = cv::BFMatcher::create(cv::NORM_HAMMING);
	matcherBF->match(descriptors1, descriptors2, matches);
	std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });

	maximumSize = std::min(maximumSize, (int)matches.size());
	good_matches.insert(good_matches.end(), matches.begin(), matches.begin() + maximumSize);

	auto final_brute_force = std::chrono::high_resolution_clock::now();
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);
	measure << "BF matcher execution time: " << durationBF.count() << " ms" << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	
	cv::Mat image_matches;
	cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("GFTT_BF.jpg", image_matches);
}



void DetectorBFKAZE(cv::Mat image1, cv::Mat image2, std::ofstream& measure) {

	cv::Ptr<cv::KAZE> detectorKAZE;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches, good_matches;
	int maximumSize = 20;

	auto start = std::chrono::high_resolution_clock::now();
	detectorKAZE = cv::KAZE::create();
	detectorKAZE->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detectorKAZE->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	auto final = std::chrono::high_resolution_clock::now();
	auto durationKAZE = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);
	measure << "KAZE detectors execution time: " << durationKAZE.count() << " ms" << std::endl;
	measure << "KAZE detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("KAZE.jpg", output);

	auto start_brute_force = std::chrono::high_resolution_clock::now();
	matcherBF = cv::BFMatcher::create(cv::NORM_L2); 
	matcherBF->match(descriptors1, descriptors2, matches);
	std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });
	maximumSize = std::min(maximumSize, (int)matches.size());
	good_matches.insert(good_matches.end(), matches.begin(), matches.begin() + maximumSize);
	auto final_brute_force = std::chrono::high_resolution_clock::now();
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);
	measure << "BF matcher execution time: " << durationBF.count() << " ms" << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;

	
	cv::Mat image_matches;
	cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("KAZE_BF.jpg", image_matches);
}

void DetectorBFAKAZE(cv::Mat image1, cv::Mat image2, std::ofstream& measure) {

	cv::Ptr<cv::AKAZE> detectorAKAZE;
	cv::Ptr<cv::DescriptorMatcher> matcherBF;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches, good_matches;
	int maximumSize = 20;

	auto start = std::chrono::high_resolution_clock::now();
	detectorAKAZE = cv::AKAZE::create();
	detectorAKAZE->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	detectorAKAZE->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	auto final = std::chrono::high_resolution_clock::now();
	auto durationAKAZE = std::chrono::duration_cast<std::chrono::milliseconds>(final - start);
	measure << "AKAZE detectors execution time: " << durationAKAZE.count() << " ms" << std::endl;
	measure << "AKAZE detected " << keypoints1.size() << " and " << keypoints2.size() << " keypoints" << std::endl;

	cv::Mat output;
	cv::drawKeypoints(image1, keypoints1, output);
	cv::imwrite("AKAZE.jpg", output);

	auto start_brute_force = std::chrono::high_resolution_clock::now();
	matcherBF = cv::BFMatcher::create(cv::NORM_L2);
	matcherBF->match(descriptors1, descriptors2, matches);
	std::sort(matches.begin(), matches.end(), [](cv::DMatch a, cv::DMatch b) { return a.distance < b.distance; });
	maximumSize = std::min(maximumSize, (int)matches.size());
	good_matches.insert(good_matches.end(), matches.begin(), matches.begin() + maximumSize);
	auto final_brute_force = std::chrono::high_resolution_clock::now();
	auto durationBF = std::chrono::duration_cast<std::chrono::milliseconds>(final_brute_force - start_brute_force);
	measure << "BF matcher execution time: " << durationBF.count() << " ms" << std::endl;
	measure << "BF matched " << good_matches.size() << " keypoints" << std::endl;
	measure << std::endl;


	cv::Mat image_matches;
	cv::drawMatches(image1, keypoints1, image2, keypoints2, good_matches, image_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imwrite("AKAZE_BF.jpg", image_matches);
}



void FeatureDetection()
{
	/* TODO 1.1 Read images created by Unity */
	cv::Mat image1 = cv::imread("D:\\3dpub_final_boss\\3d_cameras_part_2\\3DUPB-main\\3D Cameras\\Computer Vision Plugin\\Input Images\\image0.jpg", cv::IMREAD_COLOR);
	cv::Mat image2 = cv::imread("D:\\3dpub_final_boss\\3d_cameras_part_2\\3DUPB-main\\3D Cameras\\Computer Vision Plugin\\Input Images\\image1.jpg", cv::IMREAD_COLOR);

	/* Output file for measurements */
	std::ofstream measure;
	measure.open("measurements.txt");

	/* Detect feature points with SIFT. Match feature points with FLANN */
	DetectorFLANNSift(image1, image2, measure);

	/* Detect feature points with SIFT. Match feature points with BF */
	DetectorBFSift(image1, image2, measure);

	/* Detect feature points with ORB. Match feature points with BF.
	 * What would happen if we tried to match ORB detected feature points using FLANN?
	 */
	DetectorBFOrb(image1, image2, measure);

	/* TODO 1.7 Implement the following keypoints detectors and descriptors extractors:
	
	   SURF // done
	   FAST // done
	   STAR // done
	   BRISK //done
	   GFTT // done
	   KAZE // done
	   AKAZE // done
	 */

	DetectorBFSurf(image1, image2, measure);
	DetectorBFFast(image1, image2, measure);
	DetectorBFStar(image1, image2, measure);
	DetectorBFBrisk(image1, image2, measure);
	DetectorBFGFTT(image1, image2, measure); 
	DetectorBFKAZE(image1, image2, measure);
	DetectorBFAKAZE(image1, image2, measure);
	measure.close();
}

int main()
{
	FeatureDetection();
	return 0;
}