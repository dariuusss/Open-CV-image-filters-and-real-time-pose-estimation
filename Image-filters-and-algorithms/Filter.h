#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Filter {
public:
	Filter();
	~Filter();

	//se transforma imaginea color in grayscale - folosind matricea de opencv
	static void colorToGrayscale(cv::Mat colorImage);

	//se transforma imaginea color in grayscale folosind pointerul catre datele din matricea cu informatiile de culoare
	static void colorToGrayscale(cv::Vec4b* colorData, int width, int height);

	//se porneste de la imaginea color si se obtine o imagine procesata, pentru care s-a facut o medie aritmetica 
	// a intensitatilor intr-o anumita vecinatate
	static void filterColorAverage(cv::Vec4b* colorData, cv::Vec4b* colorProcessedData, int width, int height);

	//TODO: se aplica filtrul Gaussian 5x5  pe imaginea de adancime si se salveaza info in imaginea de adancime procesata 
	// filtrul Gaussian: https://www.researchgate.net/publication/325768087_Gaussian_filtering_for_FPGA_based_image_processing_with_High-Level_Synthesis_tools/figures?lo=1
	static void filterDepthGaussian(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height);

	//TODO: se aplica filtrul Gaussian 3x3 pe imaginea grayscale si se salveaza info in imaginea grayscale procesata
	static void filterGrayscaleGaussian(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height);

	//TODO: se aplica filtrul Sobel (si vertical si orizontal) pe imaginea grayscale si se afiseaza amplitudinea gradientului
	//in imaginea grayscale procesata
	//filtrul Sobel: https://en.wikipedia.org/wiki/Sobel_operator
	static void filterGrayscaleSobel(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height);

	//TODO: se aplica filtrul median (care ordoneaza intensitatile dintr-o vecinatate si alege valoarea 
	// mediana - adica valoarea de pe pozitia de mijloc) intr-o vecinatate 3x3
	//Se aplica pe imaginea grayscale si rezultatul se salveaza in imaginea grayscale procesata
	//filtrul median: https://en.wikipedia.org/wiki/Median_filter
	static void filterGrayscaleMedianFilter(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height);

	//TODO: se discretizeaza imaginea de adancime
	//Se aleg 5 intervale de adancime (de ex: de la 0 la 50, de la 50 la 100, ... de la 200 la 255)
	//Toti pixelii dintr-un interval vor lua aceeasi valoare
	//Se salveaza datele in imaginea de adancime procesata
	static void filterDepthByDistance(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height);

	//TODO: se aplica filtrul Prewitt (si vertical si orizontal) pe imaginea de adancime si se afiseaza amplitudinea gradientului
	//in imaginea de adancime procesata
	//filtrul Prewitt: https://en.wikipedia.org/wiki/Prewitt_operator
	static void filterDepthPrewitt(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height);


	//calculeaza normalele folosind o vecinatate 3x3
	static void computeNormals(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height);

	//transforma normalele (vectorii) intr-o imagine cu harta de normale
	static void Filter::transformNormalsToImage(cv::Vec4f* normalMeasureComputedData, cv::Vec4b* normalImageComputedData, int width, int height);

	//TODO: calculeaza normalele folosind o vecinatate 5x5
	static void computeNormals5x5Vicinity(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height);

	//TODO: aplica masca Sobel pe harta de normale
	//in loc sa se calculeze diferenta intre intensitati, se calculeaza un cost care tine de unghiul dintre 2 normale
	//ex:  N1   N2   N3
	//     N4   N5   N6
	//     N7   N8   N9
	// Gx = (1 - dot(N1,N3)) + 2 * (1 - dot(N4,N6)) + (1 - dot(N7,N9))
	// Gy = (1 - dot(N1,N7)) + 2 * (1 - dot(N2,N8)) + (1 - dot(N3,N9))
	// G = sqrt(sqr(Gx)+sqr(Gy))
	//sau
	// G = abs(Gx)+abs(Gy)
	static void filterNormalSobel(cv::Vec4f* normalMeasure, cv::Vec4b* normalProcessedData, int width, int height);

	static void segmentarePlanara(cv::Vec4f* normalMeasure, cv::Vec4f* normalProcessedData, int width, int height);
	

};