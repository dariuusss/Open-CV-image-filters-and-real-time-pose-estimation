#include "Filter.h"
#include "glm/glm/glm.hpp"
#include <vector>

using namespace std;

Filter::Filter() {}

Filter::~Filter() {}

struct region {
	int nr_pixels;
	uchar avg_color;
	cv::Vec4f avg_normal;
	cv::Vec4b avg_depth;
};


void Filter::colorToGrayscale(cv::Mat colorImage) {
	int width = colorImage.cols;
	int height = colorImage.rows;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec4b color = colorImage.at<cv::Vec4b>(y, x);
			uchar gray = (uchar)((float)color[0] * 0.114 + (float)color[1] * 0.587 + (float)color[2] * 0.299);
			colorImage.at<cv::Vec4b>(y, x) = cv::Vec4b(gray, gray, gray, color[3]);
		}
	}
}


void Filter::colorToGrayscale(cv::Vec4b* colorData, int width, int height) {
	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			//offset = y * width + x;
			cv::Vec4b color = colorData[offset];
			uchar gray = (color[0] + color[1] + color[2]) / 3;
			colorData[offset] = cv::Vec4b(gray, gray, gray, color[3]);
			offset++;
		}
	}
}

void Filter::filterColorAverage(cv::Vec4b* colorData, cv::Vec4b* colorProcessedData, int width, int height) {
	int offset, offset_neighbor;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			cv::Vec4f color = cv::Vec4f(0, 0, 0, 0);
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					cv::Vec4b color_neighbor = colorData[offset_neighbor];
					color += (cv::Vec4f)color_neighbor;
				}
			}

			color /= 9;
			offset = y * width + x;
			colorProcessedData[offset] = cv::Vec4b(color[0], color[1], color[2], color[3]);

		}
	}

}

void Filter::filterDepthGaussian(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {

	float pondere = 273;
	int offset , offset_neighbor;

	int gaussian_kernel[5][5] = { {1,4,7,4,1}, {4,16,26,16,4} , {7,26,41,26,41}  ,{4,16,26,16,4} , {1,4,7,4,1} };

	for (int y = 2; y < height - 2; y++) {

		for (int x = 2; x < width - 2; x++) {

			offset = y * width + x;

			cv::Vec4f depth = cv::Vec4f(0, 0, 0, 0);
			for (int k = -2; k <= 2; k++)
			{
				for (int l = -2; l <= 2; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					cv::Vec4b depth_neighbor = depthData[offset_neighbor];
					depth += (cv::Vec4f)depth_neighbor * gaussian_kernel[k + 2][l + 2];
				}
			}

			depth /= pondere;
			depthProcessedData[offset] = cv::Vec4b(depth[0], depth[1], depth[2], depth[3]);

		}

	}

}

void Filter::filterGrayscaleGaussian(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {

	int gaussian_kernel[3][3] = { {1,2,1},{2,4,2},{1,2,1} };

	float pondere = 16;
	int offset = -1, offset_neighbor;

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {

			offset = y * width + x;
			float gray_quantity = 0;
			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					uchar depth_neighbor = grayscaleData[offset_neighbor];
					gray_quantity += (float)depth_neighbor * gaussian_kernel[k + 1][l + 1];
				}
			}

			gray_quantity /= pondere;
			grayscaleProcessedData[offset] = (uchar)gray_quantity;

		}
	}

}

void Filter::filterDepthByDistance(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {

	int offset = -1, offset_neighbor, c1, c2;
	uchar average;

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {

			offset = y * width + x;
			cv::Vec4b depth = depthData[offset];
			average = (depth[0] + depth[1] + depth[2] + depth[3]) / 4;
			c1 = 0;
			c2 = 50;
			while (c1 <= 200 && !(c1 <= average && average <= c2)) {
				c1 += 50;
				c2 += 50;
			}

			average = (c1 + c2) / 2;
			depthProcessedData[offset] = cv::Vec4b(average, average, average, average);

		}

}

void Filter::filterGrayscaleMedianFilter(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {

	int offset, offset_neighbor;
	uchar values[9], aux;
	int size;
	offset = -1;
	for (int y = 1; y < height - 1; y++)
		for (int x = 1; x < width - 1; x++) {
			offset = y * width + x;
			size = 0;

			for (int k = -1; k <= 1; k++)
				for (int l = -1; l <= 1; l++) {
					offset_neighbor = (y + k) * width + (x + l);
					values[size++] = grayscaleData[offset_neighbor];
				}

			for (int i = 0; i < size - 1; i++)
				for (int j = i + 1; j < size; j++)
					if (values[i] > values[j]) {
						aux = values[i];
						values[i] = values[j];
						values[j] = aux;
					}

			grayscaleProcessedData[offset] = values[4];


		}


}

void Filter::filterGrayscaleSobel(uchar* grayscaleData, uchar* grayscaleProcessedData, int width, int height) {

	int offset, offset_neighbor;
	int Gx, Gy, G;
	offset = -1;
	int sobel_vertical[3][3] = { {1,0,-1},{2,0,-2},{1,0,-1} };
	int sobel_orizontal[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	for (int y = 1; y < height - 1; y++)
		for (int x = 1; x < width - 1; x++) {
			offset = y * width + x;

			Gx = Gy = 0;
			for (int k = -1; k <= 1; k++)
				for (int l = -1; l <= 1; l++) {
					offset_neighbor = (y + k) * width + (x + l);
					Gx += sobel_vertical[k + 1][l + 1] * grayscaleData[offset_neighbor];
					Gy += sobel_orizontal[k + 1][l + 1] * grayscaleData[offset_neighbor];
				}

			if (Gx < 0)
				Gx = -Gx;
			if (Gy < 0)
				Gy = -Gy;
			if (sqrt(Gx * Gx + Gy * Gy) >= 255)
				grayscaleProcessedData[offset] = 255;
			else
				grayscaleProcessedData[offset] = (uchar)sqrt(Gx * Gx + Gy * Gy);
		}

}

void Filter::filterDepthPrewitt(cv::Vec4b* depthData, cv::Vec4b* depthProcessedData, int width, int height) {


	int offset, offset_neighbor;
	offset = -1;
	int prewitt_x[3][3] = { {1,1,1},{0,0,0},{-1,-1,-1} };
	int prewitt_y[3][3] = { {1,0,-1}, {1,0,-1}, {1,0,-1} };
	for (int y = 1; y < height - 1; y++)
		for (int x = 1; x < width - 1; x++) {
			offset = y * width + x;

			cv::Vec4f Gx = cv::Vec4f(0, 0, 0, 0);
			cv::Vec4f Gy = cv::Vec4f(0, 0, 0, 0);

			for (int k = -1; k <= 1; k++)
			{
				for (int l = -1; l <= 1; l++)
				{
					offset_neighbor = (y + k) * width + (x + l);
					cv::Vec4b depth_neighbor = depthData[offset_neighbor];
					Gx += (cv::Vec4f)depth_neighbor * prewitt_x[k + 1][l + 1];
					Gy += (cv::Vec4f)depth_neighbor * prewitt_y[k + 1][l + 1];
				}
			}


			(depthProcessedData[offset])[0] = (uchar)sqrt(Gx[0] * Gx[0] + Gy[0] * Gy[0]);
			(depthProcessedData[offset])[1] = (uchar)sqrt(Gx[1] * Gx[1] + Gy[1] * Gy[1]);
			(depthProcessedData[offset])[2] = (uchar)sqrt(Gx[2] * Gx[2] + Gy[2] * Gy[2]);
			(depthProcessedData[offset])[3] = (uchar)sqrt(Gx[3] * Gx[3] + Gy[3] * Gy[3]);
		}


}


void Filter::computeNormals(cv::Vec4f* pointCloudData, cv::Vec4f* normalMeasureComputedData, int width, int height)
{
	glm::vec3 p_left_vec, p_right_vec, p_up_vec, p_down_vec;
	cv::Vec4f p_left, p_right, p_up, p_down;
	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;

	int offset;
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			offset = y * width + x;
			p_left = pointCloudData[offset - 1];
			p_right = pointCloudData[offset + 1];
			p_up = pointCloudData[offset - width];
			p_down = pointCloudData[offset + width];
			p_left_vec = glm::vec3(p_left[0], p_left[1], p_left[2]);
			p_right_vec = glm::vec3(p_right[0], p_right[1], p_right[2]);
			p_up_vec = glm::vec3(p_up[0], p_up[1], p_up[2]);
			p_down_vec = glm::vec3(p_down[0], p_down[1], p_down[2]);
			vec_horiz = p_right_vec - p_left_vec;
			vec_vert = p_up_vec - p_down_vec;
			normal = glm::cross(vec_horiz, vec_vert);
			if (glm::length(normal) > 0.00001)
				normal = glm::normalize(normal);
			normalMeasureComputedData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}
}


void Filter::transformNormalsToImage(cv::Vec4f* normalMeasureComputedData, cv::Vec4b* normalImageComputedData, int width, int height)
{

	int offset = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			normalImageComputedData[offset] = cv::Vec4b((abs(normalMeasureComputedData[offset][2]) + 1) / 2 * 255,
				(abs(normalMeasureComputedData[offset][1]) + 1) / 2 * 255,
				(abs(normalMeasureComputedData[offset][0]) + 1) / 2 * 255, 0);

			offset++;
		}
	}

}

void Filter::computeNormals5x5Vicinity(cv::Vec4f* pointCloudData, cv::Vec4f* myNormalMeasureData, int width, int height) {

	
	glm::vec3 p_left_vec, p_right_vec, p_up_vec, p_down_vec;
	cv::Vec4f p_left, p_right, p_up, p_down = cv::Vec4f(0,0,0,0);
	glm::vec3 vec_horiz, vec_vert;
	glm::vec3 normal;

	int offset;
	for (int y = 2; y < height - 2; y++)
	{
		for (int x = 2; x < width - 2; x++)
		{
			offset = y * width + x;

			
			p_left = pointCloudData[offset - 2];
			p_right = pointCloudData[offset + 2];
			p_up = pointCloudData[offset - 2 * width];
			p_down = pointCloudData[offset + 2 * width];


			p_left_vec = glm::vec3(p_left[0], p_left[1], p_left[2]);
			p_right_vec = glm::vec3(p_right[0], p_right[1], p_right[2]);
			p_up_vec = glm::vec3(p_up[0], p_up[1], p_up[2]);
			p_down_vec = glm::vec3(p_down[0], p_down[1], p_down[2]);
			vec_horiz = p_right_vec - p_left_vec;
			vec_vert = p_up_vec - p_down_vec;
			normal = glm::cross(vec_horiz, vec_vert);
			if (glm::length(normal) > 0.00001)
				normal = glm::normalize(normal);
			myNormalMeasureData[offset] = cv::Vec4f(normal.x, normal.y, normal.z, 1);
		}
	}
	

}

void Filter::filterNormalSobel(cv::Vec4f* normalMeasure, cv::Vec4b* normalProcessedData, int width, int height) {

	int offset;
	float G, Gx, Gy;
	glm::vec3 m1, m2;
	cv::Vec4f elem;
	int i, j;
	for (int y = 1; y < height - 1; y++)
		for(int x = 1; x < width - 1; x++) {
			offset = y * width + x;
			Gx = Gy = 0;
			for (int k = -1; k <= 1; k++) {
				i = offset - 1 + k * width;
				j = offset + 1 + k * width;
				elem = normalMeasure[i];
				m1 = glm::vec3(elem[0], elem[1], elem[2]);
				m1 = glm::normalize(m1);
				elem = normalMeasure[j];
				m2 = glm::vec3(elem[0], elem[1], elem[2]);
				m2 = glm::normalize(m2);
				Gx = Gx + 1 - glm::dot(m1,m2);
				if (k == 0)
					Gx = Gx + 1 - glm::dot(m1, m2);

			}



			for (int k = -1; k <= 1; k++) {
				i = offset + k - width;
				j = offset + k + width;
				elem = normalMeasure[i];
				m1 = glm::vec3(elem[0], elem[1], elem[2]);
				m1 = glm::normalize(m1);
				elem = normalMeasure[j];
				m2 = glm::vec3(elem[0], elem[1], elem[2]);
				m2 = glm::normalize(m2);
				Gy = Gy + 1 - glm::dot(m1, m2);
				if (k == 0)
					Gy = Gy + 1 - glm::dot(m1, m2);

			}

			G = (uchar)(255 * sqrt(Gx * Gx + Gy * Gy));
			normalProcessedData[offset] = cv::Vec4b((uchar)G, (uchar)G, (uchar)G, 0);
		}		


}



int cost(cv::Vec4f* normalMeasure, int width, int height, int y, int x,
	vector <struct region> &regions, vector <int> &matrix) { //returnez offset-ul vecinului din regiunea la care adaug sau -1 daca nu exista

	double cost_minim = INT_MAX;
	double cost = 0;
	int offset, region_idx;
	int return_offset = -1;
	glm::vec3 n_pixel, n_regiune;
	cv::Vec4f n_elem;
	double normal_cost;
	cv::Vec4f elem1,elem2;

	if (y > 0 && x > 0) { // stanga sus
		offset = y * width + x - width - 1;
		region_idx = matrix[offset];
		n_elem = regions[region_idx].avg_normal;
		n_regiune = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		n_elem = normalMeasure[y * width + x];
		n_pixel = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		cost = 1 - glm::dot(n_pixel, n_regiune); //costul generat de normale
		
		if (cost < cost_minim) {
			cost_minim = cost;
			return_offset = offset;
		}

	}

	if (y > 0) { // sus
		offset = y * width + x - width;
		region_idx = matrix[offset];
		n_elem = regions[region_idx].avg_normal;
		n_regiune = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		n_elem = normalMeasure[y * width + x];
		n_pixel = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		cost = 1 - glm::dot(n_pixel, n_regiune); //costul generat de normale

		if (cost < cost_minim) {
			cost_minim = cost;
			return_offset = offset;
		}

	}


	if (y > 0 && x + 1 < width) { // dreapta sus
		offset = y * width + x - width + 1;
		region_idx = matrix[offset];
		n_elem = regions[region_idx].avg_normal;
		n_regiune = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		n_elem = normalMeasure[y * width + x];
		n_pixel = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		cost = 1 - glm::dot(n_pixel, n_regiune); //costul generat de normale

		if (cost < cost_minim) {
			cost_minim = cost;
			return_offset = offset;
		}

	}


	if (x > 0) { // stanga
		offset = y * width + x - 1;
		region_idx = matrix[offset];
		n_elem = regions[region_idx].avg_normal;
		n_regiune = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		n_elem = normalMeasure[y * width + x];
		n_pixel = glm::normalize(glm::vec3(n_elem[0], n_elem[1], n_elem[2]));
		cost = 1 - glm::dot(n_pixel, n_regiune); //costul generat de normale

		if (cost < cost_minim) {
			cost_minim = cost;
			return_offset = offset;
		}

	}



	if (return_offset != -1) {
		if (cost_minim > 0.2)
			return_offset = -1;
	}

	return return_offset;

}



void Filter::segmentarePlanara(cv::Vec4f* normalMeasure, cv::Vec4f* normalProcessedData, int width, int height) {

	vector <struct region> regions(width * height);
	vector <int> matrix(width * height);
	int nr_regions = 0;
	for (int i = 0; i < width * height; i++) {
		regions[i].nr_pixels = regions[i].avg_color = 0;
		regions[i].avg_depth = cv::Vec4b(0, 0, 0, 0);
		regions[i].avg_normal = cv::Vec4f(0, 0, 0, 0);
	}
	int offset,idx,region_idx;
	


	for(int y = 0; y < height; y++)
		for (int x = 0; x < width; x++) {

			if (nr_regions < width * height) {
				offset = y * width + x;
				if (x == 0 && y == 0) {
					nr_regions++;
					regions[0].nr_pixels = 1;
					regions[0].avg_normal = normalMeasure[offset];
					matrix[offset] = 0;
				} else {
					idx = cost(normalMeasure, width, height, y, x, regions, matrix);
					if (idx == -1) { // regiune noua
						nr_regions++;
						regions[nr_regions - 1].nr_pixels = 1;
						regions[nr_regions - 1].avg_normal = normalMeasure[offset];
						matrix[offset] = nr_regions - 1;
					} else { // facem merge
						matrix[offset] = matrix[idx];
						region_idx = matrix[idx];
						regions[region_idx].avg_normal = (regions[region_idx].avg_normal * regions[region_idx].nr_pixels + normalMeasure[offset]) /
							(regions[region_idx].nr_pixels + 1);
						regions[region_idx].nr_pixels++;

					}
				}

				normalProcessedData[y * width + x] = (regions[matrix[offset]].avg_normal + cv::Vec4f(0.1,0.2,0.3,0)) * 10;
				//normalProcessedData[y * width + x] = cv::Vec4f(1, 0, 0, 1);
			}
		}
}
