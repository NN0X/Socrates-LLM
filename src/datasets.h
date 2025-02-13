#ifndef DATASETS_H
#define DATASETS_H

#include <vector>

void mergeDatasets(const std::vector<std::string>& datasetPaths, const std::string& outputPath);
void prepareAmazonReviewData(const std::string& inputPath, const std::string& outputPath);

#endif // DATASETS_H
