#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_WORD_LENGTH 45
#define TARGET_UNIQUE_TOKENS 50000

#ifdef DEBUG
#define TARGET_UNIQUE_TOKENS 1000
#endif

#ifdef VERBOSE
#define LOG(x) std::cout << x << "\n"
#else
#define LOG(x)
#endif

void prepareAmazonReviewData(const std::string& inputPath, const std::string& outputPath);
void tokenizeDataset(const std::string& inputPath, const std::string& outputPath);

#endif // TOKENIZER_H
