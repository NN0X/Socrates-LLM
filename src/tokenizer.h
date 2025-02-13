#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_WORD_LENGTH 45
#define TARGET_UNIQUE_TOKENS 32768

#ifdef DEBUG
#define TARGET_UNIQUE_TOKENS 500
#endif

#ifdef VERBOSE
#define LOG(x) std::cout << x << "\n"
#else
#define LOG(x)
#endif

#include <cstdint>
#include <unordered_map>
#include <queue>
#include <vector>
#include <string>

struct TokenDictionary
{
        std::unordered_map<std::string, uint16_t> tokenToIndex;
        std::unordered_map<uint16_t, std::string> indexToToken;
        std::priority_queue<std::pair<size_t, std::string>> tokensByLength;
};

void prepareAmazonReviewData(const std::string& inputPath, const std::string& outputPath);
void tokenizeDataset(const std::string& inputPath, const std::string& outputPath);
TokenDictionary loadTokenDictionary(const std::string& inputPath);
std::vector<uint16_t> tokenizeString(const std::string& input, TokenDictionary& tokenDictionary);
std::string detokenizeString(const std::vector<uint16_t>& tokens, TokenDictionary& tokenDictionary);

#endif // TOKENIZER_H
