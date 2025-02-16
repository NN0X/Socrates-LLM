#ifndef CICERO_H
#define CICERO_H

#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <cstdint>

#define DEFAULT_MAX_WORD_LENGTH 45
#define DEFAULT_UNIQUE_TOKENS 65536

#ifndef MAX_WORD_LENGTH
#define MAX_WORD_LENGTH DEFAULT_MAX_WORD_LENGTH
#endif

#ifndef TARGET_UNIQUE_TOKENS
#define TARGET_UNIQUE_TOKENS DEFAULT_UNIQUE_TOKENS
#endif

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

#endif // CICERO_H
