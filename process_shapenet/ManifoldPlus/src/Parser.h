#ifndef MANIFOLD2_PARSER_H_
#define MANIFOLD2_PARSER_H_

#include <string>
#include <map>

class Parser
{
public:
	Parser();

	void ParseArgument(int argc, char** argv);
	void AddArgument(const std::string& key, const std::string& value);

	std::string GetArgument(const std::string& key);
	std::string operator[](const std::string& key);

	void Log();
private:
	std::map<std::string, std::string> arguments_;
};

#endif