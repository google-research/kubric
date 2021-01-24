#include "Parser.h"

Parser::Parser()
{
}

void Parser::ParseArgument(int argc, char** argv) {
	for (int i = 1; i < argc - 1; ++i) {
		std::string s = argv[i];
		if (s.size() < 2)
			continue;
		if (s[0] == '-' && s[1] == '-') {
			s = s.substr(2, s.size() - 2);
			arguments_[s] = argv[i + 1];
		}
	}
}

void Parser::AddArgument(const std::string& key, const std::string& value) {
	arguments_[key] = value;
}

std::string Parser::GetArgument(const std::string& key) {
	if (arguments_.count(key) == 0)
		return "";
	return arguments_[key];
}

std::string Parser::operator[](const std::string& key) {
	if (arguments_.count(key) == 0) {
		printf("Argument not found: %s\n", key.c_str());
		exit(0);
	}
	return arguments_[key];
}

void Parser::Log() {
	printf("##################### Arguments #####################\n");
	for (auto& info : arguments_) {
		if (info.second.size() > 0)
			printf("%s: %s.\n", info.first.c_str(), info.second.c_str());
		else
			printf("%s: None.\n", info.first.c_str());
	}
	printf("#####################################################\n");
}