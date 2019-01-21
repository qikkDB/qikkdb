#pragma once
class Configuration final
{
public:
	Configuration();
	~Configuration();
	static int BlockSize() { return 0; }
	static const char* DatabaseDir() { return "../databases"; }
};

