
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;

int getdir (string dir, vector<string> &files);