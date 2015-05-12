/**
 * File: data.h
 * Author: Jeanhwea
 * Email: hujinghui@buaa.edu.cn
 */

#ifndef  _DATA_H_
#define  _DATA_H_

#include <string>
#include "tinyxml/tinyxml.h"

using namespace std;

extern size_t ntask;
extern size_t nreso;

extern float * dura;
extern bool  * depd;
extern bool  * asgn;

int loadXML(string full_filename);
int dataAllocMemory();
int dataFreeMemory();


#endif //!_DATA_H_