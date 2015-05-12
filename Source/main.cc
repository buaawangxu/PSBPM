/**
 * File: main.cc
 * Author: Jeanhwea
 * Email: hujinghui@buaa.edu.cn
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "ga.cuh"
#include "data.h"

using namespace std;

int main(int argc, char **argv)
{

    // call java to parse mpp files
    string full_filename = "tmp/input.mpp";
    // string full_filename = "tmp/C-softchoice.com.mpp";
    if (argc > 1)
        full_filename = argv[1];
    string cmd = "java -jar parser.jar " + full_filename;
    int ret_code = 0;
    ret_code = system(cmd.c_str());
    cout.flush();
    if (ret_code != 0) {
        return 0;
    }

    string xml_filename;
    int index_dot = full_filename.find_last_of('.');
    if (index_dot >= 0) {
        xml_filename = full_filename.substr(0, index_dot);
    }
    xml_filename += ".xml";

    loadXML(xml_filename);
    
    cuGaEvolve();

    // --------- clean --------------
    return 0;
}

