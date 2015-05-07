#include "data.h"

size_t ntask;
size_t nreso;

float * dura;
bool  * depd;
bool  * asgn;

using namespace std;

TiXmlDocument   * pxmldoc;
TiXmlElement    * proot,
                * psize,
                * ptasks,
                * presources,
                * pdependencies,
                * passignments;

int initElements(string full_filename);
int loadInfo();


int loadXML(string full_filename) 
{
    initElements(full_filename);
    dataAllocMemory();
    loadInfo();
    return 0;
}

/************************************************************************/
/* initial XML-elements                                                 */
/************************************************************************/
int initElements(string full_filename)
{
    pxmldoc = new TiXmlDocument(full_filename);
    int ret = pxmldoc->LoadFile();
    assert(ret != 0);
    proot = pxmldoc->RootElement();
    assert(proot != 0);
    psize = proot->FirstChildElement("Size");
    assert(psize != 0);
    ptasks = proot->FirstChildElement("Tasks");
    assert(ptasks != 0);
    presources = proot->FirstChildElement("Resources");
    assert(presources != 0);
    pdependencies = proot->FirstChildElement("Dependencies");
    assert(pdependencies != 0);
    passignments = proot->FirstChildElement("Assignments");
    assert(passignments != 0);

    return 0;
}

/************************************************************************/
/* read size information and alloc memory to store data                 */
/************************************************************************/
int dataAllocMemory()
{
    string str_size;
    str_size = psize->Attribute("TaskSize");
    ntask = stoi(str_size);
    str_size = psize->Attribute("ResourceSize");
    nreso = stoi(str_size);

    dura = (float *) calloc(ntask, sizeof(float));
    if (dura == 0) {
        fprintf(stderr, "Error: cannot alloc memory!!!");
        assert(0);
    }

    depd = (bool *) calloc(ntask * ntask, sizeof(bool));
    if (depd == 0) {
        fprintf(stderr, "Error: cannot alloc memory!!!");
        assert(0);
    }

    asgn = (bool *) calloc(ntask * nreso, sizeof(bool));
    if (asgn == 0) {
        fprintf(stderr, "Error: cannot alloc memory!!!");
        assert(0);
    }

    return 0;
}

/************************************************************************/
/* free host memory of data store                                       */
/************************************************************************/
int dataFreeMemory()
{
    if (dura != 0) {
        free(dura);
        dura = 0;
    }

    if (depd != 0) {
        free(depd);
        depd = 0;
    }

    if (asgn != 0) {
        free(asgn);
        asgn = 0;
    }

    return 0;
}

/************************************************************************/
/* load information from XML-file, such as dependencies and assignments */
/************************************************************************/
int loadInfo()
{
    string str_text;
    TiXmlElement * pele;

    for (pele = ptasks->FirstChildElement("Task"); pele != 0; pele = pele->NextSiblingElement("Task")) {
        int itask;
        str_text = pele->Attribute("id");
        itask = stoi(str_text);
        str_text = pele->Attribute("duration");
        dura[itask-1] = stof(str_text);
    }

    for (pele = pdependencies->FirstChildElement("Dependency"); pele != 0; pele = pele->NextSiblingElement("Dependency")) {
        int ipred, isucc;
        str_text = pele->Attribute("predecessor");
        ipred = stoi(str_text);
        str_text = pele->Attribute("successor");
        isucc = stoi(str_text);
        depd[(ipred-1) + (isucc-1) * ntask] = true;
    }

    for (pele = passignments->FirstChildElement("Assignment"); pele != 0; pele = pele->NextSiblingElement("Assignment")) {
        int itask, ireso;
        str_text = pele->Attribute("task");
        itask = stoi(str_text);
        str_text = pele->Attribute("resource");
        ireso = stoi(str_text);
        asgn[(itask-1) + (ireso-1) * ntask] = true;
    }

    return 0;
}


