/*
g++-10 -m64 -mssse3 -Wno-unused-result -Wno-write-strings -O2 -I. -I/usr/local/cuda-8.0/include -o obj/dang.o -c dang.cpp
g++-10 obj/SECPK1/IntGroup.o obj/dang.o obj/SECPK1/Random.o obj/Timer.o obj/SECPK1/Int.o obj/SECPK1/IntMod.o obj/SECPK1/Point.o obj/SECPK1/SECP256K1.o obj/Kangaroo.o obj/HashTable.o obj/Thread.o obj/Check.o obj/Backup.o obj/Network.o obj/Merge.o obj/PartMerge.o obj/hash/ripemd160.o obj/hash/sha256.o obj/hash/sha512.o obj/hash/ripemd160_sse.o obj/hash/sha256_sse.o obj/Bech32.o obj/Base58.o -lpthread -o dang
*/
#include "Kangaroo.h"
#include "Timer.h"
#include "SECPK1/SECP256k1.h"
#include "GPU/GPUEngine.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>

using namespace std;

#define CHECKARG(opt,n) if(a>=argc-1) {::printf(opt " missing argument #%d\n",n);exit(0);} else {a++;}

// ------------------------------------------------------------------------------------------
#define CPU_GRP_SIZE 0xFFFFFF

void printUsage() {

    printf("Kangaroo [-v] [-t nbThread] [-d dpBit] [gpu] [-check]\n");
    printf("         [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y[,g2x,g2y,...]]\n");
    printf("         inFile\n");
    printf(" -v: Print version\n");
    printf(" -gpu: Enable gpu calculation\n");
    printf(" -gpuId gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
    printf(" -g g1x,g1y,g2x,g2y,...: Specify GPU(s) kernel gridsize, default is 2*(MP),2*(Core/MP)\n");
    printf(" -d: Specify number of leading zeros for the DP method (default is auto)\n");
    printf(" -t nbThread: Secify number of thread\n");
    printf(" -w workfile: Specify file to save work into (current processed key only)\n");
    printf(" -i workfile: Specify file to load work from (current processed key only)\n");
    printf(" -wi workInterval: Periodic interval (in seconds) for saving work\n");
    printf(" -ws: Save kangaroos in the work file\n");
    printf(" -wss: Save kangaroos via the server\n");
    printf(" -wsplit: Split work file of server and reset hashtable\n");
    printf(" -wm file1 file2 destfile: Merge work file\n");
    printf(" -wmdir dir destfile: Merge directory of work files\n");
    printf(" -wt timeout: Save work timeout in millisec (default is 3000ms)\n");
    printf(" -winfo file1: Work file info file\n");
    printf(" -wpartcreate name: Create empty partitioned work file (name is a directory)\n");
    printf(" -wcheck worfile: Check workfile integrity\n");
    printf(" -m maxStep: number of operations before give up the search (maxStep*expected operation)\n");
    printf(" -s: Start in server mode\n");
    printf(" -c server_ip: Start in client mode and connect to server server_ip\n");
    printf(" -sp port: Server port, default is 17403\n");
    printf(" -nt timeout: Network timeout in millisec (default is 3000ms)\n");
    printf(" -o fileName: output result to fileName\n");
    printf(" -l: List cuda enabled devices\n");
    printf(" -check: Check GPU kernel vs CPU\n");
    printf(" inFile: intput configuration file\n");
    exit(0);

}

// ------------------------------------------------------------------------------------------

int getInt(string name,char *v) {

    int r;

    try {

        r = std::stoi(string(v));

    } catch(std::invalid_argument&) {

        printf("Invalid %s argument, number expected\n",name.c_str());
        exit(-1);

    }

    return r;

}

double getDouble(string name,char *v) {

    double r;

    try {

        r = std::stod(string(v));

    } catch(std::invalid_argument&) {

        printf("Invalid %s argument, number expected\n",name.c_str());
        exit(-1);

    }

    return r;

}

// ------------------------------------------------------------------------------------------

void getInts(string name,vector<int> &tokens,const string &text,char sep) {

    size_t start = 0,end = 0;
    tokens.clear();
    int item;

    try {

        while((end = text.find(sep,start)) != string::npos) {
            item = std::stoi(text.substr(start,end - start));
            tokens.push_back(item);
            start = end + 1;
        }

        item = std::stoi(text.substr(start));
        tokens.push_back(item);

    }
    catch(std::invalid_argument &) {

        printf("Invalid %s argument, number expected\n",name.c_str());
        exit(-1);

    }

}
// ------------------------------------------------------------------------------------------

// Default params
static int dp = -1;
static int nbCPUThread;
static string configFile = "";
static bool checkFlag = false;
static bool gpuEnable = false;
static vector<int> gpuId = { 0 };
static vector<int> gridSize;
static string workFile = "";
static string checkWorkFile = "";
static string iWorkFile = "";
static uint32_t savePeriod = 60;
static bool saveKangaroo = false;
static bool saveKangarooByServer = false;
static string merge1 = "";
static string merge2 = "";
static string mergeDest = "";
static string mergeDir = "";
static string infoFile = "";
static double maxStep = 0.0;
static int wtimeout = 3000;
static int ntimeout = 3000;
static int port = 17403;
static bool serverMode = false;
static string serverIP = "";
static string outputFile = "";
static bool splitWorkFile = false;
//-----------------------------------------------------------------------------
Int rangeStart;
Int rangeEnd;
std::vector<std::string> keysToSearch;

bool ParseConfigFile(std::string &fileName) {

    // Check file
    FILE *fp = fopen(fileName.c_str(),"rb");
    if(fp == NULL) {
        ::printf("Error: Cannot open %s %s\n",fileName.c_str(),strerror(errno));
        return false;
    }
    fclose(fp);

    // Get lines
    vector<string> lines;
    int nbLine = 0;
    string line;
    ifstream inFile(fileName);
    while(getline(inFile,line)) {

        // Remove ending \r\n
        int l = (int)line.length() - 1;
        while(l >= 0 && isspace(line.at(l))) {
            line.pop_back();
            l--;
        }

        if(line.length() > 0) {
            lines.push_back(line);
            nbLine++;
        }
    }

    if(lines.size()<3) {
        printf("Error: %s not enough arguments\n",fileName.c_str());
        return false;
    }

    rangeStart.SetBase16((char *)lines[0].c_str());
    rangeEnd.SetBase16((char *)lines[1].c_str());
    for(int i=2;i<(int)lines.size();i++) {
        keysToSearch.push_back(lines[i]);
    }

    ::printf("Start:%s\n",rangeStart.GetBase16().c_str());
    ::printf("Stop :%s\n",rangeEnd.GetBase16().c_str());
    ::printf("Keys :%d\n",(int)keysToSearch.size());

    return true;
}

void demo()
{
    char * lines = "1";
    Int pk;
    Secp256K1 *secp = new Secp256K1();
    secp->Init();

    pk.SetBase16(lines);
    Point P = secp->ComputePublicKey(&pk);

    hash160_t h1;
    secp->GetHash160(P2PKH, true, P, h1.i8);
    std::string addr = secp->GetAddress(P2PKH, true, h1.i8).c_str();

    printf("       Priv: 0x%s \n",pk.GetBase16().c_str());
    printf("       Pub : 0x%s \n",secp->GetPublicKeyHex(true,P).c_str());
    printf("       Add : %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());

    if (addr.compare("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH") == 0) {
        printf("       Priv: 0x%s \n",pk.GetBase16().c_str());
        printf("       Pub : 0x%s \n",secp->GetPublicKeyHex(true,P).c_str());
        printf("       Add : %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());
    }

}

THREAD_HANDLE LaunchThread(void *(*func) (void *), TH_PARAM *p) {
    THREAD_HANDLE h;
    pthread_create(&h, NULL, func, (void*)(p));
    return h;
}
void  JoinThreads(THREAD_HANDLE *handles, int nbThread) {
    for (int i = 0; i < nbThread; i++)
        pthread_join(handles[i], NULL);
}
void  FreeHandles(THREAD_HANDLE *handles, int nbThread) {
}

void MySolveKeyCPU(TH_PARAM *ph) {

    vector<ITEM> dps;
    double lastSent = 0;

    // Global init
    int thId = ph->threadId;

    // Create Kangaroos
    ph->nbKangaroo = CPU_GRP_SIZE;

    Int pk;
    Secp256K1 *secp = new Secp256K1();
    secp->Init();

    for(uint32_t i = 0; i < CPU_GRP_SIZE; i++) {
        pk.SetInt32(i);
        //pk.SetBase16(lines);
        Point P = secp->ComputePublicKey(&pk);

        hash160_t h1;
        secp->GetHash160(P2PKH, true, P, h1.i8);
        std::string addr = secp->GetAddress(P2PKH, true, h1.i8).c_str();

        if (addr.compare("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH1") == 0) {
            printf("       Priv: 0x%s \n",pk.GetBase16().c_str());
            printf("       Pub : 0x%s \n",secp->GetPublicKeyHex(true,P).c_str());
            printf("       Add : %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());
        }
    }
}
void *_MySolveKeyCPU(void *lpParam) {
    TH_PARAM *p = (TH_PARAM *)lpParam;
    MySolveKeyCPU(p);
    return 0;
}

string GetTimeStr(double dTime) {

    char tmp[256];

    double nbDay = dTime / 86400.0;
    if (nbDay >= 1) {

        double nbYear = nbDay / 365.0;
        if (nbYear > 1) {
            if (nbYear < 5)
                sprintf(tmp, "%.1fy", nbYear);
            else
                sprintf(tmp, "%gy", nbYear);
        } else {
            sprintf(tmp, "%.1fd", nbDay);
        }

    } else {

        int iTime = (int)dTime;
        int nbHour = (int)((iTime % 86400) / 3600);
        int nbMin = (int)(((iTime % 86400) % 3600) / 60);
        int nbSec = (int)(iTime % 60);

        if (nbHour == 0) {
            if (nbMin == 0) {
                sprintf(tmp, "%02ds", nbSec);
            } else {
                sprintf(tmp, "%02d:%02d", nbMin, nbSec);
            }
        } else {
            sprintf(tmp, "%02d:%02d:%02d", nbHour, nbMin, nbSec);
        }

    }

    return string(tmp);

}

void Run(int nbThread, std::vector<int> gridSize) {

    int nbCPUThread = nbThread;
    int nbGPUThread = 0;

    uint64_t totalThread = (uint64_t)nbCPUThread + (uint64_t)nbGPUThread;
    if(totalThread == 0) {
        ::printf("No CPU or GPU thread, exiting.\n");
        ::exit(0);
    }

    TH_PARAM *params = (TH_PARAM *)malloc(totalThread * sizeof(TH_PARAM));
    THREAD_HANDLE *thHandles = (THREAD_HANDLE *)malloc(totalThread * sizeof(THREAD_HANDLE));

    memset(params, 0,totalThread * sizeof(TH_PARAM));
    //memset(counters, 0, sizeof(counters));
    ::printf("Number of CPU thread: %d\n", nbCPUThread);

    double t0 = Timer::get_tick();

    // Lanch CPU threads
    for(int i = 0; i < nbCPUThread; i++) {
        params[i].threadId = i;
        params[i].isRunning = true;
        thHandles[i] = LaunchThread(_MySolveKeyCPU,params + i);
    }

    // Wait for end
    JoinThreads(thHandles,nbCPUThread + nbGPUThread);
    FreeHandles(thHandles,nbCPUThread + nbGPUThread);

    double t1 = Timer::get_tick();

    ::printf("\nDone: Total time %s \n" , GetTimeStr(t1-t0).c_str());

}

int main(int argc, char* argv[]) {
    //demo();
#ifdef USE_SYMMETRY
    printf("Kangaroo v" RELEASE " (with symmetry)\n");
#else
    printf("Kangaroo v" RELEASE "\n");
#endif

    // Global Init
    Timer::Init();
    rseed(Timer::getSeed32());

    // Init SecpK1
    Secp256K1 *secp = new Secp256K1();
    secp->Init();

    int a = 1;
    nbCPUThread = Timer::getCoreNumber();

    while (a < argc) {

        if(strcmp(argv[a], "-t") == 0) {
            CHECKARG("-t",1);
            nbCPUThread = getInt("nbCPUThread",argv[a]);
            a++;
        } else if(strcmp(argv[a],"-d") == 0) {
            CHECKARG("-d",1);
            dp = getInt("dpSize",argv[a]);
            a++;
        } else if (strcmp(argv[a], "-h") == 0) {
            printUsage();
        } else if(strcmp(argv[a],"-l") == 0) {

#ifdef WITHGPU
            GPUEngine::PrintCudaInfo();
#else
            printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
            exit(0);

        } else if(strcmp(argv[a],"-w") == 0) {
            CHECKARG("-w",1);
            workFile = string(argv[a]);
            a++;
        } else if(strcmp(argv[a],"-i") == 0) {
            CHECKARG("-i",1);
            iWorkFile = string(argv[a]);
            a++;
        } else if(strcmp(argv[a],"-wm") == 0) {
            CHECKARG("-wm",1);
            merge1 = string(argv[a]);
            CHECKARG("-wm",2);
            merge2 = string(argv[a]);
            a++;
            if(a<argc) {
                // classic merge
                mergeDest = string(argv[a]);
                a++;
            }
        } else if(strcmp(argv[a],"-wmdir") == 0) {
            CHECKARG("-wmdir",1);
            mergeDir = string(argv[a]);
            CHECKARG("-wmdir",2);
            mergeDest = string(argv[a]);
            a++;
        }  else if(strcmp(argv[a],"-wcheck") == 0) {
            CHECKARG("-wcheck",1);
            checkWorkFile = string(argv[a]);
            a++;
        }  else if(strcmp(argv[a],"-winfo") == 0) {
            CHECKARG("-winfo",1);
            infoFile = string(argv[a]);
            a++;
        } else if(strcmp(argv[a],"-o") == 0) {
            CHECKARG("-o",1);
            outputFile = string(argv[a]);
            a++;
        } else if(strcmp(argv[a],"-wi") == 0) {
            CHECKARG("-wi",1);
            savePeriod = getInt("savePeriod",argv[a]);
            a++;
        } else if(strcmp(argv[a],"-wt") == 0) {
            CHECKARG("-wt",1);
            wtimeout = getInt("timeout",argv[a]);
            a++;
        } else if(strcmp(argv[a],"-nt") == 0) {
            CHECKARG("-nt",1);
            ntimeout = getInt("timeout",argv[a]);
            a++;
        } else if(strcmp(argv[a],"-m") == 0) {
            CHECKARG("-m",1);
            maxStep = getDouble("maxStep",argv[a]);
            a++;
        } else if(strcmp(argv[a],"-ws") == 0) {
            a++;
            saveKangaroo = true;
        } else if(strcmp(argv[a],"-wss") == 0) {
            a++;
            saveKangarooByServer = true;
        } else if(strcmp(argv[a],"-wsplit") == 0) {
            a++;
            splitWorkFile = true;
        } else if(strcmp(argv[a],"-wpartcreate") == 0) {
            CHECKARG("-wpartcreate",1);
            workFile = string(argv[a]);
            Kangaroo::CreateEmptyPartWork(workFile);
            exit(0);
        } else if(strcmp(argv[a],"-s") == 0) {
            a++;
            serverMode = true;
        } else if(strcmp(argv[a],"-c") == 0) {
            CHECKARG("-c",1);
            serverIP = string(argv[a]);
            a++;
        } else if(strcmp(argv[a],"-sp") == 0) {
            CHECKARG("-sp",1);
            port = getInt("serverPort",argv[a]);
            a++;
        } else if(strcmp(argv[a],"-gpu") == 0) {
            gpuEnable = true;
            a++;
        } else if(strcmp(argv[a],"-gpuId") == 0) {
            CHECKARG("-gpuId",1);
            getInts("gpuId",gpuId,string(argv[a]),',');
            a++;
        } else if(strcmp(argv[a],"-g") == 0) {
            CHECKARG("-g",1);
            getInts("gridSize",gridSize,string(argv[a]),',');
            a++;
        } else if(strcmp(argv[a],"-v") == 0) {
            ::exit(0);
        } else if(strcmp(argv[a],"-check") == 0) {
            checkFlag = true;
            a++;
        } else if(a == argc - 1) {
            configFile = string(argv[a]);
            a++;
        } else {
            printf("Unexpected %s argument\n",argv[a]);
            exit(-1);
        }

    }

    if(gridSize.size() == 0) {
        for(int i = 0; i < gpuId.size(); i++) {
            gridSize.push_back(0);
            gridSize.push_back(0);
        }
    } else if(gridSize.size() != gpuId.size() * 2) {
        printf("Invalid gridSize or gpuId argument, must have coherent size\n");
        exit(-1);
    }

    if( !ParseConfigFile(configFile) )
        exit(-1);

    Run(nbCPUThread, gridSize);
/*
    Kangaroo *v = new Kangaroo(secp,dp,gpuEnable,workFile,iWorkFile,savePeriod,saveKangaroo,saveKangarooByServer,
                               maxStep,wtimeout,port,ntimeout,serverIP,outputFile,splitWorkFile);
    if(checkFlag) {
        v->Check(gpuId,gridSize);
        exit(0);
    } else {
        if(checkWorkFile.length() > 0) {
            v->CheckWorkFile(nbCPUThread,checkWorkFile);
            exit(0);
        } if(infoFile.length()>0) {
            v->WorkInfo(infoFile);
            exit(0);
        } else if(mergeDir.length() > 0) {
            v->MergeDir(mergeDir,mergeDest);
            exit(0);
        } else if(merge1.length()>0) {
            v->MergeWork(merge1,merge2,mergeDest);
            exit(0);
        } if(iWorkFile.length()>0) {
            if( !v->LoadWork(iWorkFile) )
                exit(-1);
        } else if(configFile.length()>0) {
            if( !v->ParseConfigFile(configFile) )
                exit(-1);
        } else {
            if(serverIP.length()==0) {
                ::printf("No input file to process\n");
                exit(-1);
            }
        }
        if(serverMode)
            v->RunServer();
        else
            v->Run(nbCPUThread,gpuId,gridSize);
    }
*/
    return 0;

}
