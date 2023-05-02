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
#include <random>
#include <iostream>

using namespace std;

#define CHECKARG(opt,n) if(a>=argc-1) {::printf(opt " missing argument #%d\n",n);exit(0);} else {a++;}

// ------------------------------------------------------------------------------------------

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
#define CPU_GRP_SIZE 0xFFFFFF

typedef  struct {
    Int rangeStart;
    Int rangeEnd;
    Int CurrentPos;
    uint64_t step;
    bool found;
    pthread_mutex_t  tMutex;
} TASK;

TASK gTask;

Int GetNextTask() {

    LOCK(gTask.tMutex)

    Int t;
    t.SetInt32(0);
    if (gTask.CurrentPos.IsLower(&gTask.rangeEnd)) {
        t = gTask.CurrentPos;
        gTask.CurrentPos.Add(gTask.step + 1);
    }

    UNLOCK(gTask.tMutex)

    return t;
}

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

    gTask.rangeStart.SetBase16((char *)lines[0].c_str());
    gTask.CurrentPos.SetBase16((char *)lines[0].c_str());
    gTask.rangeEnd.SetBase16((char *)lines[1].c_str());
    gTask.step = CPU_GRP_SIZE;
    gTask.found = false;

    for(int i=2;i<(int)lines.size();i++) {
        keysToSearch.push_back(lines[i]);
    }

    ::printf("Start:%s\n",gTask.rangeStart.GetBase16().c_str());
    ::printf("Stop :%s\n",gTask.rangeEnd.GetBase16().c_str());
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

bool Output(Int *pk) {

    FILE* f = stdout;
    bool needToClose = false;

    if(outputFile.length() > 0) {
        f = fopen(outputFile.c_str(),"a");
        if(f == NULL) {
            printf("Cannot open %s for writing\n",outputFile.c_str());
            f = stdout;
        }
        else {
            needToClose = true;
        }
    }

    if(!needToClose)
        ::printf("\n");

    Secp256K1 *secp = new Secp256K1();
    secp->Init();

    Point PR = secp->ComputePublicKey(pk);

    //::fprintf(f,"Key#%2d [%d%c]Addr:  0x%s \n",keyIdx,sType,sInfo,keysToSearch[0]);
    ::fprintf(f,"       Priv: 0x%s \n",pk->GetBase16().c_str());
    ::fprintf(f,"       Pub : 0x%s \n",secp->GetPublicKeyHex(true,PR).c_str());
    hash160_t h1;
    secp->GetHash160(P2PKH, true, PR, h1.i8);
    ::fprintf(f,"       Add : %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());

    if(needToClose)
        fclose(f);

    return true;
}

void MySolveKeyCPU(TH_PARAM *ph) {

    vector<ITEM> dps;
    double lastSent = 0;

    // Global init
    int thId = ph->threadId;

    // Create Kangaroos
    ph->nbKangaroo = gTask.step;

    Int pk;
    pk.SetInt32(0);
    Secp256K1 *secp = new Secp256K1();
    secp->Init();

    pk = GetNextTask();
    //pk.AddOne();

    while (!gTask.found) //!pk.IsZero()
    {
        uint64_t step = gTask.step;
        Int last = pk;
        last.Add(step);
        printf("SolveKeyCPU Thread %d : %s - %s\n", thId, pk.GetBase16().c_str(), last.GetBase16().c_str());

        for(uint64_t i = 0; i < step; i++, pk.AddOne()) {
            Point P = secp->ComputePublicKey(&pk);

            hash160_t h1;
            secp->GetHash160(P2PKH, true, P, h1.i8);
            std::string addr = secp->GetAddress(P2PKH, true, h1.i8).c_str();

            if (addr.compare(keysToSearch[0]) == 0) {
                printf("       Priv: 0x%s \n",pk.GetBase16().c_str());
                printf("       Pub : 0x%s \n",secp->GetPublicKeyHex(true,P).c_str());
                printf("       Addr: %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());
                Output(&pk);
                gTask.found = true;
            }
        }

        pk = GetNextTask();
    }

    printf("SolveKeyCPU Thread %d exit\n", thId);
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

typedef unsigned long long cycles_t;
inline cycles_t currentcycles() {
    cycles_t result;
    __asm__ __volatile__ ("rdtsc" : "=A" (result));
    return result;
}

void worker(char * start, char * target, int step) {
    Int pk;
    pk.SetBase16(start);
    Secp256K1 *secp = new Secp256K1();
    secp->Init();
    hash160_t h1;
    printf("       Priv Sart: %064x \n", pk.GetBase16().c_str());
    for(int i = 0; i < 1<<step; i++)
    {
        Point P = secp->ComputePublicKey(&pk);
        secp->GetHash160(P2PKH, true, P, h1.i8);
        std::string addr = secp->GetAddress(P2PKH, true, h1.i8).c_str();
        if (addr.compare(target) == 0) {
            printf("       Priv: 0x%s \n",pk.GetBase16().c_str());
            printf("       Pub : 0x%s \n",secp->GetPublicKeyHex(true,P).c_str());
            printf("       Addr: %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());
            Output(&pk);
            gTask.found = true;
        }
        pk.AddOne();
    }
    printf("       Priv End : %64x \n",pk.GetBase16().c_str());
}

void bigIntRand(int index, int step) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int64_t> dis(0, INT64_MAX);
    uint64_t random_num = dis(gen);
    random_num = random_num >> (64 + step - index);

    Int rnd = Int(random_num);
    rnd.ShiftL(step);
    //::printf("%s\n", rnd.GetBase16().c_str());

    Int pos = Int(int64_t(1));
    pos.ShiftL(index-1);
    //::printf("%s\n", pos.GetBase16().c_str());
    pos.Add(&pos, &rnd);
    ::printf("\n%s\n", pos.GetBase16().c_str());
    double T0 = Timer::get_tick();
    worker(pos.GetBase16().data(), "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so", step);
    double T1 = Timer::get_tick();
    ::printf("\nDone: Total time %s, Step %d \n" , GetTimeStr(T1-T0).c_str(), step);
}

void demo2() {
    char * lines = "5B3F38AF935A3640D158E871CE6E9666DB862636383386EE510F18CCC3BD72EB";
    Int pk;
    pk.SetInt32(0);
    Secp256K1 *secp = new Secp256K1();
    secp->Init();
    pk.SetBase16(lines);

    double t1;
    double p5, p1, p2, p3, p4;
    p1 = p2 = p3 = p4 = p5 = 0;

    hash160_t h1;
    double T0 = Timer::get_tick();

    for(int i = 0; i < 1<<24; i++)
    {
        //t1 = Timer::get_tick();
        Point P = secp->ComputePublicKey(&pk);
        //p2 += Timer::get_tick() - t1;

        //t1 = Timer::get_tick();
        secp->GetHash160(P2PKH, true, P, h1.i8);
        //p3 += Timer::get_tick() - t1;

        //t1 = Timer::get_tick();
        std::string addr = secp->GetAddress(P2PKH, true, h1.i8).c_str();
        //p4 += Timer::get_tick() - t1;

        //t1 = Timer::get_tick();
        if (addr.compare("16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN") == 0) {
            printf("       Priv: 0x%s \n",pk.GetBase16().c_str());
            printf("       Pub : 0x%s \n",secp->GetPublicKeyHex(true,P).c_str());
            printf("       Addr: %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());
            Output(&pk);
            gTask.found = true;
        }
        //p5 += Timer::get_tick() - t1;

        //t1 = Timer::get_tick();
        pk.AddOne();
        //p1 += Timer::get_tick() - t1;
    }

    double T1 = Timer::get_tick();
    ::printf("\nDone: Total time %s \n" , GetTimeStr(T1-T0).c_str());
    printf("       Priv: 0x%s \n",pk.GetBase16().c_str());

    //::printf("AddOne     : %s\n", GetTimeStr(p1).c_str());
    //::printf("PublicKey  : %s\n", GetTimeStr(p2).c_str());
    //::printf("GetHash160 : %s\n", GetTimeStr(p3).c_str());
    //::printf("GetAddress : %s\n", GetTimeStr(p4).c_str());
    //::printf("compare    : %s\n", GetTimeStr(p5).c_str());
}

void demo3() {

    //char * lines = "5B3F38AF935A3640D158E871CE6E9666DB862636383386EE510F18CCC3BD72EB";
    Int pk;
    pk.SetInt32(1);
    Secp256K1 *secp = new Secp256K1();
    secp->Init();
    //pk.SetBase16(lines);

    hash160_t h1;
    Point P = secp->ComputePublicKey(&pk);
    secp->GetHash160(P2PKH, true, P, h1.i8);
    std::string addr = secp->GetAddress(P2PKH, true, h1.i8).c_str();
    printf("       Priv: 0x%s \n",pk.GetBase16().c_str());
    printf("       Pub : 0x%s \n",secp->GetPublicKeyHex(true,P).c_str());
    printf("       Addr: %s\n", secp->GetAddress(P2PKH, true, h1.i8).c_str());

    //------------------------------

    char * lines = "0000000000000000000000000000000000000000000000000000000000000001";
    Int pk2;
    pk2.SetInt32(0);
    Secp256K1 *secp2 = new Secp256K1();
    secp2->Init();
    pk2.SetBase16(lines);

    hash160_t h2;
    Point P2 = secp2->ComputePublicKey(&pk2);
    secp2->GetHash160(P2PKH, true, P2, h2.i8);
    std::string addr2 = secp2->GetAddress(P2PKH, true, h2.i8).c_str();
    printf("       Priv: 0x%s \n",pk2.GetBase16().c_str());
    printf("       Pub : 0x%s \n",secp2->GetPublicKeyHex(true,P2).c_str());
    printf("       Addr: %s\n", secp2->GetAddress(P2PKH, true, h2.i8).c_str());

    //------------------------------
    char * lines3 = "1";
    Int pk3;
    pk3.SetInt32(1);
    Secp256K1 *secp3 = new Secp256K1();
    secp3->Init();
    pk3.SetBase16(lines3);

    hash160_t h3;
    Point P3 = secp3->ComputePublicKey(&pk3);
    secp3->GetHash160(P2PKH, true, P3, h3.i8);
    std::string addr3 = secp3->GetAddress(P2PKH, true, h3.i8).c_str();
    printf("       Priv: 0x%s \n",pk3.GetBase16().c_str());
    printf("       Pub : 0x%s \n",secp3->GetPublicKeyHex(true,P3).c_str());
    printf("       Addr: %s\n", secp3->GetAddress(P2PKH, true, h3.i8).c_str());
}

int main(int argc, char* argv[]) {
    demo3();
    bigIntRand(66, 24);
    demo2();
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
        }else if (strcmp(argv[a], "-demo") == 0) {
            demo2();
            return 0 ;
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

