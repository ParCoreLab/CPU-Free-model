
#include <vector>
#include <string>

// #include <cub/util_device.cuh>
#include <sstream>
#pragma once

// namespace mersenne {

// /* Period parameters */
// // const unsigned int N          = 624;
// // const unsigned int M          = 397;
// // const unsigned int MATRIX_A   = 0x9908b0df; /* constant vector a */
// // const unsigned int UPPER_MASK = 0x80000000; /* most significant w-r bits */
// // const unsigned int LOWER_MASK = 0x7fffffff; /* least significant r bits */

// // static unsigned int mt[N];  /* the array for the state vector  */
// // static int mti = N + 1;     /* mti==N+1 means mt[N] is not initialized */

// /* initializes mt[N] with a seed */
// void init_genrand(unsigned int s);

// /* initialize by an array with array-length */
// /* init_key is the array for initializing keys */
// /* key_length is its length */
// /* slight change for C++, 2004/2/26 */
// void init_by_array(unsigned int init_key[], int key_length);


// /* generates a random number on [0,0xffffffff]-interval */
// unsigned int genrand_int32(void);


// } // namespace mersenne
static unsigned long long g_num_rand_samples = 0;

namespace mersenne {

/* Period parameters */
const unsigned int N          = 624;
const unsigned int M          = 397;
const unsigned int MATRIX_A   = 0x9908b0df; /* constant vector a */
const unsigned int UPPER_MASK = 0x80000000; /* most significant w-r bits */
const unsigned int LOWER_MASK = 0x7fffffff; /* least significant r bits */

static unsigned int mt[N];  /* the array for the state vector  */
static int mti = N + 1;     /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void init_genrand(unsigned int s)
{
    mt[0] = s & 0xffffffff;
    for (mti = 1; mti < N; mti++)
    {
        mt[mti] = (1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);

        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for mtiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */

        mt[mti] &= 0xffffffff;
        /* for >32 bit machines */
    }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned int init_key[], int key_length)
{
    int i, j, k;
    init_genrand(19650218);
    i = 1;
    j = 0;
    k = (N > key_length ? N : key_length);
    for (; k; k--)
    {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525))
            + init_key[j] + j;  /* non linear */
        mt[i] &= 0xffffffff;    /* for WORDSIZE > 32 machines */
        i++;
        j++;
        if (i >= N)
        {
            mt[0] = mt[N - 1];
            i = 1;
        }
        if (j >= key_length) j = 0;
    }
    for (k = N - 1; k; k--)
    {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i; /* non linear */
        mt[i] &= 0xffffffff; /* for WORDSIZE > 32 machines */
        i++;
        if (i >= N)
        {
            mt[0] = mt[N - 1];
            i = 1;
        }
    }

    mt[0] = 0x80000000; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned int genrand_int32(void)
{
    unsigned int y;
    static unsigned int mag01[2] = { 0x0, MATRIX_A };

    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N)
    { /* generate N words at one time */
        int kk;

        if (mti == N + 1) /* if init_genrand() has not been called, */
        init_genrand(5489); /* a defat initial seed is used */

        for (kk = 0; kk < N - M; kk++)
        {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < N - 1; kk++)
        {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }

    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}



} // namespace mersenne


struct CommandLineArgs
{

    std::vector<std::string>    keys;
    std::vector<std::string>    values;
    std::vector<std::string>    args;
#ifdef __NVCC__
    cudaDeviceProp              deviceProp;
#endif // __NVCC__
    float                       device_giga_bandwidth;
    size_t                      device_free_physmem;
    size_t                      device_total_physmem;

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv) :
        keys(10),
        values(10)
    {
        using namespace std;

        // Initialize mersenne generator
        unsigned int mersenne_init[4]=  {0x123, 0x234, 0x345, 0x456};
        mersenne::init_by_array(mersenne_init, 4);

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find('=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i)
        {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }


    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    int NumNakedArgs()
    {
        return args.size();
    }


    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void GetCmdLineArgument(int index, T &val)
    {
        using namespace std;
        if (index < args.size()) {
            istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i)
        {
            if (keys[i] == string(arg_name))
            {
                istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }


    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        using namespace std;

        if (CheckCmdLineFlag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for (int i = 0; i < keys.size(); ++i)
            {
                if (keys[i] == string(arg_name))
                {
                    string val_string(values[i]);
                    istringstream str_stream(val_string);
                    string::size_type old_pos = 0;
                    string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while ((new_pos = val_string.find(',', old_pos)) != string::npos)
                    {
                        if (new_pos != old_pos)
                        {
                            str_stream.width(new_pos - old_pos);
                            str_stream >> val;
                            vals.push_back(val);
                        }

                        // skip over comma
                        str_stream.ignore(1);
                        old_pos = new_pos + 1;
                    }

                    // Read last value
                    str_stream >> val;
                    vals.push_back(val);
                }
            }
        }
    }


    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return (int) keys.size();
    }

#ifdef __NVCC__

    /**
     * Initialize device
     */
    cudaError_t DeviceInit(int dev = -1)
    {
        cudaError_t error = cudaSuccess;

        do
        {
            int deviceCount;
            error = CubDebug(cudaGetDeviceCount(&deviceCount));
            if (error) break;

            if (deviceCount == 0) {
                fprintf(stderr, "No devices supporting CUDA.\n");
                exit(1);
            }
            if (dev < 0)
            {
                GetCmdLineArgument("device", dev);
            }
            if ((dev > deviceCount - 1) || (dev < 0))
            {
                dev = 0;
            }

            error = CubDebug(cudaSetDevice(dev));
            if (error) break;

            CubDebugExit(cudaMemGetInfo(&device_free_physmem, &device_total_physmem));

            int ptx_version;
            error = CubDebug(cub::PtxVersion(ptx_version));
            if (error) break;

            error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
            if (error) break;

            if (deviceProp.major < 1) {
                fprintf(stderr, "Device does not support CUDA.\n");
                exit(1);
            }

            device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;

            if (!CheckCmdLineFlag("quiet"))
            {
                printf(
                        "Using device %d: %s (PTX version %d, SM%d, %d SMs, "
                        "%lld free / %lld total MB physmem, "
                        "%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
                    dev,
                    deviceProp.name,
                    ptx_version,
                    deviceProp.major * 100 + deviceProp.minor * 10,
                    deviceProp.multiProcessorCount,
                    (unsigned long long) device_free_physmem / 1024 / 1024,
                    (unsigned long long) device_total_physmem / 1024 / 1024,
                    device_giga_bandwidth,
                    deviceProp.memoryClockRate,
                    (deviceProp.ECCEnabled) ? "on" : "off");
                fflush(stdout);
            }

        } while (0);

        return error;
    }

#endif // __NVCC__

};
