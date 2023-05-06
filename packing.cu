/*

Packing File

*/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <vector>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

using namespace std;

int *cpu_h_grid, *res, *eq, *les;
float *cpu_ld_lim, *res_f;
const float MAX_LD = 100;

__global__ void add_int_to_mat(int *G, int h, int b, int xl, int xu, int yl, int yu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu)
    {
        G[i * b + j] += h;
    }
}

__global__ void add_float_to_mat(float *G, float ld, int b, int xl, int xu, int yl, int yu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu)
    {
        G[i * b + j] += ld;
    }
}

__global__ void update_lim(float *G, float v_load, float stress, int b, int xl, int xu, int yl, int yu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu)
    {
        G[i * b + j] -= stress;
        if (v_load < G[i * b + j])
            G[i * b + j] = v_load;
    }
}

__global__ void make_check_eq(int *eq, int *G, int val, int b, int xl, int xu, int yl, int yu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu)
    {
        if (G[i * b + j] == val)
        {
            eq[i * b + j] = 1;
        }
    }
}

__global__ void make_check_les(int *eq, float *G, float val, int b, int xl, int xu, int yl, int yu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu)
    {
        if (val <= G[i * b + j])
        {
            eq[i * b + j] = 1;
        }
    }
}

struct Location
{
    int x, y, z;
    Location()
    {
        x = -1, y = -1, z = -1;
    }

    Location(Location &L)
    {
        x = L.x, y = L.y, z = L.z;
    }
};

struct Item
{
    int id, dst;
    float wt;
    int l, b, h;
    long vol;
    int l1, b1, h1;

    vector<int> orientation;
    vector<float> stackload;

    int max_dim, v_ld_lim;
    bool packed;
    Location pos;

    Item(int, int, float, int, int, int, vector<int>, vector<float>);
    Item(Item &I);

    bool operator < (const Item& I) const
    {
        if(dst != I.dst) {
            return dst < I.dst;
        }
        float m1 = 0, m2 = 0;
        for(int i = 0; i < 3; i++) {
            m1 = max(m1, (float)orientation[i] * stackload[i]);
            m2 = max(m2, (float)I.orientation[i] * I.stackload[i]);
        }
        return m1 < m2;
    }
};

Item::Item(int Id, int Dst, float Wt, int L, int B, int H, vector<int> ornt, vector<float> stld)
{
    id = Id, dst = Dst, wt = wt;
    l = L, b = B, h = H;
    l1 = l, b1 = b, h1 = h;

    vol = l * 1L * b * 1L * h;

    stackload = stld;
    orientation = ornt;

    max_dim = max({l, b, h});
    v_ld_lim = stackload[2];

    packed = false;
    pos = Location();
}

Item::Item(Item &I)
{
    id = I.id, dst = I.dst, wt = I.wt;
    l = I.l, b = I.b, h = I.h;
    l1 = I.l1, b1 = I.b1, h1 = I.h1;
    
    vol = l * 1L * b * 1L * h;

    stackload = I.stackload;
    orientation = I.orientation;

    max_dim = I.max_dim;
    v_ld_lim = I.v_ld_lim;
    packed = I.packed;
    pos = Location(I.pos);
}

struct Container
{
    int L;
    int B;
    int H;

    long vol;

    int *h_grid;
    float *ld_lim;

    std::set<std::pair<int, int>> pos;
    std::vector<Item> packedI;

    Container(int, int, int);
    Container(Container &C);
    ~Container();
};

Container::Container(int l, int b, int h)
{
    L = l;
    B = b;
    H = h;

    vol = L * 1l * B * 1l * H;    

    pos.insert({0, 0});

    cudaMalloc((void **)&h_grid, L * B * sizeof(int));
    cudaMemcpy(h_grid, cpu_h_grid, L * B * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&ld_lim, L * B * sizeof(float));
    cudaMemcpy(ld_lim, cpu_ld_lim, L * B * sizeof(float), cudaMemcpyHostToDevice);
}

Container::Container(Container &C)
{
    L = C.L;
    B = C.B;
    H = C.H;

    vol = L * 1l * B * 1l * H;

    pos = C.pos;

    packedI = vector<Item>();
    for(Item& i : C.packedI) {
        packedI.push_back(Item(i));
    }

    cudaMalloc((void **)&h_grid, L * B * sizeof(int));
    cudaMemcpy(h_grid, C.h_grid, L * B * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaMalloc((void **)&ld_lim, L * B * sizeof(float));
    cudaMemcpy(ld_lim, C.ld_lim, L * B * sizeof(float), cudaMemcpyDeviceToDevice);
}

Container::~Container()
{
    cudaFree(h_grid);
    cudaFree(ld_lim);
}

void setup(int l, int b)
{
    fstream f;
    f.open("res1.txt", ios::out);
    f << "";
    f.close();

    cpu_h_grid = (int *)malloc(l * b * sizeof(int));
    cpu_ld_lim = (float *)malloc(l * b * sizeof(float));
    res = (int *)malloc(l * b * sizeof(int));
    res_f = (float *)malloc(l * b * sizeof(float));

    cudaMalloc((void **)&eq, l * b * sizeof(int));
    cudaMalloc((void **)&les, l * b * sizeof(int));

    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < b; j++)
        {
            cpu_h_grid[i * b + j] = 0;
            cpu_ld_lim[i * b + j] = MAX_LD;
        }
    }

    return;
}

void deAlloc()
{
    cudaFree(les);
    cudaFree(eq);

    free(res_f);
    free(res);
    free(cpu_ld_lim);
    free(cpu_h_grid);
}

template <typename T>
void print_2D(T *res, int l, int b)
{
    fstream f;
    f.open("res1.txt", ios::app);
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < b; j++)
        {
            f << res[i * b + j] << "\t";
        }
        f << "\n";
    }
    f << "\n";
    f.close();
}

int main(int argc, char *argv[])
{
    int L, B, H, n;
    fstream f;
    f.open(argv[1], ios::in);
    f >> L >> B >> H >> n;

    printf("%d %d %d %d\n", L, B, H, n);

    int id, dst;
    float wt;
    int l, b, h;
    vector<int> ornt(3);
    vector<float> stld(3);

    vector<Item> I;
    for (int i = 0; i < n; i++)
    {
        f >> id >> dst >> wt;
        f >> l >> b >> h;
        f >> ornt[0] >> ornt[1] >> ornt[2];
        f >> stld[0] >> stld[1] >> stld[2];
        I.push_back(Item(id, dst, wt, l, b, h, ornt, stld));
    }
    f.close();

    setup(L, B);
    Container C(L, B, H);

    std::sort(I.begin(), I.end());


    float iVol = 0;
    for(Item & i : I) {
        // printf("%d ", i.id);
        iVol += i.vol;
    }
    // printf("\n");

    float cVol = C.vol;
    printf("%f\n", iVol/cVol);

    

    deAlloc();

    return 0;
}
