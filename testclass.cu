/*

file to test CUDA Kernel

*/
#include <fstream>
#include <stdio.h>
#include <set>
#include <vector>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

using namespace std;

int *init_zer, *res, *eq, *les;
float *init_inf, *res_f;
const float MAX_LD = 100;

__global__ void addToGridInt(int *G, int h, int b, int xl, int xu, int yl, int yu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu)
    {
        G[i * b + j] += h;
    }
}

__global__ void addToGridFloat(float *G, float ld, int b, int xl, int xu, int yl, int yu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu)
    {
        G[i * b + j] += ld;
    }
}

__global__ void updLoadLim(float *G, float v_load, float stress, int b, int xl, int xu, int yl, int yu)
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

__global__ void checkEq(int *eq, int *G, int val, int b, int xl, int xu, int yl, int yu)
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

__global__ void checkLeq(int *eq, float *G, float val, int b, int xl, int xu, int yl, int yu)
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

class Container
{
private:
    int L;
    int B;

public:
    int *h_grid;
    float *ld_lim;
    // std::set<std::pair<int, int>> pos;
    // std::vector<Item> packedI;

    Container(int L, int B);
    Container(Container &C);
    ~Container();
};

Container::Container(int l, int b)
{
    L = l;
    B = b;

    cudaMalloc((void **)&h_grid, L * B * sizeof(int));
    cudaMemcpy(h_grid, init_zer, L * B * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&ld_lim, L * B * sizeof(float));
    cudaMemcpy(ld_lim, init_inf, L * B * sizeof(float), cudaMemcpyHostToDevice);
}

Container::Container(Container &C)
{
    L = C.L;
    B = C.B;
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

    init_zer = (int *)malloc(l * b * sizeof(int));
    init_inf = (float *)malloc(l * b * sizeof(float));
    res = (int *)malloc(l * b * sizeof(int));
    res_f = (float *)malloc(l * b * sizeof(float));

    cudaMalloc((void **)&eq, l * b * sizeof(int));
    cudaMalloc((void **)&les, l * b * sizeof(int));

    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < b; j++)
        {
            init_zer[i * b + j] = 0;
            init_inf[i * b + j] = MAX_LD;
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
    free(init_inf);
    free(init_zer);
}

template <typename Option>
void print2D(Option *res, int l, int b)
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

int main()
{
    int l, b;
    printf("Enter L, B: ");
    scanf("%d %d", &l, &b);

    setup(l, b);

    Container C(l, b);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(1 + l / threadsPerBlock.x, 1 + b / threadsPerBlock.y);

    int h, xl, xu, yl, yu;
    printf("Enter h, xl, xu, yl, yu: ");
    scanf("%d %d %d %d %d", &h, &xl, &xu, &yl, &yu);
    cudaDeviceSynchronize();
    addToGridInt<<<numBlocks, threadsPerBlock>>>(C.h_grid, h, b, xl, xu, yl, yu);
    cudaDeviceSynchronize();

    cudaMemcpy(res, C.h_grid, l * b * sizeof(int), cudaMemcpyDeviceToHost);

    print2D<int>(res, l, b);

    float vl, stress;
    printf("Enter vl, stress: ");
    scanf("%f %f", &vl, &stress);
    updLoadLim<<<numBlocks, threadsPerBlock>>>(C.ld_lim, vl, stress, b, xl, xu, yl, yu);
    cudaMemcpy(res_f, C.ld_lim, l * b * sizeof(float), cudaMemcpyDeviceToHost);

    print2D<float>(res_f, l, b);

    Container C1(C);

    printf("Enter h, xl, xu, yl, yu: ");
    scanf("%d %d %d %d %d", &h, &xl, &xu, &yl, &yu);
    cudaDeviceSynchronize();
    addToGridInt<<<numBlocks, threadsPerBlock>>>(C1.h_grid, h, b, xl, xu, yl, yu);

    cudaMemcpy(res, C1.h_grid, l * b * sizeof(int), cudaMemcpyDeviceToHost);

    print2D<int>(res, l, b);

    printf("Enter vl, stress: ");
    scanf("%f %f", &vl, &stress);
    updLoadLim<<<numBlocks, threadsPerBlock>>>(C1.ld_lim, vl, stress, b, xl, xu, yl, yu);
    cudaMemcpy(res_f, C1.ld_lim, l * b * sizeof(float), cudaMemcpyDeviceToHost);

    print2D<float>(res_f, l, b);

    int val;
    printf("Enter val xl, xu, yl, yu: ");
    scanf("%d %d %d %d %d", &val, &xl, &xu, &yl, &yu);
    cudaMemcpy(eq, init_zer, l * b * sizeof(int), cudaMemcpyHostToDevice);
    checkEq<<<numBlocks, threadsPerBlock>>>(eq, C1.h_grid, val, b, xl, xu, yl, yu);
    cudaMemcpy(res, eq, l * b * sizeof(int), cudaMemcpyDeviceToHost);
    print2D<int>(res, l, b);

    thrust::device_ptr<int> dev_ptr(eq);
    int total = thrust::reduce(thrust::device, dev_ptr, dev_ptr + (l * b), 0);
    printf("%d %d", total, (xu - xl) * (yu - yl));

    float load;
    printf("Enter load, xl, xu, yl, yu: ");
    scanf("%f %d %d %d %d", &load, &xl, &xu, &yl, &yu);
    cudaMemcpy(eq, init_zer, l * b * sizeof(int), cudaMemcpyHostToDevice);
    checkLeq<<<numBlocks, threadsPerBlock>>>(eq, C1.ld_lim, load, b, xl, xu, yl, yu);
    cudaMemcpy(res, eq, l * b * sizeof(int), cudaMemcpyDeviceToHost);
    print2D<int>(res, l, b);

    total = thrust::reduce(thrust::device, dev_ptr, dev_ptr + (l * b), 0);
    printf("%d %d", total, (xu - xl) * (yu - yl));

    deAlloc();

    return 0;
}
