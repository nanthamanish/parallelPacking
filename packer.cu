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

const int TREE_WIDTH = 5;
int* init_zer, * res, * eq, * les;
float* init_inf, * res_f;
const float MAX_LD = 1;

bool DEBUG = true, DEBUG_3D = false;

char tmp;

template < typename T >
void print2D(T* res, int l, int b) {
    fstream f("res.txt", ios::out);
    f << '\t';
    for (int j = 0; j < b; j++) {
        f << j << '\t';
    }
    f << '\n';
    for (int i = 0; i < l; i++) {
        f << i << '\t';
        for (int j = 0; j < b; j++) {
            f << res[i * b + j] << "\t";
        }
        f << "\n";
    }
    f << "\n";
}

__global__ void addToGridInt(int* G, int val, int b, int xl, int xu, int yl, int yu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu) {
        G[i * b + j] += val;
    }
}

__global__ void updLoadLim(float* G, float ld_lim, float load, int b, int xl, int xu, int yl, int yu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu) {
        G[i * b + j] -= load;
        if (ld_lim < G[i * b + j])
            G[i * b + j] = ld_lim;
    }
}

__global__ void checkEq(int* eq, int* G, int val, int b, int xl, int xu, int yl, int yu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu) {
        if (G[i * b + j] == val) {
            eq[i * b + j] = 1;
        }
    }
}

__global__ void checkLeq(int* eq, float* G, float val, int b, int xl, int xu, int yl, int yu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (xl <= i && i < xu && yl <= j && j < yu) {
        if (val <= G[i * b + j]) {
            eq[i * b + j] = 1;
        }
    }
}

struct Location {
    int x, y, z;
    Location() {
        x = -1, y = -1, z = -1;
    }

    Location(int X, int Y, int Z) {
        x = X, y = Y, z = Z;
    }

    Location(Location& L) {
        x = L.x, y = L.y, z = L.z;
    }
};

struct Item {
    int id, dst;
    float wt;
    int l, b, h;
    long vol;
    int l1, b1, h1;

    int valid;

    vector < int > orientation;
    vector < float > stackload;

    int max_dim;
    float v_ld_lim;
    bool packed;
    Location pos;

    Item();
    Item(int, int, float, int, int, int, vector < int >, vector < float >);
    Item(Item& I);

    bool operator < (const Item& I) const {
        if (dst != I.dst) {
            return dst < I.dst;
        }
        float m1 = 0, m2 = 0;
        for (int i = 0; i < 3; i++) {
            m1 = max(m1, (float)orientation[i] * stackload[i]);
            m2 = max(m2, (float)I.orientation[i] * I.stackload[i]);
        }
        if (m1 != m2) {
            return m1 < m2;
        }
        return id < I.id;
    }

    float stress_load() {
        float base_area = l1 * b1;
        return wt / base_area;
    }

    void setPackDim(int lo, int bo, int ho) {
        l1 = lo;
        b1 = bo;
        h1 = ho;
    }

    void printObj() {
        printf("%d %d %.3f\n", id, dst, wt);
        printf("%d %d %d\n", l, b, h);
        printf("%d %d %d\n", orientation[0], orientation[1], orientation[2]);
        printf("%.3f %.3f %.3f, v: %3f\n", stackload[0], stackload[1], stackload[2], v_ld_lim);
        //printf("\n");
    }
};

Item::Item() {
    id = 0, dst = 0, wt = 0;
    l = 0, b = 0, h = 0;
    l1 = 0, b1 = 0, h1 = 0;
    vol = 0;
    max_dim = 0;
    v_ld_lim = 0;
    packed = 0;
    valid = 1;
    pos = Location();
}

Item::Item(int Id, int Dst, float Wt, int L, int B, int H, vector < int > ornt, vector < float > stld) {
    id = Id, dst = Dst, wt = Wt;
    l = L, b = B, h = H;
    l1 = l, b1 = b, h1 = h;

    vol = l * 1.0 * b * 1.0 * h;

    stackload = stld;
    orientation = ornt;

    max_dim = max({
        l,
        b,
        h
        });
    v_ld_lim = stackload[2];

    packed = 0;
    valid = 1;
    pos = Location();
}

Item::Item(Item& I) {
    id = I.id, dst = I.dst, wt = I.wt;
    l = I.l, b = I.b, h = I.h;
    l1 = I.l1, b1 = I.b1, h1 = I.h1;

    vol = l * 1.0 * b * 1.0 * h;

    stackload = I.stackload;
    orientation = I.orientation;

    max_dim = I.max_dim;
    v_ld_lim = I.v_ld_lim;
    packed = I.packed;
    valid = I.valid;
    pos = Location(I.pos);
}

struct Container {
    int L;
    int B;
    int H;

    float vol, util_vol;

    int* h_grid;
    float* ld_lim;

    std::set < std::pair < int, int >> corners;
    std::vector < Item > packedI;

    Container();
    Container(int, int, int);
    Container(Container& C);
    ~Container();

    Location fit(int, int, int, float);
    float volUtil() {
        return (float)util_vol / (float)vol;
    }

    int itemCount() {
        return packedI.size();
    }

    void printObj() {
        printf("%d %d %d\n", L, B, H);
        printf("%.3f %.3f\n", vol, util_vol);
        printf("\n");
    }

};

Container::Container() {
    L = 0, B = 0, H = 0;
    vol = 0, util_vol = 0;
}

Container::Container(int l, int b, int h) {
    L = l;
    B = b;
    H = h;

    vol = L * 1.0 * B * 1.0 * H;
    util_vol = 0;

    corners.insert({
        0,
        0
        });

    cudaMalloc((void**)&h_grid, L * B * sizeof(int));
    cudaMemcpy(h_grid, init_zer, L * B * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&ld_lim, L * B * sizeof(float));
    cudaMemcpy(ld_lim, init_inf, L * B * sizeof(float), cudaMemcpyHostToDevice);
}

Container::Container(Container& C) {
    L = C.L;
    B = C.B;
    H = C.H;

    vol = L * 1.0 * B * 1.0 * H;
    util_vol = C.util_vol;

    corners = C.corners;

    packedI = vector < Item >();
    for (Item& i : C.packedI) {
        packedI.push_back(Item(i));
    }

    cudaMalloc((void**)&h_grid, L * B * sizeof(int));
    cudaMemcpy(h_grid, C.h_grid, L * B * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaMalloc((void**)&ld_lim, L * B * sizeof(float));
    cudaMemcpy(ld_lim, C.ld_lim, L * B * sizeof(float), cudaMemcpyDeviceToDevice);
}

Container::~Container() {
    cudaFree(h_grid);
    cudaFree(ld_lim);
}

Location Container::fit(int l, int b, int h, float load) {
    if (DEBUG_3D) {
        cout << "Inside Fit: ";
        // cout << l << ' ' << b << ' ' << h << ' ' << load << '\n';
    }
    Location loc;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(1 + L / threadsPerBlock.x, 1 + B / threadsPerBlock.y);

    for (auto p : corners) {
        if (DEBUG_3D) {
            cout << p.first << " " << p.second << '\n';
        }

        int pos_valid = 1;
        int x = p.first, y = p.second;
        cudaMemcpy(res, h_grid, L * B * sizeof(int), cudaMemcpyDeviceToHost);
        int base = res[x * B + y];

        if (x + l > L || y + b > B || base + h > H) {
            pos_valid = 0;
            continue;
        }

        int xl = x, xu = x + l;
        int yl = y, yu = y + b;
        int area = (xu - xl) * (yu - yl);
        int total;

        if (DEBUG_3D) {
            cout << xl << ' ' << xu << ' ';
            cout << yl << ' ' << yu << ' ';
            cout << area << '\n';
            cout << "Base:" << base << '\n';
        }

        cudaMemcpy(eq, init_zer, L * B * sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        checkEq << < numBlocks, threadsPerBlock >> > (eq, h_grid, base, B, xl, xu, yl, yu);
        thrust::device_ptr < int > dev_ptr(eq);
        total = thrust::reduce(thrust::device, dev_ptr, dev_ptr + (L * B), 0);
        if (DEBUG_3D) {
            cout << "H tot:" << total << '\n';

            cudaMemcpy(res, eq, L * B * sizeof(int), cudaMemcpyDeviceToHost);
            print2D<int>(res, L, B);
            cout << "Eq of H in file: ";
            cin >> tmp;

            cudaMemcpy(res, h_grid, L * B * sizeof(int), cudaMemcpyDeviceToHost);
            print2D<int>(res, L, B);
            cout << "H in file: ";
            cin >> tmp;
        }
        if (total != area) {
            pos_valid = 0;
            continue;
        }

        cudaMemcpy(les, init_zer, L * B * sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        checkLeq << < numBlocks, threadsPerBlock >> > (les, ld_lim, load, B, xl, xu, yl, yu);
        thrust::device_ptr < int > dev_ptr2(les);
        total = thrust::reduce(thrust::device, dev_ptr2, dev_ptr2 + (L * B), 0);
        if (DEBUG_3D) {
            cout << "L tot:" << total << '\n';

            cudaMemcpy(res, les, L * B * sizeof(int), cudaMemcpyDeviceToHost);
            print2D<int>(res, L, B);
            cout << "Les of L in file: ";
            cin >> tmp;

            cudaMemcpy(res_f, ld_lim, L * B * sizeof(int), cudaMemcpyDeviceToHost);
            print2D<float>(res_f, L, B);
            cout << "L in file: ";
            cin >> tmp;
        }
        if (total != area) {
            pos_valid = 0;
            continue;
        }

        if (pos_valid) {
            loc = Location(x, y, base);
            break;
        }
    }

    if (DEBUG_3D) {
        cout << loc.x << ' ' << loc.y << ' ' << loc.z << ": ";
        cin >> tmp;
        cout << '\n';
    }
    return loc;
}

struct State {
    Container C;
    float g;

    bool operator < (const State& t) const {
        if (g != t.g) {
            return g < t.g;
        }
        return C.packedI.size() > t.C.packedI.size();
    }

    State() {
        g = 0;
        C = Container();
    }
    State(float g, Container C) {
        this->g = g;
        this->C = C;
    }

    State(State& O) {
        C = O.C;
        g = O.g;
    }

    void printObj() {
        cout << "g: " << g << '\n';
        cout << "C: ";
        C.printObj();
    }
};

void setup(int L, int B) {

    init_zer = (int*)malloc(L * B * sizeof(int));
    init_inf = (float*)malloc(L * B * sizeof(float));
    res = (int*)malloc(L * B * sizeof(int));
    res_f = (float*)malloc(L * B * sizeof(float));

    cudaMalloc((void**)&eq, L * B * sizeof(int));
    cudaMalloc((void**)&les, L * B * sizeof(int));

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < B; j++) {
            init_zer[i * B + j] = 0;
            init_inf[i * B + j] = MAX_LD;
        }
    }

    if (DEBUG) {
        cout << "set up done\n\n";
    }

    return;
}

void deAlloc() {
    cudaFree(les);
    cudaFree(eq);

    free(res_f);
    free(res);
    free(init_inf);
    free(init_zer);
}

vector < Item > allowedOrientations(Item& I) {
    // if (DEBUG) {
    //     cout << "\tInside Allowed Orientations\n";
    //     I.printObj();
    // }
    vector < Item > res(6);
    if (I.orientation[1] == 1) {
        res[0] = Item(I);
        res[0].setPackDim(I.h, I.b, I.l);
        res[0].v_ld_lim = I.stackload[0];

        res[1] = Item(I);
        res[1].setPackDim(I.b, I.h, I.l);
        res[1].v_ld_lim = I.stackload[0];
    }
    else {
        res[0].valid = 0;
        res[1].valid = 0;
    }

    if (I.orientation[1] == 1) {
        res[2] = Item(I);
        res[2].setPackDim(I.l, I.h, I.b);
        res[2].v_ld_lim = I.stackload[1];

        res[3] = Item(I);
        res[3].setPackDim(I.h, I.l, I.b);
        res[3].v_ld_lim = I.stackload[1];
    }
    else {
        res[2].valid = 0;
        res[3].valid = 0;
    }

    if (I.orientation[2] == 1) {
        res[4] = Item(I);
        res[4].setPackDim(I.l, I.b, I.h);
        res[4].v_ld_lim = I.stackload[2];

        res[5] = Item(I);
        res[5].setPackDim(I.b, I.l, I.h);
        res[5].v_ld_lim = I.stackload[2];
    }
    else {
        res[4].valid = 0;
        res[5].valid = 0;
    }

    // if (DEBUG) {
    //     cout << "\tExiting Allowed Orientations\n";
    // }

    return res;
}

void packItem(Container& C, Item& I) {
    if (DEBUG) {
        // cout << "Inside Pack Item\n";
    }
    if (I.pos.x == -1) {
        return;
    }
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(1 + C.L / threadsPerBlock.x, 1 + C.B / threadsPerBlock.y);

    // updating h_grid;
    int xl = I.pos.x, xu = I.pos.x + I.l1;
    int yl = I.pos.y, yu = I.pos.y + I.b1;
    if (DEBUG) {
        // cout << xl << ' ' << xu << ' ' << yl << ' ' << yu << '\n';
    }
    cudaDeviceSynchronize();
    addToGridInt << < numBlocks, threadsPerBlock >> > (C.h_grid, I.h1, C.B, xl, xu, yl, yu);
    // if (DEBUG) {
    //     cout << I.h1 << '\n';
    //     cudaMemcpy(res, C.h_grid, C.L * C.B * sizeof(int), cudaMemcpyDeviceToHost);
    //     print2D<int>(res, C.L, C.B);
    //     cout << "H in file: ";
    //     cin >> tmp;
    // }
    // updating ld_lim;
    float load = I.stress_load();
    float ld_lim = I.v_ld_lim;
    cudaDeviceSynchronize();
    updLoadLim << < numBlocks, threadsPerBlock >> > (C.ld_lim, ld_lim, load, C.B, xl, xu, yl, yu);
    // if (DEBUG) {
    //     cout << ld_lim << ' ' << load << '\n';
    //     cudaMemcpy(res_f, C.ld_lim, C.L * C.B * sizeof(int), cudaMemcpyDeviceToHost);
    //     print2D<float>(res_f, C.L, C.B);
    //     cout << "L in file: ";
    //     cin >> tmp;
    // }

    if (I.pos.x + I.l1 < C.L) {
        C.corners.insert({
            I.pos.x + I.l1,
            I.pos.y
            });
    }
    if (I.pos.y + I.b1 < C.B) {
        C.corners.insert({
            I.pos.y,
            I.pos.y + I.b1
            });
    }
    if (I.pos.z + I.h1 < C.H) {
        C.corners.insert({
            I.pos.x,
            I.pos.y
            });
    }

    I.packed = 1;
    C.packedI.push_back(Item(I));
    C.util_vol += I.vol;

    if (DEBUG) {
        // cout << "Exiting Pack Item\n\n";
    }

    return;
}

float greedyPack(Container C, vector < Item >& items, int start) {
    if (DEBUG) {
        // cout << "Inside Greedy\n";
    }
    for (int i = start; i >= 0; i--) {
        if (DEBUG) {
            // cout << i << '\n';
        }
        Item I = items[i];
        vector < Item > Iarr = allowedOrientations(I);

        for (int j = 0; j < 6; j++) {
            if (DEBUG) {
                // cout << j << '\n';
            }
            if (Iarr[j].valid == 0) {
                continue;
            }
            Iarr[j].pos = C.fit(Iarr[j].l1, Iarr[j].b1, Iarr[j].h1, Iarr[j].stress_load());

            if (Iarr[j].pos.x != -1) {
                packItem(C, Iarr[j]);
                break;
            }
        }
    }

    if (DEBUG) {
        cout << C.volUtil() << "\n\n";
    }

    if (DEBUG) {
        // cout << "Exiting Greedy\n\n";
    }
    return C.volUtil();
}

bool greater_pair(std::pair < float, Container >& a, std::pair < float, Container >& b) {
    if (a.first > b.first) {
        return true;
    }
    else if (b.first > a.first) {
        return false;
    }
    else {
        return a.second.itemCount() <= b.second.itemCount(); // more larger packages more likely to have been packed in packing with less overall packages
    }
}

int main(int argc, char* argv[]) {
    // Reading Inputs
    int L, B, H, n;
    fstream f;
    f.open(argv[1], ios::in);
    f >> L >> B >> H >> n;

    printf("%d %d %d %d\n", L, B, H, n);

    int id, dst;
    float wt;
    int l, b, h;
    vector < int > ornt(3);
    vector < float > stld(3);

    vector < Item > items;
    for (int i = 0; i < n; i++) {
        f >> id >> dst >> wt;
        f >> l >> b >> h;
        f >> ornt[0] >> ornt[1] >> ornt[2];
        f >> stld[0] >> stld[1] >> stld[2];
        items.push_back(Item(id, dst, wt, l, b, h, ornt, stld));
        if (DEBUG) {
            // cout << items[i].id << ' ' << items[i].v_ld_lim << '\n';
        }
    }
    f.close();

    // Setup
    setup(L, B);

    //Container Object
    Container C(L, B, H);
    C.printObj();

    std::sort(items.begin(), items.end());

    // if (DEBUG) {
    //     for (Item I : items) {
    //         I.printObj();
    //     }
    // }

    float iVol = 0;
    for (Item& i : items) {
        iVol += i.vol;
    }

    printf("%f\n\n", iVol / C.vol);

    vector < State > tree;

    float g = greedyPack(C, items, n - 1);
    State S(g, C);
    if (DEBUG) {
        //S.printObj();
    }
    tree.push_back(State(S));
    if (DEBUG) {
        cout << "Tree len: " << tree.size() << '\n';
        // tree[0].printObj();
    }

    for (int i = n - 1; i >= 0; i--) {
        if (DEBUG) {
            cout << i << '\n';
        }
        Item I = items[i];
        if (DEBUG) {
            //I.printObj();
        }

        vector < Item > Iarr = allowedOrientations(I);

        for (int k = tree.size() - 1; k >= 0; k--) {
            Container C_state(tree[k].C);

            if (DEBUG) {
                // cout << k << '\n';
                // tree[k].printObj();
            }
            cout << "Empty\n"; 
            Container C1(C_state);
            cout << C1.volUtil() << '\n';
            tree.push_back(State(greedyPack(C1, items, i - 1), C1));

            for (int j = 0; j < 6; j++) {
                if (Iarr[j].valid == 0) {
                    continue;
                }
                if (DEBUG) {
                    cout << j << '\n';
                }

                Container C_new(C_state);
                cout << C_new.volUtil() << '\n';

                if (DEBUG) {
                    cout << "Tree Size: " << tree.size() << " k: " << k << '\n';
                    cudaMemcpy(res, tree[k].C.h_grid, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                    print2D<int>(res, L, B);
                    cout << "Pre fit tree[k].C H in file: ";
                    cin >> tmp;

                    cudaMemcpy(res_f, tree[k].C.ld_lim, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                    print2D<float>(res_f, L, B);
                    cout << "Pre fit tree[k].C L in file: ";
                    cin >> tmp;

                    cudaMemcpy(res, C_new.h_grid, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                    print2D<int>(res, L, B);
                    cout << "Pre fit C_new H in file: ";
                    cin >> tmp;

                    cudaMemcpy(res_f, C_new.ld_lim, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                    print2D<float>(res_f, L, B);
                    cout << "Pre fit C_new L in file: ";
                    cin >> tmp;
                }

                DEBUG_3D = true;
                Iarr[j].pos = C_new.fit(Iarr[j].l1, Iarr[j].b1, Iarr[j].h1, Iarr[j].stress_load());
                DEBUG_3D = false;

                if (Iarr[j].pos.x != -1) {
                    if (DEBUG) {
                        for (auto t : tree) {
                            cout << t.C.volUtil() << ' ';
                        }
                        cout << '\n';
                        
                        cout << "Tree Size: " << tree.size() << " k: " << k << '\n';
                        cudaMemcpy(res, tree[k].C.h_grid, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                        print2D<int>(res, L, B);
                        cout << "Pre Pack tree[k].C H in file: ";
                        cin >> tmp;

                        cudaMemcpy(res_f, tree[k].C.ld_lim, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                        print2D<float>(res_f, L, B);
                        cout << "Pre Pack tree[k].C L in file: ";
                        cin >> tmp;
                    }
                    packItem(C_new, Iarr[j]);

                    if (DEBUG) {
                        for (auto t : tree) {
                            cout << t.C.volUtil() << ' ';
                        }
                        cout << '\n';

                        cout << "Tree Size: " << tree.size() << " k: " << k << '\n';
                        cudaMemcpy(res, tree[k].C.h_grid, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                        print2D<int>(res, L, B);
                        cout << "Post Pack tree[k].C H in file: ";
                        cin >> tmp;

                        cudaMemcpy(res_f, tree[k].C.ld_lim, L * B * sizeof(int), cudaMemcpyDeviceToHost);
                        print2D<float>(res_f, L, B);
                        cout << "Post Pack tree[k].C L in file: ";
                        cin >> tmp;
                    }

                    tree.push_back(State(greedyPack(C_new, items, i - 1), C_new));
                }
            }
            tree.erase(tree.begin() + k);
        }

        std::sort(tree.begin(), tree.end());

        cout << n - i << ": ";
        for (State s : tree) {
            string st = to_string(s.g);
            if (st.size() < 5) {
                st += "0";
            }
            cout << st << " ";
        }
        cout << "\n";

        if (tree.size() > TREE_WIDTH) {
            tree.resize(TREE_WIDTH);
        }

        cout << "After Resizing:\n";
        cout << n - i << ": ";
        for (State s : tree) {
            string st = to_string(s.g);
            if (st.size() < 5) {
                st += "0";
            }
            cout << st << " ";
        }
        cout << "\n\n";
    }

    Container resC = tree[0].C;
    printf("%f", resC.volUtil());

    deAlloc();

    return 0;
}