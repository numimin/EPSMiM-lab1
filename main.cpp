#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <chrono>

double f(int n, double tau) {
    constexpr double f0 = 1.0;
    constexpr double t0 = 1.5;
    constexpr double gamma = 4.0;
    const double a = 2 * M_PI * f0 * (n * tau - t0);
    return std::exp((- a * a) / (gamma * gamma)) * std::sin(a);
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage: ./main Nx Ny Nt" << std::endl;
        return EXIT_FAILURE;
    }

    int nx = std::stoi(argv[1]);
    int ny = std::stoi(argv[2]);
    int nt = std::stoi(argv[3]);
    int sx = std::stoi(argv[4]);
    int sy = std::stoi(argv[5]);

    const auto start = std::chrono::high_resolution_clock::now();

    constexpr double xa = 0.0;
    constexpr double xb = 4.0;
    constexpr double ya = 0.0;
    constexpr double yb = 4.0;

    const double hx = (xb - xa) / (nx - 1);
    const double hy = (yb - ya) / (ny - 1);

    const double hxrec = 1 / (2 * hx * hx);
    const double hyrec = 1 / (2 * hy * hy);

    const double tau = (nx <= 1000 && ny <= 1000) ? 0.01 : 0.001;

    int prevIndex = 0;
    int currIndex = 1;

    std::vector<double> u[] = {std::vector<double>(nx * ny, 0), std::vector<double>(nx * ny, 0)};
    std::vector<double> p(nx * ny, 0);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            if (x < nx / 2) {
                p[y * nx + x] = 0.1 * 0.1;
            } else {
                p[y * nx + x] = 0.2 * 0.2;
            }
        }
    }

    for (int i = 0; i < nt; i++) {
        double maxElement = 0;
        for (int y = 1; y < ny - 1; y++) {
            for (int x = 1; x < nx - 1; x++) {
                int index = y * nx + x;
                double uc = u[currIndex][index];
                double ur = u[currIndex][y * nx + x + 1];
                double ul = u[currIndex][y * nx + x - 1];
                double ut = u[currIndex][(y + 1) * nx + x];
                double ud = u[currIndex][(y - 1) * nx + x];
                double pc = p[index];
                double pd = p[(y - 1) * nx + x];
                double pdl = p[(y - 1) * nx + x - 1];
                double pl =p[y * nx + x - 1];
                double fij = 0;
                if (x == sx && y == sy) {
                    fij = f(i, tau);
                }

                u[prevIndex][index] = 2 * uc - u[prevIndex][index] + (tau * tau) * (
                        fij + ((ur - uc) * (pd + pc) + (ul - uc) * (pdl + pl)) * hxrec +
                                ((ut - uc) * (pl + pc) + (ud - uc) * (pdl + pd)) * hyrec
                        );

                if (u[prevIndex][index] > maxElement) {
                    maxElement = u[prevIndex][index];
                }
            }
        }

        std::swap(prevIndex, currIndex);

        std::cout << i << std::endl;
        std::cout << maxElement << std::endl;    
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << duration.count() / 1000.0 << "s" << std::endl;

    FILE* file = std::fopen("./main.dat", "w");
    std::fwrite(&u[prevIndex][0], sizeof(double), nx * ny, file);
    std::fclose(file);

    return EXIT_SUCCESS;
}