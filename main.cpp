#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>

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

    double xa = 0.0;
    double xb = 4.0;
    double ya = 0.0;
    double yb = 4.0;

    double hx = (xb - xa) / (nx - 1);
    double hy = (yb - ya) / (ny - 1);

    double tau = (nx <= 1000 && ny <= 1000) ? 0.01 : 0.001;

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
                double fij = (x == sx && y == sy) ? f(i, tau) : 0;

                u[prevIndex][index] = 2 * uc - u[prevIndex][index] + (tau * tau) * (
                        fij + ((ur - uc) * (pd + pc) + (ul - uc) * (pdl + pl)) / (2 * hx * hx) +
                                ((ut - uc) * (pl + pc) + (ud - uc) * (pdl + pd)) / (2 * hy * hy)
                        );
            }
        }

        std::swap(prevIndex, currIndex);

        //std::cout << i << std::endl;
        //std::cout << *std::max_element(u[prevIndex].begin(), u[prevIndex].end()) << std::endl;
    }

    FILE* file = std::fopen("./main.dat", "w");
    std::fwrite(&u[prevIndex][0], sizeof(double), nx * ny, file);
    std::fclose(file);

    return EXIT_SUCCESS;
}
