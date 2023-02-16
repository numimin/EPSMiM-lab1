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
        std::cout << "Usage: ./main Nx Ny Nt Sx Sy" << std::endl;
        return EXIT_FAILURE;
    }

    const int nx = std::stoi(argv[1]);
    const int ny = std::stoi(argv[2]);
    const int nt = std::stoi(argv[3]);
    const int sx = std::stoi(argv[4]);
    const int sy = std::stoi(argv[5]);

    const auto start = std::chrono::high_resolution_clock::now();

    const double xa = 0.0;
    const double xb = 4.0;
    const double ya = 0.0;
    const double yb = 4.0;

    const double hx = (xb - xa) / (nx - 1);
    const double hy = (yb - ya) / (ny - 1);

    const double hxrec = 1 / (2 * hx * hx);
    const double hyrec = 1 / (2 * hy * hy);

    const double tau = (nx <= 1000 && ny <= 1000) ? 0.01 : 0.001;
    const double tau2 = tau * tau;

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
                const int index = y * nx + x;
                const double uc = u[currIndex][index];
                const double ur = u[currIndex][index + 1];
                const double ul = u[currIndex][index - 1];
                const double ut = u[currIndex][index + nx];
                const double ud = u[currIndex][index - nx];
                const double pc = p[index];
                const double pd = p[index - nx];
                const double pdl = p[index - nx - 1];
                const double pl = p[index - 1];

                u[prevIndex][index] = 2 * uc - u[prevIndex][index] + tau2 * (
                        ((ur - uc) * (pd + pc) + (ul - uc) * (pdl + pl)) * hxrec +
                                ((ut - uc) * (pl + pc) + (ud - uc) * (pdl + pd)) * hyrec
                        );
            }
        }

        u[prevIndex][sy * nx + sx] += tau2 * f(i, tau);

        std::swap(prevIndex, currIndex);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << duration.count() / 1000.0 << "s" << std::endl;

    FILE* file = std::fopen("./main.dat", "w");
    std::fwrite(&u[currIndex][0], sizeof(double), nx * ny, file);
    std::fclose(file);

    return EXIT_SUCCESS;
}
