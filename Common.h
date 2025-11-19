#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

struct Point2D {
    double x;
    double y;
};

using TrajectoryId = std::int64_t;
using Trajectory   = std::vector<Point2D>;

// Distancia Euclídea al cuadrado entre dos puntos
inline double squared_euclidean(const Point2D& a, const Point2D& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Distancia de Fréchet
inline double frechet_distance_squared(const Trajectory& P, const Trajectory& Q) {
    const std::size_t m = P.size();
    const std::size_t n = Q.size();

    if (m == 0 || n == 0) {
        return 0.0;
    }

    // Matriz de tamaño m*n almacenada en un vector 1D
    std::vector<double> c(m * n, 0.0);

    auto idx = [n](std::size_t i, std::size_t j) {
        return i * n + j;
    };

    // c[0][0]
    c[idx(0, 0)] = squared_euclidean(P[0], Q[0]);

    // Primera columna
    for (std::size_t i = 1; i < m; ++i) {
        const double d = squared_euclidean(P[i], Q[0]);
        c[idx(i, 0)] = std::max(c[idx(i - 1, 0)], d);
    }

    // Primera fila
    for (std::size_t j = 1; j < n; ++j) {
        const double d = squared_euclidean(P[0], Q[j]);
        c[idx(0, j)] = std::max(c[idx(0, j - 1)], d);
    }

    // Resto de la matriz
    for (std::size_t i = 1; i < m; ++i) {
        for (std::size_t j = 1; j < n; ++j) {
            const double d = squared_euclidean(P[i], Q[j]);
            const double c1 = std::max(c[idx(i - 1, j)],   d);
            const double c2 = std::max(c[idx(i - 1, j - 1)], d);
            const double c3 = std::max(c[idx(i, j - 1)],   d);
            c[idx(i, j)] = std::min(std::min(c1, c2), c3);
        }
    }

    return c[idx(m - 1, n - 1)];
}

// Distancia de Fréchet (raíz cuadrada de la distancia al cuadrado)
inline double frechet_distance(const Trajectory& P, const Trajectory& Q) {
    return std::sqrt(frechet_distance_squared(P, Q));
}
