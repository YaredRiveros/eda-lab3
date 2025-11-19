#ifndef TRAJECTORY_SEARCH_H
#define TRAJECTORY_SEARCH_H

#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>
#include <cstddef>
#include "Common.h"

// TrajectorySearch optimizado para fuerza bruta:
// - filtro por bounding box (cota inferior barata)
// - heap (max-heap) de tamaño k para mantener k mejores
// - cálculo de Frechet con poda por cutoff (early-abandon)
// Solo se edita este archivo conforme a la instrucción.

class TrajectorySearch {
public:
    using DataItem = std::pair<TrajectoryId, Trajectory>;

    explicit TrajectorySearch(std::vector<DataItem> data)
        : data_(std::move(data))
    {
        // Preconstruimos bounding boxes para cada trayectoria para usar como filtro rápido.
        bboxes_.reserve(data_.size());
        for (const auto& item : data_) bboxes_.push_back(compute_bbox(item.second));
    }

    void build() {
        // No se necesita estructura adicional. Los precomputos ya se hicieron en el constructor.
    }

    // k-NN exacto optimizado (brute-force con filtros y poda)
    std::vector<TrajectoryId> knn_search(const Trajectory& query, std::size_t k) const {
        std::vector<TrajectoryId> result;
        const std::size_t n = data_.size();
        if (n == 0 || k == 0) return result;

        const std::size_t kk = std::min(k, n);

        // Precomputar bbox de la query
        const BBox qbox = compute_bbox(query);

        // Max-heap que mantiene (distancia_al_cuadrado, id).
        // El top es el peor entre los actuales k (mayor distancia; si empate, mayor id).
        struct HeapComp {
            bool operator()(const std::pair<double, TrajectoryId>& a,
                            const std::pair<double, TrajectoryId>& b) const {
                if (a.first != b.first) return a.first < b.first; // menor distancia -> "menor" (mejor)
                return a.second < b.second; // menor id -> "menor" (mejor)
            }
        };
        std::priority_queue<std::pair<double, TrajectoryId>,
                            std::vector<std::pair<double, TrajectoryId>>,
                            HeapComp> heap;

        const double INF = std::numeric_limits<double>::infinity();
        const double eps = 1e-12;

        // current worst (cutoff) helper lambda
        auto current_worst = [&heap, INF]() -> double {
            if (heap.empty()) return INF;
            return heap.top().first;
        };

        // Iterate over dataset
        for (std::size_t idx = 0; idx < n; ++idx) {
            const auto& item = data_[idx];
            const TrajectoryId id = item.first;

            // 1) filtro barato por bounding box (cota inferior)
            const double lb = bbox_lb_squared(qbox, bboxes_[idx]);

            double worst = current_worst();
            if (heap.size() == kk) {
                // Si la cota inferior ya es mayor que la peor distancia, saltamos.
                if (lb > worst + eps) continue;
            }

            // 2) calcular Frechet con poda por cutoff (si hay kk elementos, usamos worst como cutoff)
            double cutoff = (heap.size() < kk) ? INF : (worst - eps);
            double dist2 = frechet_distance_squared_with_cutoff(query, item.second, cutoff);

            // 3) insertar en heap si procede
            if (heap.size() < kk) {
                heap.emplace(dist2, id);
            } else {
                // si nuevo es mejor que el peor actual (según cmp de HeapComp), sustituir
                // recordar: HeapComp considera menor distancia y menor id como "mejor"
                HeapComp cmp;
                std::pair<double, TrajectoryId> cand(dist2, id);
                if (cmp(cand, heap.top())) {
                    heap.pop();
                    heap.emplace(cand);
                }
            }
        }

        // Extraer resultados del heap y ordenarlos ascendentemente por (dist, id)
        std::vector<std::pair<double, TrajectoryId>> chosen;
        chosen.reserve(heap.size());
        while (!heap.empty()) {
            chosen.push_back(heap.top());
            heap.pop();
        }

        // Orden ascendente: primero por distancia, luego por id
        std::sort(chosen.begin(), chosen.end(),
                  [eps](const std::pair<double, TrajectoryId>& a,
                        const std::pair<double, TrajectoryId>& b) {
                      if (std::fabs(a.first - b.first) > eps) return a.first < b.first;
                      return a.second < b.second;
                  });

        result.reserve(chosen.size());
        for (const auto& p : chosen) result.push_back(p.second);

        return result;
    }

private:
    // Bounding box structure
    struct BBox {
        double minx, maxx;
        double miny, maxy;
    };

    // Calcula bbox de una trayectoria (si está vacía devuelve caja degenerada en 0)
    static BBox compute_bbox(const Trajectory& T) {
        if (T.empty()) return BBox{0.0, 0.0, 0.0, 0.0};
        double minx = T[0].x, maxx = T[0].x;
        double miny = T[0].y, maxy = T[0].y;
        for (std::size_t i = 1; i < T.size(); ++i) {
            if (T[i].x < minx) minx = T[i].x;
            if (T[i].x > maxx) maxx = T[i].x;
            if (T[i].y < miny) miny = T[i].y;
            if (T[i].y > maxy) maxy = T[i].y;
        }
        return BBox{minx, maxx, miny, maxy};
    }

    // Cota inferior (squared) entre dos bounding boxes:
    // distancia minima posible entre cualquier punto de la caja A y cualquier punto de la caja B.
    static double bbox_lb_squared(const BBox& A, const BBox& B) {
        double dx = 0.0;
        if (A.maxx < B.minx) dx = B.minx - A.maxx;
        else if (B.maxx < A.minx) dx = A.minx - B.maxx;

        double dy = 0.0;
        if (A.maxy < B.miny) dy = B.miny - A.maxy;
        else if (B.maxy < A.miny) dy = A.miny - B.maxy;

        return dx * dx + dy * dy;
    }

    // Cálculo de la distancia de Fréchet al cuadrado con poda por cutoff:
    // si durante el cálculo descubrimos que la distancia será > cutoff, devolvemos
    // un valor > cutoff (por ejemplo cutoff + 1.0) para indicar abandono.
    // Implementación con memoria O(n) usando dos filas.
    static double frechet_distance_squared_with_cutoff(const Trajectory& P,
                                                       const Trajectory& Q,
                                                       double cutoff) {
        const std::size_t m = P.size();
        const std::size_t n = Q.size();

        if (m == 0 || n == 0) return 0.0;

        const double INF = cutoff + 1.0; // valor que garantiza > cutoff en caso de abandono

        // indices: i = 0..m-1, j = 0..n-1
        // mantendremos prev[j] = c[i-1][j], curr[j] = c[i][j]
        std::vector<double> prev(n, INF), curr(n, INF);

        // i = 0, j = 0
        curr[0] = squared_euclidean(P[0], Q[0]);

        // primera fila j = 1..n-1
        for (std::size_t j = 1; j < n; ++j) {
            const double d = squared_euclidean(P[0], Q[j]);
            curr[j] = std::max(curr[j - 1], d);
            // No se poda en la primera fila porque aún no se puede decidir
        }

        // Si m == 1, devolver curr[n-1] o abandono si ya > cutoff
        double min_row = curr[0];
        for (std::size_t j = 1; j < n; ++j) if (curr[j] < min_row) min_row = curr[j];
        if (m == 1) {
            if (cutoff < std::numeric_limits<double>::infinity() && min_row > cutoff) return INF;
            return curr[n - 1];
        }

        // iterar i = 1..m-1
        for (std::size_t i = 1; i < m; ++i) {
            prev.swap(curr);
            // calcular curr[0]
            const double d0 = squared_euclidean(P[i], Q[0]);
            curr[0] = std::max(prev[0], d0);

            // rastrear min en la fila para posible poda
            double row_min = curr[0];

            // j = 1..n-1
            for (std::size_t j = 1; j < n; ++j) {
                const double d = squared_euclidean(P[i], Q[j]);
                // c[i][j] = min( max(c[i-1][j], d), max(c[i-1][j-1], d), max(c[i][j-1], d) )
                double c1 = std::max(prev[j], d);
                double c2 = std::max(prev[j - 1], d);
                double c3 = std::max(curr[j - 1], d);
                double cij = c1;
                if (c2 < cij) cij = c2;
                if (c3 < cij) cij = c3;
                curr[j] = cij;
                if (cij < row_min) row_min = cij;
            }

            // poda: si el mínimo de la fila excede cutoff, entonces la entrada final
            // (c[m-1][n-1]) será >= row_min > cutoff, por lo que podemos abandonar.
            if (cutoff < std::numeric_limits<double>::infinity() && row_min > cutoff) {
                return INF;
            }
        }

        return curr[n - 1];
    }

    std::vector<DataItem> data_;
    std::vector<BBox> bboxes_;
};

#endif
