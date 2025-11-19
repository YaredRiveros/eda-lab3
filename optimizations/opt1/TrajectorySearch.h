#ifndef TRAJECTORY_SEARCH_H
#define TRAJECTORY_SEARCH_H

#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>
#include <numeric>
#include <cstddef>
#include "Common.h"

// TrajectorySearch: fuerza-bruta pero optimizada
// - cotas inferiores seguras (bbox, inicio, fin)
// - ordenación ligera por centroides para evaluar candidatos prometedores primero
// - heap de tamaño k + poda por cutoff
// - Frechet con early-abandon y memoria O(n)
class TrajectorySearch {
public:
    using DataItem = std::pair<TrajectoryId, Trajectory>;

    explicit TrajectorySearch(std::vector<DataItem> data)
        : data_(std::move(data))
    {
        const std::size_t n = data_.size();
        bboxes_.reserve(n);
        centroids_.reserve(n);
        starts_.reserve(n);
        ends_.reserve(n);

        for (const auto& item : data_) {
            const Trajectory& T = item.second;
            bboxes_.push_back(compute_bbox(T));
            centroids_.push_back(compute_centroid(T));
            if (!T.empty()) {
                starts_.push_back(T.front());
                ends_.push_back(T.back());
            } else {
                starts_.push_back(Point2D{0.0, 0.0});
                ends_.push_back(Point2D{0.0, 0.0});
            }
        }
    }

    // No-op (precomputos ya en constructor)
    void build() { }

    // k-NN exacto: devuelve k ids ordenados por distancia asc, desempate por id asc.
    std::vector<TrajectoryId> knn_search(const Trajectory& query, std::size_t k) const {
        std::vector<TrajectoryId> result;
        const std::size_t n = data_.size();
        if (n == 0 || k == 0) return result;
        const std::size_t kk = std::min(k, n);

        // Precomputados de la query
        const BBox qbox = compute_bbox(query);
        const Point2D qcent = compute_centroid(query);
        const Point2D qstart = query.empty() ? Point2D{0.0,0.0} : query.front();
        const Point2D qend   = query.empty() ? Point2D{0.0,0.0} : query.back();

        // Orden de indices por distancia entre centroides (heurístico, ascendente)
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) {
                      double da = squared_euclidean(centroids_[a], qcent);
                      double db = squared_euclidean(centroids_[b], qcent);
                      if (da != db) return da < db;
                      return data_[a].first < data_[b].first; // tie-break determinista
                  });

        // Max-heap que guarda los mejores k encontrados hasta el momento.
        // top() => peor candidato actual (mayor distancia; en empate mayor id)
        struct HeapComp {
            bool operator()(const std::pair<double, TrajectoryId>& a,
                            const std::pair<double, TrajectoryId>& b) const {
                if (a.first != b.first) return a.first < b.first; // mayor distancia => "mayor"
                return a.second < b.second;                       // mayor id => "mayor"
            }
        };
        std::priority_queue<std::pair<double, TrajectoryId>,
                            std::vector<std::pair<double, TrajectoryId>>,
                            HeapComp> heap;

        const double INF = std::numeric_limits<double>::infinity();
        const double eps = 1e-12;

        auto current_worst = [&heap, INF]() -> double {
            if (heap.empty()) return INF;
            return heap.top().first;
        };

        // Iterar en orden heurístico
        for (std::size_t idx_pos = 0; idx_pos < n; ++idx_pos) {
            std::size_t idx = order[idx_pos];
            const auto& item = data_[idx];
            const TrajectoryId id = item.first;

            // LB1: bounding-box LB (barato y seguro)
            double lb_bbox = bbox_lb_squared(qbox, bboxes_[idx]);

            // LB2: start / end points (seguros)
            double lb_start = squared_euclidean(qstart, starts_[idx]);
            double lb_end   = squared_euclidean(qend,   ends_[idx]);

            double lb = lb_bbox;
            if (lb_start > lb) lb = lb_start;
            if (lb_end   > lb) lb = lb_end;

            double worst = current_worst();
            if (heap.size() == kk) {
                if (lb > worst + eps) continue; // poda segura
            }

            // Si pasamos la cota, calcular Fréchet con poda usando cutoff = worst - eps
            double cutoff = (heap.size() < kk) ? INF : (worst - eps);
            double dist2 = frechet_distance_squared_with_cutoff(query, item.second, cutoff);

            // Insertar en heap si procede
            if (heap.size() < kk) {
                heap.emplace(dist2, id);
            } else {
                HeapComp cmp;
                std::pair<double, TrajectoryId> cand(dist2, id);
                if (cmp(cand, heap.top())) { // cand es "mejor" que el peor
                    heap.pop();
                    heap.emplace(cand);
                }
            }
        }

        // Extraer y ordenar resultado final: asc por (dist, id)
        std::vector<std::pair<double, TrajectoryId>> chosen;
        chosen.reserve(heap.size());
        while (!heap.empty()) {
            chosen.push_back(heap.top());
            heap.pop();
        }

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
    // -----------------------
    // Representaciones auxiliares
    // -----------------------
    struct BBox { double minx, maxx, miny, maxy; };

    static BBox compute_bbox(const Trajectory& T) {
        if (T.empty()) return BBox{0.0,0.0,0.0,0.0};
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

    static Point2D compute_centroid(const Trajectory& T) {
        if (T.empty()) return Point2D{0.0, 0.0};
        double sx = 0.0, sy = 0.0;
        for (const auto& p : T) { sx += p.x; sy += p.y; }
        double inv = 1.0 / static_cast<double>(T.size());
        return Point2D{sx * inv, sy * inv};
    }

    static double bbox_lb_squared(const BBox& A, const BBox& B) {
        double dx = 0.0;
        if (A.maxx < B.minx) dx = B.minx - A.maxx;
        else if (B.maxx < A.minx) dx = A.minx - B.maxx;
        double dy = 0.0;
        if (A.maxy < B.miny) dy = B.miny - A.maxy;
        else if (B.maxy < A.miny) dy = A.miny - B.maxy;
        return dx*dx + dy*dy;
    }

    // -----------------------
    // Frechet (al cuadrado) con pruning por cutoff
    // -----------------------
    static double frechet_distance_squared_with_cutoff(const Trajectory& P,
                                                       const Trajectory& Q,
                                                       double cutoff) {
        const std::size_t m = P.size();
        const std::size_t n = Q.size();
        if (m == 0 || n == 0) return 0.0;

        const double INF = cutoff + 1.0;
        std::vector<double> prev(n, INF), curr(n, INF);

        // primera celda
        curr[0] = squared_euclidean(P[0], Q[0]);
        for (std::size_t j = 1; j < n; ++j) {
            double d = squared_euclidean(P[0], Q[j]);
            curr[j] = std::max(curr[j-1], d);
        }

        if (m == 1) {
            if (cutoff < std::numeric_limits<double>::infinity()) {
                double minrow = curr[0];
                for (std::size_t j = 1; j < n; ++j) if (curr[j] < minrow) minrow = curr[j];
                if (minrow > cutoff) return INF;
            }
            return curr[n-1];
        }

        for (std::size_t i = 1; i < m; ++i) {
            prev.swap(curr);
            // j = 0
            curr[0] = std::max(prev[0], squared_euclidean(P[i], Q[0]));
            double row_min = curr[0];
            for (std::size_t j = 1; j < n; ++j) {
                double d = squared_euclidean(P[i], Q[j]);
                double c1 = std::max(prev[j], d);
                double c2 = std::max(prev[j-1], d);
                double c3 = std::max(curr[j-1], d);
                double cij = c1;
                if (c2 < cij) cij = c2;
                if (c3 < cij) cij = c3;
                curr[j] = cij;
                if (cij < row_min) row_min = cij;
            }
            // poda segura: si el mínimo en la fila ya excede el cutoff -> abandono
            if (cutoff < std::numeric_limits<double>::infinity() && row_min > cutoff) {
                return INF;
            }
        }
        return curr[n-1];
    }

    // -----------------------
    // Datos precomputados
    // -----------------------
    std::vector<DataItem> data_;
    std::vector<BBox> bboxes_;
    std::vector<Point2D> centroids_;
    std::vector<Point2D> starts_;
    std::vector<Point2D> ends_;
};

#endif
