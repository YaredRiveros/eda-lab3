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
#include <cstdint>
#include <functional>
#include "Common.h"

// Versión corregida de TrajectorySearch:
// - Se evita usar rep-Hausdorff como cota inferior para poda (era incorrecto).
// - rep-Hausdorff se conserva sólo como heurístico de ordenación (candidate set).
// - Poda segura: bbox, start, end.
// - Heap k, Frechet con early-abandon, buffers reusables.
// - Garantía: búsqueda exacta k-NN (desempate por id ascendente).

class TrajectorySearch {
public:
    using DataItem = std::pair<TrajectoryId, Trajectory>;

    explicit TrajectorySearch(std::vector<DataItem> data, std::size_t candidate_factor = 8)
        : data_(std::move(data)), candidate_factor_(candidate_factor)
    {
        const std::size_t n = data_.size();
        if (n == 0) {
            L_ = 0;
            return;
        }

        // Asumimos longitud fija L (enunciado).
        L_ = data_[0].second.size();

        bboxes_.reserve(n);
        centroids_.reserve(n);
        starts_.reserve(n);
        ends_.reserve(n);
        reps_.reserve(n);
        ids_.reserve(n);

        for (const auto& item : data_) {
            ids_.push_back(item.first);
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
            reps_.push_back(compute_representatives(T));
        }

        // Reserve workspaces reutilizables para Fréchet.
        if (L_ > 0) {
            workspace_prev_.assign(L_, std::numeric_limits<double>::infinity());
            workspace_curr_.assign(L_, std::numeric_limits<double>::infinity());
        }
    }

    void build() { }

    // k-NN exacto: devuelve k ids ordenados por (dist asc, id asc)
    std::vector<TrajectoryId> knn_search(const Trajectory& query, std::size_t k) const {
        std::vector<TrajectoryId> result;
        const std::size_t n = data_.size();
        if (n == 0 || k == 0) return result;

        const std::size_t kk = std::min(k, n);
        const double eps = 1e-12;
        const double INF = std::numeric_limits<double>::infinity();

        // Precomputados query
        const BBox qbox = compute_bbox(query);
        const Point2D qcent = compute_centroid(query);
        const Point2D qstart = query.empty() ? Point2D{0.0,0.0} : query.front();
        const Point2D qend   = query.empty() ? Point2D{0.0,0.0} : query.back();
        const std::vector<Point2D> qreps = compute_representatives(query);

        // ----------------------------
        // Candidate set: ordenamos por rep-Hausdorff (heurístico, NO PODA).
        // rep_hausdorff puede ser mayor que la Fréchet real, por eso NO lo usamos para poda.
        // ----------------------------
        std::vector<std::pair<double, std::size_t>> rep_dists;
        rep_dists.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            double d = rep_hausdorff_squared(qreps, reps_[i]);
            rep_dists.emplace_back(d, i);
        }

        const std::size_t cand_size = std::min<std::size_t>(n, candidate_factor_ * kk);
        if (cand_size == n) {
            std::sort(rep_dists.begin(), rep_dists.end(),
                      [](const auto& a, const auto& b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });
        } else {
            std::partial_sort(rep_dists.begin(), rep_dists.begin() + cand_size, rep_dists.end(),
                              [](const auto& a, const auto& b) {
                                  if (a.first != b.first) return a.first < b.first;
                                  return a.second < b.second;
                              });
        }

        std::vector<std::size_t> candidates;
        candidates.reserve(cand_size);
        for (std::size_t i = 0; i < cand_size; ++i) candidates.push_back(rep_dists[i].second);

        // ----------------------------
        // Heap (max-heap) para mantener mejores k
        // top() = peor candidato actual (mayor distancia; si empate mayor id)
        // ----------------------------
        struct HeapComp {
            bool operator()(const std::pair<double, TrajectoryId>& a,
                            const std::pair<double, TrajectoryId>& b) const {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            }
        };
        std::priority_queue<std::pair<double, TrajectoryId>,
                            std::vector<std::pair<double, TrajectoryId>>,
                            HeapComp> heap;

        auto current_worst = [&heap, INF]() -> double {
            if (heap.empty()) return INF;
            return heap.top().first;
        };

        // ----------------------------
        // Evaluar candidate set primero (reduce cutoff rápido)
        // ----------------------------
        for (std::size_t idx : candidates) {
            const auto& item = data_[idx];
            const TrajectoryId id = item.first;

            // PODA SEGURA: solo LBs que sabemos <= Fréchet(full)
            double lb_bbox  = bbox_lb_squared(qbox, bboxes_[idx]);
            double lb_start = squared_euclidean(qstart, starts_[idx]);
            double lb_end   = squared_euclidean(qend,   ends_[idx]);

            double lb = lb_bbox;
            if (lb_start > lb) lb = lb_start;
            if (lb_end   > lb) lb = lb_end;

            double worst = current_worst();
            if (heap.size() == kk && lb > worst + eps) continue; // poda segura

            double cutoff = (heap.size() < kk) ? INF : (worst - eps);
            double dist2 = frechet_distance_squared_with_cutoff(query, item.second, cutoff);

            if (heap.size() < kk) {
                heap.emplace(dist2, id);
            } else {
                HeapComp cmp;
                std::pair<double, TrajectoryId> candp(dist2, id);
                if (cmp(candp, heap.top())) {
                    heap.pop();
                    heap.emplace(candp);
                }
            }
        }

        // ----------------------------
        // Evaluar resto de items, con las mismas LBs seguras
        // ----------------------------
        std::vector<char> in_cand(n, 0);
        for (std::size_t idx : candidates) in_cand[idx] = 1;

        for (std::size_t idx = 0; idx < n; ++idx) {
            if (in_cand[idx]) continue;
            const auto& item = data_[idx];
            const TrajectoryId id = item.first;

            double lb_bbox  = bbox_lb_squared(qbox, bboxes_[idx]);
            double lb_start = squared_euclidean(qstart, starts_[idx]);
            double lb_end   = squared_euclidean(qend,   ends_[idx]);

            double lb = lb_bbox;
            if (lb_start > lb) lb = lb_start;
            if (lb_end   > lb) lb = lb_end;

            double worst = current_worst();
            if (heap.size() == kk && lb > worst + eps) continue; // poda segura

            double cutoff = (heap.size() < kk) ? INF : (worst - eps);
            double dist2 = frechet_distance_squared_with_cutoff(query, item.second, cutoff);

            if (heap.size() < kk) {
                heap.emplace(dist2, id);
            } else {
                HeapComp cmp;
                std::pair<double, TrajectoryId> candp(dist2, id);
                if (cmp(candp, heap.top())) {
                    heap.pop();
                    heap.emplace(candp);
                }
            }
        }

        // ----------------------------
        // Extraer resultados y ordenar asc por (dist, id)
        // ----------------------------
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
    // Auxiliares y precomputos
    // -----------------------
    struct BBox { double minx, maxx, miny, maxy; };

    static BBox compute_bbox(const Trajectory& T) {
        if (T.empty()) return BBox{0.0,0.0,0.0,0.0};
        double minx = T[0].x, maxx = T[0].x;
        double miny = T[0].y, maxy = T[0].y;
        for (std::size_t i = 1; i < T.size(); ++i) {
            const double x = T[i].x;
            const double y = T[i].y;
            if (x < minx) minx = x;
            if (x > maxx) maxx = x;
            if (y < miny) miny = y;
            if (y > maxy) maxy = y;
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

    // Representatives: {first, mid, last, centroid}
    static std::vector<Point2D> compute_representatives(const Trajectory& T) {
        std::vector<Point2D> reps;
        if (T.empty()) return reps;
        reps.push_back(T.front());
        reps.push_back(T[T.size() / 2]);
        reps.push_back(T.back());
        reps.push_back(compute_centroid(T));
        return reps;
    }

    // LB: squared minimal distance between two bboxes (0 si se solapan)
    static double bbox_lb_squared(const BBox& A, const BBox& B) {
        double dx = 0.0;
        if (A.maxx < B.minx) dx = B.minx - A.maxx;
        else if (B.maxx < A.minx) dx = A.minx - B.maxx;
        double dy = 0.0;
        if (A.maxy < B.miny) dy = B.miny - A.maxy;
        else if (B.maxy < A.miny) dy = A.miny - B.maxy;
        return dx*dx + dy*dy;
    }

    // Directed Hausdorff (squared) between small rep sets A->B
    // (Se usa SOLO para ordenación de candidatos, NO para poda)
    static double directed_hausdorff_squared(const std::vector<Point2D>& A,
                                             const std::vector<Point2D>& B) {
        if (A.empty() || B.empty()) return 0.0;
        double maxmin = 0.0;
        for (const auto& a : A) {
            double best = std::numeric_limits<double>::infinity();
            for (const auto& b : B) {
                double d = squared_euclidean(a, b);
                if (d < best) best = d;
            }
            if (best > maxmin) maxmin = best;
        }
        return maxmin;
    }

    static double rep_hausdorff_squared(const std::vector<Point2D>& A,
                                        const std::vector<Point2D>& B) {
        if (A.empty() || B.empty()) return 0.0;
        double d1 = directed_hausdorff_squared(A, B);
        double d2 = directed_hausdorff_squared(B, A);
        return (d1 > d2) ? d1 : d2;
    }

    // Frechet discreto (al cuadrado) con cutoff y reuse de workspaces.
    // Devuelve INF (>cutoff) si se poda.
    double frechet_distance_squared_with_cutoff(const Trajectory& P,
                                                const Trajectory& Q,
                                                double cutoff) const
    {
        const std::size_t m = P.size();
        const std::size_t n = Q.size();

        if (m == 0 || n == 0) return 0.0;

        const bool use_workspace = (workspace_prev_.size() >= n && workspace_curr_.size() >= n);
        if (!use_workspace) {
            // Fallback local buffers
            std::vector<double> prev(n, cutoff + 1.0), curr(n, cutoff + 1.0);
            curr[0] = squared_euclidean(P[0], Q[0]);
            for (std::size_t j = 1; j < n; ++j) {
                double d = squared_euclidean(P[0], Q[j]);
                curr[j] = std::max(curr[j-1], d);
            }
            if (m == 1) {
                if (cutoff < std::numeric_limits<double>::infinity()) {
                    double minrow = curr[0];
                    for (std::size_t j = 1; j < n; ++j) if (curr[j] < minrow) minrow = curr[j];
                    if (minrow > cutoff) return cutoff + 1.0;
                }
                return curr[n-1];
            }
            for (std::size_t i = 1; i < m; ++i) {
                prev.swap(curr);
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
                if (cutoff < std::numeric_limits<double>::infinity() && row_min > cutoff) {
                    return cutoff + 1.0;
                }
            }
            return curr[n-1];
        }

        // Reusable workspaces
        std::vector<double>& prev = workspace_prev_;
        std::vector<double>& curr = workspace_curr_;
        double INF_VAL = cutoff + 1.0;
        for (std::size_t j = 0; j < n; ++j) prev[j] = INF_VAL, curr[j] = INF_VAL;

        curr[0] = squared_euclidean(P[0], Q[0]);
        for (std::size_t j = 1; j < n; ++j) {
            double d = squared_euclidean(P[0], Q[j]);
            curr[j] = std::max(curr[j-1], d);
        }
        if (m == 1) {
            if (cutoff < std::numeric_limits<double>::infinity()) {
                double minrow = curr[0];
                for (std::size_t j = 1; j < n; ++j) if (curr[j] < minrow) minrow = curr[j];
                if (minrow > cutoff) return INF_VAL;
            }
            return curr[n-1];
        }
        for (std::size_t i = 1; i < m; ++i) {
            prev.swap(curr);
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
            if (cutoff < std::numeric_limits<double>::infinity() && row_min > cutoff) {
                return INF_VAL;
            }
        }
        return curr[n-1];
    }

    // -----------------------
    // Datos precomputados
    // -----------------------
    std::vector<DataItem> data_;
    std::vector<TrajectoryId> ids_;
    std::vector<BBox> bboxes_;
    std::vector<Point2D> centroids_;
    std::vector<Point2D> starts_;
    std::vector<Point2D> ends_;
    std::vector<std::vector<Point2D>> reps_;
    std::size_t candidate_factor_;
    std::size_t L_;

    // Workspaces reusables (mutable para que knn_search const sea thread-safe si no se comparte)
    mutable std::vector<double> workspace_prev_;
    mutable std::vector<double> workspace_curr_;
};

#endif
