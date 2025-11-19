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

// TrajectorySearch basado en BVH (Bounding Volume Hierarchy).
// - Construye un BVH binario sobre las bounding boxes de las trayectorias.
// - Query: best-first traversal por cota de bbox (lower bound)
// - Poda segura: bbox_lb_squared (y start/end LB dentro de hojas) -> NO depende de desigualdad triangular
// - Frechet con early-abandon (cutoff) y workspace reutilizable.
// - Exactitud preservada: sólo se poda con cotas válidas.

class TrajectorySearch {
public:
    using DataItem = std::pair<TrajectoryId, Trajectory>;

    explicit TrajectorySearch(std::vector<DataItem> data,
                              std::size_t leaf_size = 8,
                              std::size_t candidate_factor = 8)
        : data_(std::move(data)),
          leaf_size_(leaf_size),
          candidate_factor_(candidate_factor)
    {
        const std::size_t n = data_.size();
        if (n == 0) {
            L_ = 0;
            return;
        }

        L_ = data_[0].second.size(); // asume longitud fija por enunciado

        bboxes_.reserve(n);
        centroids_.reserve(n);
        starts_.reserve(n);
        ends_.reserve(n);
        ids_.reserve(n);

        // precomputos por trayectoria
        for (std::size_t i = 0; i < n; ++i) {
            ids_.push_back(data_[i].first);
            const Trajectory& T = data_[i].second;
            bboxes_.push_back(compute_bbox(T));
            centroids_.push_back(compute_centroid(T));
            if (!T.empty()) {
                starts_.push_back(T.front());
                ends_.push_back(T.back());
            } else {
                starts_.push_back(Point2D{0.0,0.0});
                ends_.push_back(Point2D{0.0,0.0});
            }
        }

        // reservar workspaces reutilizables para Frechet
        if (L_ > 0) {
            workspace_prev_.assign(L_, std::numeric_limits<double>::infinity());
            workspace_curr_.assign(L_, std::numeric_limits<double>::infinity());
        }

        // construir BVH
        build_bvh();
    }

    void build() { /* interfaz: todo ya construido en constructor */ }

    // k-NN exacto usando BVH + poda segura por bbox & start/end + Frechet con cutoff
    std::vector<TrajectoryId> knn_search(const Trajectory& query, std::size_t k) const {
        std::vector<TrajectoryId> result;
        const std::size_t n = data_.size();
        if (n == 0 || k == 0) return result;
        const std::size_t kk = std::min(k, n);

        const double eps = 1e-12;
        const double INF = std::numeric_limits<double>::infinity();

        // precomputados query
        const BBox qbox = compute_bbox(query);
        const Point2D qstart = query.empty() ? Point2D{0.0,0.0} : query.front();
        const Point2D qend   = query.empty() ? Point2D{0.0,0.0} : query.back();

        // max-heap con los k mejores (top() = peor)
        struct HeapComp {
            bool operator()(const std::pair<double,TrajectoryId>& a,
                            const std::pair<double,TrajectoryId>& b) const {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            }
        };
        std::priority_queue<std::pair<double,TrajectoryId>,
                            std::vector<std::pair<double,TrajectoryId>>,
                            HeapComp> best_k;

        auto current_worst = [&best_k, INF]() -> double {
            if (best_k.empty()) return INF;
            return best_k.top().first;
        };

        // Best-first traversal: min-heap de nodos por bbox_lb con query bbox
        struct NodeEntry {
            double lb;
            int node_idx;
            bool operator>(const NodeEntry& o) const { return lb > o.lb; }
        };
        std::priority_queue<NodeEntry, std::vector<NodeEntry>, std::greater<NodeEntry>> node_pq;

        // push root
        if (!bvh_nodes_.empty()) {
            double root_lb = bbox_lb_squared(qbox, bvh_nodes_[0].bbox);
            node_pq.push(NodeEntry{root_lb, 0});
        }

        // traversal
        while (!node_pq.empty()) {
            NodeEntry ne = node_pq.top(); node_pq.pop();
            double node_lb = ne.lb;
            int node_idx = ne.node_idx;

            double worst = current_worst();
            if (best_k.size() == kk && node_lb > worst + eps) {
                // poda segura: cualquier traj dentro de este nodo tiene distancia >= node_lb > worst
                continue;
            }

            const BVHNode& node = bvh_nodes_[node_idx];
            if (node.is_leaf) {
                // ----- FIX: iterar sobre indices_ mapping (antes se usaba data_[i] directamente, eso era incorrecto)
                // node.start_idx..node.end_idx son posiciones en indices_; hay que mapear indices_[t] -> índice real en data_
                for (std::size_t t = node.start_idx; t < node.end_idx; ++t) {
                    std::size_t data_idx = indices_[t]; // <-- CORRECCIÓN IMPORTANTÍSIMA
                    const TrajectoryId id = data_[data_idx].first;
                    const Trajectory& traj = data_[data_idx].second;

                    // cotas seguras: bbox de la trayectoria y puntos start/end
                    double lb_bbox = bbox_lb_squared(qbox, bboxes_[data_idx]);
                    double lb_start = squared_euclidean(qstart, starts_[data_idx]);
                    double lb_end   = squared_euclidean(qend,   ends_[data_idx]);

                    double lb = lb_bbox;
                    if (lb_start > lb) lb = lb_start;
                    if (lb_end   > lb) lb = lb_end;

                    worst = current_worst();
                    if (best_k.size() == kk && lb > worst + eps) continue; // poda segura por LBs

                    double cutoff = (best_k.size() < kk) ? INF : (worst - eps);
                    double dist2 = frechet_distance_squared_with_cutoff(query, traj, cutoff);

                    if (best_k.size() < kk) {
                        best_k.emplace(dist2, id);
                    } else {
                        HeapComp cmp;
                        std::pair<double,TrajectoryId> cand(dist2, id);
                        if (cmp(cand, best_k.top())) {
                            best_k.pop();
                            best_k.emplace(cand);
                        }
                    }
                }
            } else {
                // push children (orden no relevante; usamos lb)
                if (node.left >= 0) {
                    double lb_left = bbox_lb_squared(qbox, bvh_nodes_[node.left].bbox);
                    node_pq.push(NodeEntry{lb_left, node.left});
                }
                if (node.right >= 0) {
                    double lb_right = bbox_lb_squared(qbox, bvh_nodes_[node.right].bbox);
                    node_pq.push(NodeEntry{lb_right, node.right});
                }
            }
        }

        // extraer y ordenar resultado por (dist asc, id asc)
        std::vector<std::pair<double,TrajectoryId>> chosen;
        chosen.reserve(best_k.size());
        while (!best_k.empty()) {
            chosen.push_back(best_k.top());
            best_k.pop();
        }
        std::sort(chosen.begin(), chosen.end(),
                  [eps](const std::pair<double,TrajectoryId>& a, const std::pair<double,TrajectoryId>& b) {
                      if (std::fabs(a.first - b.first) > eps) return a.first < b.first;
                      return a.second < b.second;
                  });
        result.reserve(chosen.size());
        for (const auto& p : chosen) result.push_back(p.second);
        return result;
    }

private:
    // -----------------------
    // BVH internals
    // -----------------------
    struct BBox { double minx, maxx, miny, maxy; };
    
    struct BVHNode {
        BBox bbox;
        int left = -1;
        int right = -1;
        bool is_leaf = false;
        std::size_t start_idx = 0; // indices_ range [start_idx, end_idx)
        std::size_t end_idx = 0;
    };

    void build_bvh() {
        const std::size_t n = data_.size();
        indices_.resize(n);
        for (std::size_t i = 0; i < n; ++i) indices_[i] = i;
        bvh_nodes_.clear();
        bvh_nodes_.reserve(n * 2);
        build_bvh_node(0, n);
    }

    // build node covering indices_[l, r)
    int build_bvh_node(std::size_t l, std::size_t r) {
        // Create node and push to vector; return its index.
        BVHNode node;
        // compute node bbox as union (uses indices_ mapping)
        BBox nb = compute_bbox_from_indices(l, r);
        node.bbox = nb;

        std::size_t node_index = bvh_nodes_.size();
        bvh_nodes_.push_back(node); // placeholder; will fill later

        std::size_t count = r - l;
        if (count <= leaf_size_) {
            // make leaf: store range in indices_ array [l, r)
            bvh_nodes_[node_index].is_leaf = true;
            bvh_nodes_[node_index].start_idx = l;
            bvh_nodes_[node_index].end_idx = r;
            return static_cast<int>(node_index);
        }

        // choose split axis by bbox extent
        double dx = nb.maxx - nb.minx;
        double dy = nb.maxy - nb.miny;
        int axis = (dx >= dy) ? 0 : 1;

        // partition indices_[l..r) by median of centroid coordinate
        std::size_t mid = l + (count / 2);
        std::nth_element(indices_.begin() + l, indices_.begin() + mid, indices_.begin() + r,
                         [&](std::size_t a, std::size_t b) {
                             if (axis == 0) return centroids_[a].x < centroids_[b].x;
                             return centroids_[a].y < centroids_[b].y;
                         });
        int left_child = build_bvh_node(l, mid);
        int right_child = build_bvh_node(mid, r);

        bvh_nodes_[node_index].left = left_child;
        bvh_nodes_[node_index].right = right_child;
        bvh_nodes_[node_index].is_leaf = false;
        bvh_nodes_[node_index].start_idx = 0;
        bvh_nodes_[node_index].end_idx = 0;
        return static_cast<int>(node_index);
    }

    // compute union bbox over indices_[l..r)
    BBox compute_bbox_from_indices(std::size_t l, std::size_t r) const {
        if (l >= r) return BBox{0.0,0.0,0.0,0.0};
        std::size_t first_idx = indices_[l];
        BBox nb = bboxes_[first_idx];
        for (std::size_t t = l + 1; t < r; ++t) {
            std::size_t idx = indices_[t];
            const BBox& b = bboxes_[idx];
            if (b.minx < nb.minx) nb.minx = b.minx;
            if (b.maxx > nb.maxx) nb.maxx = b.maxx;
            if (b.miny < nb.miny) nb.miny = b.miny;
            if (b.maxy > nb.maxy) nb.maxy = b.maxy;
        }
        return nb;
    }

    // -----------------------
    // Utilities (same seguros de antes)
    // -----------------------
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
        if (T.empty()) return Point2D{0.0,0.0};
        double sx = 0.0, sy = 0.0;
        for (const auto& p : T) { sx += p.x; sy += p.y; }
        double inv = 1.0 / static_cast<double>(T.size());
        return Point2D{sx * inv, sy * inv};
    }

    // minimal squared distance between two axis-aligned bboxes (0 if overlap)
    static double bbox_lb_squared(const BBox& A, const BBox& B) {
        double dx = 0.0;
        if (A.maxx < B.minx) dx = B.minx - A.maxx;
        else if (B.maxx < A.minx) dx = A.minx - B.maxx;
        double dy = 0.0;
        if (A.maxy < B.miny) dy = B.miny - A.maxy;
        else if (B.maxy < A.miny) dy = A.miny - B.maxy;
        return dx*dx + dy*dy;
    }

    // Frechet con cutoff (reuso de workspaces) - igual que implementación probada
    double frechet_distance_squared_with_cutoff(const Trajectory& P,
                                                const Trajectory& Q,
                                                double cutoff) const
    {
        const std::size_t m = P.size();
        const std::size_t n = Q.size();
        if (m == 0 || n == 0) return 0.0;

        const bool use_workspace = (workspace_prev_.size() >= n && workspace_curr_.size() >= n);
        if (!use_workspace) {
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
    // Datos y precomputos
    // -----------------------
    std::vector<DataItem> data_;
    std::vector<TrajectoryId> ids_;
    std::vector<BBox> bboxes_;
    std::vector<Point2D> centroids_;
    std::vector<Point2D> starts_;
    std::vector<Point2D> ends_;

    // BVH representation
    std::vector<std::size_t> indices_; // permutation of 0..n-1 used by BVH build
    std::vector<BVHNode> bvh_nodes_;
    std::size_t leaf_size_;
    std::size_t candidate_factor_;
    std::size_t L_;

    // Workspaces for Frechet (mutable, reused)
    mutable std::vector<double> workspace_prev_;
    mutable std::vector<double> workspace_curr_;
};

#endif
