#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cstdlib>

#include "Common.h"
#include "TrajectorySearch.h"

struct LabeledTrajectory {
    TrajectoryId id;
    Trajectory   points;
};

static std::vector<LabeledTrajectory> load_trajectories_file(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Error: no se pudo abrir el archivo: " << path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<LabeledTrajectory> result;
    std::string line;
    std::size_t line_num = 0;

    while (std::getline(fin, line)) {
        ++line_num;
        if (line.empty()) continue;

        std::istringstream iss(line);
        TrajectoryId id;
        if (!(iss >> id)) {
            std::cerr << "Error en " << path << ", linea " << line_num << ": id invalido.\n";
            std::exit(EXIT_FAILURE);
        }

        std::vector<double> coords;
        double val;
        while (iss >> val) coords.push_back(val);

        if (coords.size() % 2 != 0) {
            std::cerr << "Error en " << path << ", linea " << line_num
                      << ": numero impar de coordenadas. Deberia ser par (x,y).\n";
            std::exit(EXIT_FAILURE);
        }

        Trajectory traj;
        traj.reserve(coords.size() / 2);
        for (std::size_t i = 0; i < coords.size(); i += 2)
            traj.push_back(Point2D{coords[i], coords[i + 1]});

        result.push_back(LabeledTrajectory{id, std::move(traj)});
    }

    return result;
}

static std::unordered_map<TrajectoryId, std::vector<TrajectoryId>> load_answers_file(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Error: no se pudo abrir respuestas: " << path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::unordered_map<TrajectoryId, std::vector<TrajectoryId>> map;
    std::string line;
    std::size_t line_num = 0;

    while (std::getline(fin, line)) {
        ++line_num;
        if (line.empty()) continue;

        std::istringstream iss(line);
        TrajectoryId qid;
        if (!(iss >> qid)) {
            std::cerr << "Error leyendo qid en answers, linea " << line_num << "\n";
            std::exit(EXIT_FAILURE);
        }

        std::vector<TrajectoryId> neigh;
        TrajectoryId id;
        while (iss >> id) neigh.push_back(id);

        if (neigh.empty()) {
            std::cerr << "Error: query sin vecinos en answers, linea " << line_num << "\n";
            std::exit(EXIT_FAILURE);
        }

        map.emplace(qid, std::move(neigh));
    }

    return map;
}

int main(int argc, char** argv) {
    const std::string path_tr  = argv[1];
    const std::string path_q   = argv[2];
    const std::string path_ans = argv[3];
    const std::size_t K        = std::stoull(argv[4]);

    if (K == 0) {
        std::cerr << "Error: K debe ser > 0.\n";
        return EXIT_FAILURE;
    }

    auto base    = load_trajectories_file(path_tr);
    auto queries = load_trajectories_file(path_q);
    auto answers = load_answers_file(path_ans);

    std::vector<TrajectorySearch::DataItem> items;
    items.reserve(base.size());
    for (auto& e : base) items.emplace_back(e.id, e.points);

    TrajectorySearch index(std::move(items));
    index.build();

    std::size_t total = queries.size();
    std::size_t exact_match = 0;
    std::size_t total_hits  = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (auto& q : queries) {
        auto it = answers.find(q.id);
        if (it == answers.end()) continue;

        const auto& expected_full = it->second;
        if (expected_full.size() < K) {
            std::cerr << "Advertencia: respuesta para qid=" << q.id
                      << " tiene menos de K=" << K << "\n";
            continue;
        }

        std::vector<TrajectoryId> expected(expected_full.begin(),
                                           expected_full.begin() + K);

        auto pred = index.knn_search(q.points, K);

        bool ok = true;
        for (std::size_t i = 0; i < K; ++i) {
            if (pred[i] != expected[i]) {
                ok = false;
                break;
            }
        }
        if (ok) exact_match++;

        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j)
                if (pred[i] == expected[j]) {
                    total_hits++;
                    break;
                }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double query_t = std::chrono::duration<double>(t1 - t0).count();

    double total_d = static_cast<double>(total);
    double exact_pct = (total_d > 0.0) ? (100.0 * static_cast<double>(exact_match) / total_d) : 0.0;

    std::cout << "Total queries: " << total << "\n";
    std::cout << "Exact matches: " << exact_match << " (" << exact_pct << "%)\n";
    std::cout << "Neighbor hits: " << total_hits << " / " << (total * K) << "\n";
    std::cout << "Query time:    " << query_t << " s\n";

    return 0;
}
