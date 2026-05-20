// Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
#include <emscripten/bind.h>
#include <coacd.h>
#include <vector>
#include <cstdint>

using namespace emscripten;

struct WasmHull {
    std::vector<float> vertices;  // flat xyz
    std::vector<uint32_t> indices; // flat triangle indices
};

struct WasmResult {
    std::vector<WasmHull> hulls;
};

WasmResult decompose(
    val js_vertices,   // Float64Array, length N*3
    val js_faces,      // Int32Array, length M*3
    double threshold,
    int max_convex_hull,
    int prep_resolution,
    int sample_resolution,
    int mcts_nodes,
    int mcts_iteration,
    int mcts_max_depth,
    int max_ch_vertex,
    bool merge
) {
    unsigned int vert_len = js_vertices["length"].as<unsigned int>();
    unsigned int face_len = js_faces["length"].as<unsigned int>();

    unsigned int n_verts = vert_len / 3;
    unsigned int n_faces = face_len / 3;

    coacd::Mesh input;
    input.vertices.resize(n_verts);
    input.indices.resize(n_faces);

    for (unsigned int i = 0; i < n_verts; i++) {
        input.vertices[i] = {
            js_vertices[i * 3].as<double>(),
            js_vertices[i * 3 + 1].as<double>(),
            js_vertices[i * 3 + 2].as<double>()
        };
    }

    for (unsigned int i = 0; i < n_faces; i++) {
        input.indices[i] = {
            js_faces[i * 3].as<int>(),
            js_faces[i * 3 + 1].as<int>(),
            js_faces[i * 3 + 2].as<int>()
        };
    }

    auto parts = coacd::CoACD(
        input,
        threshold,
        max_convex_hull,
        "auto",
        prep_resolution,
        sample_resolution,
        mcts_nodes,
        mcts_iteration,
        mcts_max_depth,
        false,  // pca
        merge,
        false,  // decimate
        max_ch_vertex,
        false,  // extrude
        0.01,   // extrude_margin
        "ch",   // apx_mode
        0,      // seed
        false   // real_metric
    );

    WasmResult result;
    result.hulls.reserve(parts.size());

    for (auto& part : parts) {
        WasmHull hull;
        hull.vertices.reserve(part.vertices.size() * 3);
        hull.indices.reserve(part.indices.size() * 3);

        for (auto& v : part.vertices) {
            hull.vertices.push_back(static_cast<float>(v[0]));
            hull.vertices.push_back(static_cast<float>(v[1]));
            hull.vertices.push_back(static_cast<float>(v[2]));
        }

        for (auto& f : part.indices) {
            hull.indices.push_back(static_cast<uint32_t>(f[0]));
            hull.indices.push_back(static_cast<uint32_t>(f[1]));
            hull.indices.push_back(static_cast<uint32_t>(f[2]));
        }

        result.hulls.push_back(std::move(hull));
    }

    return result;
}

EMSCRIPTEN_BINDINGS(coacd_module) {
    register_vector<float>("VectorFloat");
    register_vector<uint32_t>("VectorUint32");

    value_object<WasmHull>("WasmHull")
        .field("vertices", &WasmHull::vertices)
        .field("indices", &WasmHull::indices);

    register_vector<WasmHull>("VectorWasmHull");

    value_object<WasmResult>("WasmResult")
        .field("hulls", &WasmResult::hulls);

    function("decompose", &decompose);
}
