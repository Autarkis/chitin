#include <emscripten/bind.h>
#include "PreProcessor.h"
#include "Reconstructors.h"
#include <vector>
#include <cstdint>
#include <algorithm>

using namespace emscripten;
using namespace PoissonRecon;

static const unsigned int FEMSig = FEMDegreeAndBType<
    Reconstructor::Poisson::DefaultFEMDegree,
    Reconstructor::Poisson::DefaultFEMBoundary
>::Signature;
using FEMSigs = IsotropicUIntPack<3, FEMSig>;

struct PoissonOutput {
    std::vector<float> vertices;    // flat xyz
    std::vector<uint32_t> indices;  // flat triangle indices
};

// Input stream: feeds oriented point samples to the solver
struct InputSampleStream : public Reconstructor::InputOrientedSampleStream<float, 3> {
    const double* positions;
    const double* normals;
    unsigned int count;
    unsigned int current;

    InputSampleStream(const double* p, const double* n, unsigned int c)
        : positions(p), normals(n), count(c), current(0) {}

    void reset() override { current = 0; }
    bool read(Point<float, 3>& p, Point<float, 3>& n) override {
        if (current >= count) return false;
        unsigned int i = current * 3;
        p[0] = static_cast<float>(positions[i]);
        p[1] = static_cast<float>(positions[i + 1]);
        p[2] = static_cast<float>(positions[i + 2]);
        n[0] = static_cast<float>(normals[i]);
        n[1] = static_cast<float>(normals[i + 1]);
        n[2] = static_cast<float>(normals[i + 2]);
        current++;
        return true;
    }
};

// Output vertex stream: captures position + density
struct OutputVertexStream : public Reconstructor::OutputLevelSetVertexStream<float, 3> {
    std::vector<float> positions;
    std::vector<float> densities;

    size_t size() const override {
        return densities.size();
    }
    size_t write(const Point<float, 3>& p, const Point<float, 3>&, const float& d) override {
        positions.push_back(p[0]);
        positions.push_back(p[1]);
        positions.push_back(p[2]);
        densities.push_back(d);
        return densities.size() - 1;
    }
};

// Output polygon stream: captures polygons and fan-triangulates
struct OutputPolygonStream : public Reconstructor::OutputFaceStream<2> {
    std::vector<uint32_t> indices;

    size_t size() const override {
        return indices.size() / 3;
    }
    size_t write(const std::vector<node_index_type>& polygon) override {
        size_t start = indices.size() / 3;
        if (polygon.size() < 3) return start;
        for (size_t i = 1; i + 1 < polygon.size(); i++) {
            indices.push_back(static_cast<uint32_t>(polygon[0]));
            indices.push_back(static_cast<uint32_t>(polygon[i]));
            indices.push_back(static_cast<uint32_t>(polygon[i + 1]));
        }
        return start;
    }
};

PoissonOutput poissonReconstruct(
    val js_positions,       // Float64Array, length N*3
    val js_normals,         // Float64Array, length N*3
    int depth,              // octree depth (4-10)
    float density_quantile  // low-density trim (0.0 to disable)
) {
    ThreadPool::ParallelizationType = (ThreadPool::ParallelType)0;

    unsigned int pos_len = js_positions["length"].as<unsigned int>();
    unsigned int n_points = pos_len / 3;

    std::vector<double> positions(pos_len);
    std::vector<double> normals(pos_len);

    for (unsigned int i = 0; i < pos_len; i++) {
        positions[i] = js_positions[i].as<double>();
        normals[i] = js_normals[i].as<double>();
    }

    InputSampleStream sampleStream(positions.data(), normals.data(), n_points);

    Reconstructor::Poisson::SolutionParameters<float> solverParams;
    solverParams.depth = depth;

    auto implicit = Reconstructor::Poisson::Solver<float, 3, FEMSigs>::Solve(
        sampleStream, solverParams
    );

    OutputVertexStream vertexStream;
    OutputPolygonStream polygonStream;

    Reconstructor::LevelSetExtractionParameters extractionParams;
    implicit->extractLevelSet(vertexStream, polygonStream, extractionParams);
    delete implicit;

    // Density trimming
    if (density_quantile > 0.0f && !vertexStream.densities.empty()) {
        unsigned int n_verts = static_cast<unsigned int>(vertexStream.densities.size());

        // Compute quantile threshold
        std::vector<float> sorted_densities(vertexStream.densities);
        std::sort(sorted_densities.begin(), sorted_densities.end());
        unsigned int quantile_idx = static_cast<unsigned int>(
            density_quantile * static_cast<float>(n_verts)
        );
        if (quantile_idx >= n_verts) quantile_idx = n_verts - 1;
        float threshold = sorted_densities[quantile_idx];

        // Mark vertices to keep
        std::vector<int> remap(n_verts, -1);
        std::vector<float> trimmed_verts;
        trimmed_verts.reserve(n_verts * 3);
        uint32_t new_idx = 0;

        for (unsigned int i = 0; i < n_verts; i++) {
            if (vertexStream.densities[i] >= threshold) {
                remap[i] = static_cast<int>(new_idx);
                trimmed_verts.push_back(vertexStream.positions[i * 3]);
                trimmed_verts.push_back(vertexStream.positions[i * 3 + 1]);
                trimmed_verts.push_back(vertexStream.positions[i * 3 + 2]);
                new_idx++;
            }
        }

        // Remap triangles, discard any referencing removed vertices
        std::vector<uint32_t> trimmed_indices;
        trimmed_indices.reserve(polygonStream.indices.size());

        for (size_t i = 0; i + 2 < polygonStream.indices.size(); i += 3) {
            int a = remap[polygonStream.indices[i]];
            int b = remap[polygonStream.indices[i + 1]];
            int c = remap[polygonStream.indices[i + 2]];
            if (a >= 0 && b >= 0 && c >= 0) {
                trimmed_indices.push_back(static_cast<uint32_t>(a));
                trimmed_indices.push_back(static_cast<uint32_t>(b));
                trimmed_indices.push_back(static_cast<uint32_t>(c));
            }
        }

        PoissonOutput output;
        output.vertices = std::move(trimmed_verts);
        output.indices = std::move(trimmed_indices);
        return output;
    }

    PoissonOutput output;
    output.vertices = std::move(vertexStream.positions);
    output.indices = std::move(polygonStream.indices);
    return output;
}

EMSCRIPTEN_BINDINGS(poisson_module) {
    register_vector<float>("VectorFloat");
    register_vector<uint32_t>("VectorUint32");

    value_object<PoissonOutput>("PoissonOutput")
        .field("vertices", &PoissonOutput::vertices)
        .field("indices", &PoissonOutput::indices);

    function("poissonReconstruct", &poissonReconstruct);
}
