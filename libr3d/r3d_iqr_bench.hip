#include <hip/hip_runtime.h>
#include "r3d_iqr.h"
#include "rad_plugin.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
extern "C" int r3d_rad_gemm_iqr_scalar(const RadArgs*, RadStream);
#include "r3d_iqr_bench_host.h"

namespace iqr_bench {
void checked(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
struct Allocation {
    void* data = nullptr;
    size_t bytes;
    explicit Allocation(size_t count) : bytes(count) { checked(hipMalloc(&data, count)); }
    ~Allocation() { if (data) (void)hipFree(data); }
    Allocation(const Allocation&) = delete;
    void upload(const void* source, size_t count) {
        require(count <= bytes, "upload too large");
        checked(hipMemcpy(data, source, count, hipMemcpyHostToDevice));
    }
};
struct Stream {
    hipStream_t value{};
    Stream() { checked(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { (void)hipStreamDestroy(value); }
};
struct Event {
    hipEvent_t value{};
    Event() { checked(hipEventCreate(&value)); }
    ~Event() { (void)hipEventDestroy(value); }
};
RadTensor tensor(void* data, uint32_t type, int rows, int width) {
    RadTensor result{};
    result.data = data; result.dtype = type; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = width;
    rad_tensor_pack(&result);
    return result;
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{};
    result.key = key; result.kind = RAD_P_INT; result.ival = value;
    return result;
}
struct Runner {
    int rows;
    size_t slot_bytes, slot_count, cursor = 0;
    Allocation weights, scales, activation, activation_scale, output;
    Stream stream;
    Event begin, end;
    std::array<RadTensor, 5> operands;
    std::array<RadParam, 6> params;
    RadArgs arguments{};
    Runner(int count, const Model& model) : rows(count),
        slot_bytes(model.weight.size() + model.scales.size() * sizeof(float)),
        slot_count((rotation_bytes + slot_bytes - 1) / slot_bytes),
        weights(model.weight.size() * slot_count),
        scales(model.scales.size() * sizeof(float) * slot_count),
        activation(size_t(rows) * reduction), activation_scale(size_t(rows) * groups * 4),
        output((size_t(rows) * columns + 512) * 2),
        operands{{tensor(activation.data, RAD_I8, rows, reduction),
            tensor(activation_scale.data, RAD_F32, rows, groups),
            tensor(weights.data, r3d_iqr::weight_dtype, columns, reduction),
            tensor(scales.data, r3d_iqr::scale_dtype, columns, groups),
            tensor(static_cast<uint16_t*>(output.data) + 256, RAD_BF16, rows, columns)}},
        params{{integer("M", rows), integer("N", columns), integer("K", reduction),
            integer("group", 128), integer("impl", 0), integer("w_rb", 1)}} {
        for (size_t slot = 0; slot < slot_count; ++slot) {
            checked(hipMemcpy(static_cast<uint8_t*>(weights.data) + slot * model.weight.size(),
                model.weight.data(), model.weight.size(), hipMemcpyHostToDevice));
            checked(hipMemcpy(static_cast<float*>(scales.data) + slot * model.scales.size(),
                model.scales.data(), model.scales.size() * 4, hipMemcpyHostToDevice));
        }
        arguments.t = operands.data(); arguments.n_t = operands.size();
        arguments.p = params.data(); arguments.n_p = params.size();
    }
    void select(size_t slot) {
        operands[2].data = static_cast<uint8_t*>(weights.data) + slot * (32 + size_t(columns) * 1600);
        operands[3].data = static_cast<float*>(scales.data) + slot * (columns + groups);
    }
    void launch(int implementation, size_t slot) {
        select(slot); params[4].ival = implementation;
        require(r3d_rad_gemm_iqr_scalar(&arguments, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
            "production API refused launch");
    }
    float batch(int implementation, bool cold, int iterations) {
        checked(hipEventRecord(begin.value, stream.value));
        for (int iteration = 0; iteration < iterations; ++iteration)
            launch(implementation, cold ? cursor++ % slot_count : 0);
        checked(hipEventRecord(end.value, stream.value));
        checked(hipEventSynchronize(end.value));
        float milliseconds = 0;
        checked(hipEventElapsedTime(&milliseconds, begin.value, end.value));
        return milliseconds;
    }
    void guards() {
        std::array<uint16_t, 256> before{}, after{};
        checked(hipMemcpy(before.data(), output.data, 512, hipMemcpyDeviceToHost));
        checked(hipMemcpy(after.data(), static_cast<uint16_t*>(output.data) + 256 + size_t(rows) * columns,
            512, hipMemcpyDeviceToHost));
        for (uint16_t value : before) require(value == 0x5a5a, "leading output guard");
        for (uint16_t value : after) require(value == 0x5a5a, "trailing output guard");
    }
    void verify(const Model& model, const std::vector<int>& implementations, std::ostream& report) {
        const auto selected = positions(rows);
        bool passed = true;
        for (unsigned repeat = 0; repeat != 2; ++repeat) {
            Input input(rows, repeat);
            const auto expected = reference(model, input, selected);
            activation.upload(input.activation.data(), input.activation.size());
            activation_scale.upload(input.scale.data(), input.scale.size() * 4);
            for (int implementation : implementations) {
                checked(hipMemsetAsync(output.data, 0x5a, output.bytes, stream.value));
                launch(implementation, repeat ? slot_count - 1 : 0);
                checked(hipStreamSynchronize(stream.value));
                std::vector<uint16_t> actual(size_t(rows) * columns);
                checked(hipMemcpy(actual.data(), operands[4].data, actual.size() * 2, hipMemcpyDeviceToHost));
                size_t mismatches = 0;
                for (size_t index = 0; index < selected.size(); ++index) {
                    if (actual[selected[index]] != expected[index]) {
                        if (mismatches < 16)
                            std::fprintf(stderr,
                                "mismatch M=%d impl=%d repeat=%u row=%zu col=%zu expected=%04x actual=%04x\n",
                                rows, implementation, repeat, selected[index] / columns,
                                selected[index] % columns, unsigned(expected[index]), unsigned(actual[selected[index]]));
                        ++mismatches;
                    }
                }
                guards();
                report << rows << ',' << implementation << ',' << repeat << ',' << selected.size()
                    << ',' << mismatches << ",512," << (repeat ? slot_count - 1 : 0) << '\n';
                report.flush();
                passed &= !mismatches;
            }
        }
        require(passed, "BF16 output mismatch against independent CPU oracle");
    }
};
struct Samples {
    std::vector<double> values;
    std::array<std::vector<double>, 5> rounds;
};
double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction * (values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    return values[lower] + (values[std::min(lower + 1, values.size() - 1)] - values[lower]) * (position - lower);
}
void summaries(const std::string& directory, const std::map<std::tuple<int, bool, int>, Samples>& data) {
    std::ofstream summary(directory + "/summary.csv");
    summary << "M,state,impl,batches,median_us,p95_us,mean_us,sample_cv_percent,min_us,max_us\n";
    summary << std::setprecision(10);
    for (const auto& [key, samples] : data) {
        const auto [rows, cold, implementation] = key;
        const double mean = std::accumulate(samples.values.begin(), samples.values.end(), 0.0) / samples.values.size();
        double variance = 0;
        for (double value : samples.values) variance += (value - mean) * (value - mean);
        const auto limits = std::minmax_element(samples.values.begin(), samples.values.end());
        summary << rows << ',' << (cold ? "rotation_cold" : "hot") << ',' << implementation << ','
            << samples.values.size() << ',' << percentile(samples.values, .5) << ','
            << percentile(samples.values, .95) << ',' << mean << ','
            << 100 * std::sqrt(variance / (samples.values.size() - 1)) / mean << ','
            << *limits.first << ',' << *limits.second << '\n';
    }
    std::ofstream paired(directory + "/paired.csv");
    paired << "M,state,baseline_impl,candidate_impl,paired_rounds,baseline_over_candidate_geomean,ci95_low,ci95_high,method\n";
    for (const auto& [key, baseline] : data) {
        const auto [rows, cold, implementation] = key;
        if (implementation != 1) continue;
        const auto& candidate = data.at({rows, cold, 2});
        std::array<double, 5> logarithms{};
        for (size_t round = 0; round < logarithms.size(); ++round) {
            double total = 0;
            require(baseline.rounds[round].size() == 20 && candidate.rounds[round].size() == 20,
                "paired round must contain 20 mirrored batches per implementation");
            for (size_t index = 0; index < 20; index += 2) {
                const double base = (baseline.rounds[round][index] + baseline.rounds[round][index + 1]) / 2;
                const double next = (candidate.rounds[round][index] + candidate.rounds[round][index + 1]) / 2;
                total += std::log(base / next);
            }
            logarithms[round] = total / 10;
        }
        const double mean = std::accumulate(logarithms.begin(), logarithms.end(), 0.0) / 5;
        double variance = 0;
        for (double value : logarithms) variance += (value - mean) * (value - mean);
        const double half = 2.7764451051977987 * std::sqrt(variance / 4 / 5);
        paired << rows << ',' << (cold ? "rotation_cold" : "hot") << ",1,2,5,"
            << std::exp(mean) << ',' << std::exp(mean - half) << ',' << std::exp(mean + half)
            << ",paired_mirrored_unit_log_ratio_round_cluster_t4\n";
    }
}
int execute(const char* model_path, const std::string& directory, int verify_rows = 0) {
    const auto started = std::chrono::steady_clock::now();
    int devices = 0; checked(hipGetDeviceCount(&devices));
    require(devices == 1, "must expose exactly GPU4");
    hipDeviceProp_t properties{}; checked(hipGetDeviceProperties(&properties, 0));
    require(properties.pciBusID == 0x83 && properties.pciDeviceID == 0,
        "GPU PCI identity must be physical 83:00.0");
    require(std::strncmp(properties.gcnArchName, "gfx1100", 7) == 0, "gfx1100 required");
    int runtime = 0, driver = 0;
    checked(hipRuntimeGetVersion(&runtime)); checked(hipDriverGetVersion(&driver));
    size_t free_bytes = 0, total_bytes = 0; checked(hipMemGetInfo(&free_bytes, &total_bytes));
    std::ofstream metadata(directory + "/device.json");
    metadata << "{\"name\":\"" << properties.name << "\",\"arch\":\"" << properties.gcnArchName
        << "\",\"pci\":\"0000:83:00.0\",\"visible_devices\":" << devices
        << ",\"hip_runtime\":" << runtime << ",\"hip_driver\":" << driver
        << ",\"vram_free_before_bytes\":" << free_bytes << ",\"vram_total_bytes\":" << total_bytes << "}\n";
    metadata.close();
    Model model(model_path);
    std::ofstream correctness(directory + "/correctness.csv"), raw(directory + "/raw.csv");
    correctness << "M,impl,input_repeat,cpu_outputs,bf16_mismatches,guard_elements,weight_slot\n";
    raw << "source,M,N,K,state,round,unit,position,impl,batch_ms,iterations,latency_us,warm_gpu_ms,warm_iterations,active_weight_bytes,rotation_bytes,allocated_bytes,slot_start,slot_end\n";
    raw << std::setprecision(10);
    std::map<std::tuple<int, bool, int>, Samples> data;
    Random order{order_seed};
    uint64_t total_iterations = 0;
    double measured_gpu_ms = 0, warm_gpu_ms = 0;
    size_t maximum_bytes = 0;
    const std::vector<int> row_cases = verify_rows ? std::vector<int>{verify_rows}
        : std::vector<int>{1, 4, 16, 32, 128, 512, 2048};
    for (int rows : row_cases) {
        const std::vector<int> implementations = rows <= 32 ? std::vector<int>{0, 1, 2} : std::vector<int>{1, 2};
        Runner runner(rows, model);
        runner.verify(model, implementations, correctness);
        if (verify_rows) return 0;
        std::array<double, 3> warmed{};
        std::array<uint64_t, 3> warm_iterations{};
        std::array<std::array<int, 3>, 2> iterations{{{{1, 1, 1}}, {{1, 1, 1}}}};
        const size_t allocated = runner.weights.bytes + runner.scales.bytes + runner.activation.bytes
            + runner.activation_scale.bytes + runner.output.bytes;
        maximum_bytes = std::max(maximum_bytes, allocated);
        for (int implementation : implementations) {
            while (warmed[implementation] < 2000) {
                const int count = iterations[0][implementation];
                const float elapsed = runner.batch(implementation, false, count);
                warmed[implementation] += elapsed; warm_iterations[implementation] += count;
                if (elapsed < 2) iterations[0][implementation] *= 2;
            }
            warm_gpu_ms += warmed[implementation];
            for (int state = 0; state != 2; ++state) {
                while (runner.batch(implementation, state, iterations[state][implementation]) < 2)
                    iterations[state][implementation] *= 2;
            }
        }
        for (int round = 0; round != 5; ++round) {
            const int first_state = order.next() % 2;
            for (int state_index = 0; state_index != 2; ++state_index) {
                const bool cold = (first_state + state_index) % 2;
                for (int unit = 0; unit != 10; ++unit) {
                    auto sequence = implementations;
                    for (size_t count = sequence.size(); count > 1; --count)
                        std::swap(sequence[count - 1], sequence[order.next() % count]);
                    const auto forward = sequence;
                    sequence.insert(sequence.end(), forward.rbegin(), forward.rend());
                    for (size_t position = 0; position < sequence.size(); ++position) {
                        const int implementation = sequence[position];
                        int& count = iterations[cold][implementation];
                        float elapsed = 0;
                        size_t slot_start = 0;
                        do {
                            slot_start = runner.cursor % runner.slot_count;
                            elapsed = runner.batch(implementation, cold, count);
                            if (elapsed < 1) count *= 2;
                        } while (elapsed < 1);
                        const double latency = elapsed * 1000.0 / count;
                        auto& samples = data[{rows, cold, implementation}];
                        samples.values.push_back(latency); samples.rounds[round].push_back(latency);
                        total_iterations += count; measured_gpu_ms += elapsed;
                        raw << "real_row_slice," << rows << ',' << columns << ',' << reduction << ','
                            << (cold ? "rotation_cold" : "hot") << ',' << round << ',' << unit << ',' << position << ','
                            << implementation << ',' << elapsed << ',' << count << ',' << latency << ','
                            << warmed[implementation] << ',' << warm_iterations[implementation] << ','
                            << runner.slot_bytes << ',' << runner.slot_bytes * runner.slot_count << ',' << allocated << ','
                            << (cold ? slot_start : 0) << ',' << (cold ? (runner.cursor - 1) % runner.slot_count : 0) << '\n';
                    }
                }
                raw.flush();
            }
        }
        runner.guards();
        summaries(directory, data);
        std::printf("M=%d completed implementations=%zu batches_per_impl_state=100 rotation=%zu bytes\n",
            rows, implementations.size(), runner.slot_bytes * runner.slot_count);
        std::fflush(stdout);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::ofstream result(directory + "/completed.json");
    result << std::setprecision(12) << "{\"status\":\"pass\",\"timed_cases\":36,\"timed_batches\":3600,\"timed_iterations\":"
        << total_iterations << ",\"measured_gpu_seconds\":" << measured_gpu_ms / 1000
        << ",\"warm_gpu_seconds\":" << warm_gpu_ms / 1000 << ",\"wall_seconds\":" << seconds
        << ",\"maximum_device_allocation_bytes\":" << maximum_bytes
        << ",\"host_decoded_weight_bytes\":" << model.decoded.size()
        << ",\"full_model_tokens_per_second\":null,\"hardware_counters\":null}\n";
    return 0;
}
}
int main(int argc, char** argv) {
    try {
        iqr_bench::host_tests();
        if (argc == 2 && std::string(argv[1]) == "--host") {
            std::puts("{\"host_tests\":\"pass\",\"checks\":6,\"gpu_used\":false}");
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--model-host") {
            iqr_bench::Model model(argv[2]);
            std::puts("{\"model_host\":\"pass\",\"table_states\":65536,\"decoded_weights\":20971520}");
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--verify")
            return iqr_bench::execute(argv[2], argv[3], std::stoi(argv[4]));
        if (argc != 3) throw std::runtime_error("usage: r3d_iqr_bench MODEL OUTPUT_DIRECTORY | --host");
        return iqr_bench::execute(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "r3d_iqr_bench: %s\n", error.what()); return 2;
    }
}
