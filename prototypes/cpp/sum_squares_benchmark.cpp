#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    char* n_end = nullptr;
    char* repeats_end = nullptr;
    const auto n = std::strtoull(argv[1], &n_end, 10);
    const auto repeats = std::strtoull(argv[2], &repeats_end, 10);
    if (n_end == argv[1] || *n_end != '\0' || repeats_end == argv[2] || *repeats_end != '\0') return 3;
    if (n == 0 || n > 1'000'000 || repeats == 0 || repeats > 100'000 || n * repeats > 100'000'000) return 4;

    std::uint64_t checksum = 0;
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
        std::uint64_t subtotal = 0;
        for (std::uint64_t value = 0; value < n; ++value) {
            subtotal += value * value;
        }
        checksum += subtotal;
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    std::cout << nanoseconds << ' ' << checksum << '\n';
    return 0;
}
