#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

bool parse_u64(const char* text, std::uint64_t& value) {
    char* end = nullptr;
    value = std::strtoull(text, &end, 10);
    return end != text && *end == '\0';
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string task(argv[1]);
    std::uint64_t n = 0;
    std::uint64_t repeats = 0;
    if (!parse_u64(argv[2], n) || !parse_u64(argv[3], repeats)) return 3;
    if (n > 1'000'000 || repeats == 0 || repeats > 100'000 || n * repeats > 100'000'000) return 4;

    const auto started = std::chrono::steady_clock::now();
    if (task == "hashsum") {
        std::uint64_t checksum = 0;
        for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
            std::uint64_t subtotal = 0;
            for (std::uint64_t value = 0; value < n; ++value) {
                std::uint64_t mixed = value + 0x9E3779B97F4A7C15ULL;
                mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
                mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
                mixed ^= mixed >> 31;
                const auto sample = mixed & 0xFFFFFFFFULL;
                subtotal += sample * sample;
            }
            checksum += subtotal;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        std::cout << elapsed << ' ' << checksum << '\n';
        return 0;
    }

    if (task == "sieve") {
        std::uint64_t total_count = 0;
        std::uint64_t total_sum = 0;
        for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
            std::vector<std::uint8_t> flags(static_cast<std::size_t>(n + 1), 1);
            flags[0] = 0;
            if (n >= 1) flags[1] = 0;
            for (std::uint64_t p = 2; p * p <= n; ++p) {
                if (flags[static_cast<std::size_t>(p)] != 0) {
                    for (std::uint64_t composite = p * p; composite <= n; composite += p) {
                        flags[static_cast<std::size_t>(composite)] = 0;
                    }
                }
            }
            for (std::uint64_t value = 2; value <= n; ++value) {
                if (flags[static_cast<std::size_t>(value)] != 0) {
                    ++total_count;
                    total_sum += value;
                }
            }
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        std::cout << elapsed << ' ' << total_count << ' ' << total_sum << '\n';
        return 0;
    }
    if (task == "sort") {
        std::uint64_t total = 0;
        std::uint64_t weighted_total = 0;
        for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
            std::vector<std::uint64_t> values(static_cast<std::size_t>(n));
            for (std::uint64_t index = 0; index < n; ++index) {
                values[static_cast<std::size_t>(index)] = (index * 2'654'435'761ULL + 12'345ULL) & 0xFFFFFFFFULL;
            }
            std::sort(values.begin(), values.end());
            std::uint64_t checksum = 0;
            std::uint64_t weighted = 0;
            for (std::uint64_t index = 0; index < n; ++index) {
                checksum += values[static_cast<std::size_t>(index)];
                weighted += (index + 1) * values[static_cast<std::size_t>(index)];
            }
            total += checksum;
            weighted_total += weighted;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
        std::cout << elapsed << ' ' << total << ' ' << weighted_total << '\n';
        return 0;
    }
    return 5;
}
