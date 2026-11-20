#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto n = std::strtoull(argv[1], nullptr, 10);
    if (n > 1000000ULL) return 3;
    std::uint64_t total = 0;
    for (std::uint64_t i = 0; i < n; ++i) total += i * i;
    std::cout << total << '\n';
    return 0;
}
