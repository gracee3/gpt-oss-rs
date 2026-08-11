#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace {

struct alignas(64) TileConfig {
  std::uint8_t palette_id;
  std::uint8_t start_row;
  std::uint8_t reserved[14];
  std::uint16_t colsb[8];
  std::uint8_t rows[8];
};

static_assert(sizeof(TileConfig) == 64);

} // namespace

extern "C" int gpt_oss_amx_int8_tile(const std::int8_t *a,
                                      const std::int8_t *b, std::int32_t *c,
                                      std::uint32_t rows) noexcept {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return 1;
  }
  if (rows == 0 || rows > 16) {
    return 2;
  }

  TileConfig config{};
  config.palette_id = 1;
  config.colsb[0] = 64;
  config.rows[0] = static_cast<std::uint8_t>(rows);
  config.colsb[1] = 32;
  config.rows[1] = static_cast<std::uint8_t>(rows);
  config.colsb[2] = 64;
  config.rows[2] = 8;

  _tile_loadconfig(&config);
  _tile_zero(0);
  _tile_loadd(1, a, 32);
  _tile_loadd(2, b, 64);
  _tile_dpbssd(0, 1, 2);
  _tile_stored(0, c, 64);
  _tile_release();
  return 0;
}
