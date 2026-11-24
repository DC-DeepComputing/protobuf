// RISC-V RVV SIMD UTF-8 validation, adapted from utf8_range_neon.inc.
// VLEN=256: vuint8m1_t holds 32 bytes -> 2x the NEON 16-byte block.
// The algorithm is the Google "range" UTF-8 validator: classify each byte by
// its high nibble into a required [min,max] range, with cross-block carry from
// the previous block's codepoint lengths. Bit-exact with the naive validator.

#include <riscv_vector.h>

// vextq_u8(a, b, 16-k) emulation for k=1..3 (VLEN-block):
// returns [a[vl-k..vl-1], b[0..vl-k-1]] (concatenate a:b, take bytes from vl-k).
static FORCE_INLINE_ATTR vuint8m1_t utf8_rvv_vext_last(
    vuint8m1_t prev, vuint8m1_t cur, int k, size_t vl) {
  // vslideup(rt,rs,k) = [rt[0..k-1], rs[0..vl-k-1]]; put prev's last k at front.
  vuint8m1_t tail = __riscv_vslidedown_vx_u8m1(prev, (int)(vl - (size_t)k), vl);
  return __riscv_vslideup_vx_u8m1(tail, cur, k, vl);
}

static FORCE_INLINE_ATTR size_t utf8_range_ValidateUTF8Simd(
    const char* data_original, const char* data, const char* end,
    int return_position) {
  // Lookup tables (16 entries each, on high nibble).
  const uint8_t first_len_tbl[16] = {
      0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,3};
  const uint8_t range_min_tbl[16] = {
      0x00,0x80,0x80,0x80,0xA0,0x80,0x90,0x80,
      0xC2,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
  const uint8_t range_max_tbl[16] = {
      0x7F,0xBF,0xBF,0xBF,0xBF,0x9F,0xBF,0x8F,
      0xF4,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  // Range adjustment table, 32 bytes indexed by (prev_byte - 0xE0),
  // interleaved as [E0..EF values | F0..FF values] (vld2q_u8 result):
  //   index 0(E0)=2, index 13(ED)=3, index 16(F0)=3, index 20(F4)=4.
  const uint8_t range_adjust_tbl[32] = {
      2,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,   // E0..EF
      3,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0};  // F0..FF

  const size_t vl = __riscv_vsetvl_e8m1(32);

  // Load lookup tables into vector registers once.
  const vuint8m1_t vt_first_len   = __riscv_vle8_v_u8m1(first_len_tbl, vl);
  const vuint8m1_t vt_min         = __riscv_vle8_v_u8m1(range_min_tbl, vl);
  const vuint8m1_t vt_max         = __riscv_vle8_v_u8m1(range_max_tbl, vl);
  const vuint8m1_t vt_adjust      = __riscv_vle8_v_u8m1(range_adjust_tbl, vl);

  vuint8m1_t prev_input    = __riscv_vmv_v_x_u8m1(0, vl);
  vuint8m1_t prev_first_len= __riscv_vmv_v_x_u8m1(0, vl);
  vbool8_t any_bad = __riscv_vmsne_vx_u8m1_b8(
      __riscv_vmv_v_x_u8m1(0, vl), 0, vl);  // all-false mask
  const vuint8m1_t const_1 = __riscv_vmv_v_x_u8m1(1, vl);
  const vuint8m1_t const_2 = __riscv_vmv_v_x_u8m1(2, vl);
  const vuint8m1_t const_e0= __riscv_vmv_v_x_u8m1(0xE0, vl);

  while ((size_t)(end - data) >= vl) {
    const vuint8m1_t input = __riscv_vle8_v_u8m1((const uint8_t*)data, vl);

    // high nibble = input >> 4
    const vuint8m1_t high = __riscv_vsrl_vx_u8m1(input, 4, vl);

    // first_len = first_len_tbl[high]
    const vuint8m1_t first_len = __riscv_vrgather_vv_u8m1(vt_first_len, high, vl);

    // range = first_range_tbl[high] == (high >= 12 ? 8 : 0)
    // Replaces a vrgather (microcoded on X100) with compare+merge.
    vbool8_t hi_ge_c = __riscv_vmsgtu_vx_u8m1_b8(high, 11, vl);
    vuint8m1_t range = __riscv_vmerge_vxm_u8m1(
        __riscv_vmv_v_x_u8m1(0, vl), 8, hi_ge_c, vl);

    // range |= vext(prev_first_len, first_len, vl-1)  (shift by 1: previous lane)
    {
      vuint8m1_t s = utf8_rvv_vext_last(prev_first_len, first_len, 1, vl);
      range = __riscv_vor_vv_u8m1(range, s, vl);
    }
    // shift2 = vext(prev_first_len, first_len, vl-2) - 1 (saturating)
    {
      vuint8m1_t s = utf8_rvv_vext_last(prev_first_len, first_len, 2, vl);
      vbool8_t nz = __riscv_vmsne_vx_u8m1_b8(s, 0, vl);
      s = __riscv_vsub_vx_u8m1(s, 1, vl);
      s = __riscv_vmerge_vxm_u8m1(s, 0, __riscv_vmnot_m_b8(nz, vl), vl);  // saturating
      range = __riscv_vor_vv_u8m1(range, s, vl);
    }
    // shift3 = vext(prev_first_len, first_len, vl-3) - 2
    {
      vuint8m1_t s = utf8_rvv_vext_last(prev_first_len, first_len, 3, vl);
      vbool8_t ge2 = __riscv_vmsgtu_vx_u8m1_b8(s, 1, vl);
      s = __riscv_vsub_vx_u8m1(s, 2, vl);
      s = __riscv_vmerge_vxm_u8m1(s, 0, __riscv_vmnot_m_b8(ge2, vl), vl);  // saturating
      range = __riscv_vor_vv_u8m1(range, s, vl);
    }
    // range_adjust: idx = vext(prev_input, input, vl-1) - 0xE0
    {
      vuint8m1_t s = utf8_rvv_vext_last(prev_input, input, 1, vl);
      s = __riscv_vsub_vv_u8m1(s, const_e0, vl);
      // vrgather with out-of-range idx (>31) returns 0, matching vqtbl2q_u8.
      vuint8m1_t adj = __riscv_vrgather_vv_u8m1(vt_adjust, s, vl);
      range = __riscv_vadd_vv_u8m1(range, adj, vl);
    }

    const vuint8m1_t min_range = __riscv_vrgather_vv_u8m1(vt_min, range, vl);
    const vuint8m1_t max_range = __riscv_vrgather_vv_u8m1(vt_max, range, vl);

    // error |= (input < min_range) | (input > max_range)
    const vbool8_t lt = __riscv_vmsltu_vv_u8m1_b8(input, min_range, vl);
    const vbool8_t gt = __riscv_vmsgtu_vv_u8m1_b8(input, max_range, vl);
    const vbool8_t bad = __riscv_vmor_mm_b8(lt, gt, vl);
    any_bad = __riscv_vmor_mm_b8(any_bad, bad, vl);

    // For return_position we must stop at the first block containing an error.
    if (return_position && __riscv_vcpop_m_b8(bad, vl) != 0) break;

    prev_input = input;
    prev_first_len = first_len;
    data += vl;
  }

  // Detect any error via mask popcount.
  int any_error = (__riscv_vcpop_m_b8(any_bad, vl) != 0);

  if (return_position && data == data_original) {
    return utf8_range_ValidateUTF8Naive(data, end, return_position);
  }

  // Back up to the start of a codepoint that may straddle the block boundary.
  // Extract the last 4 bytes of prev_input as a word (little-endian).
  uint8_t tailbytes[4] = {0,0,0,0};
  {
    size_t start = vl >= 4 ? vl - 4 : 0;
    vuint8m1_t last4 = __riscv_vslidedown_vx_u8m1(prev_input, (int)start, vl);
    __riscv_vse8_v_u8m1(tailbytes, last4, 4);
  }
  // tailbytes[3] is the last byte (matches NEON codepoint[3] semantics);
  // check directly to avoid the int32 byte-order mismatch of memcpy.
  int skip = 0;
  if (!utf8_range_IsTrailByteOk((char)tailbytes[3])) skip = 1;
  else if (!utf8_range_IsTrailByteOk((char)tailbytes[2])) skip = 2;
  else if (!utf8_range_IsTrailByteOk((char)tailbytes[1])) skip = 3;
  data -= skip;

  if (return_position) {
    return (data - data_original) +
           utf8_range_ValidateUTF8Naive(data, end, return_position);
  }
  if (any_error) return 0;
  return utf8_range_ValidateUTF8Naive(data, end, return_position);
}
