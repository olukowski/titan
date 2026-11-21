//! Scalar functions that `core` lacks, written in plain Rust so every target (native and WASM)
//! computes bit-identical results. All of them work in `f64` internally and round once to `f32`.

use core::f64::consts::{FRAC_PI_2, PI};

/// Returns the square root of `x`, correctly rounded as IEEE 754 requires.
///
/// Returns NaN for NaN and for negative inputs other than `-0.0`; `sqrt(-0.0)` is `-0.0`.
#[must_use]
pub fn sqrt(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || x == f32::INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let biased_exp = (bits >> 23) as i32;
    let fraction = bits & 0x007F_FFFF;
    // Write x = m · 2^e with m an integer in [2^23, 2^24).
    let (m, e) = if biased_exp == 0 {
        let shift = fraction.leading_zeros() - 8;
        (fraction << shift, -149 - shift as i32)
    } else {
        (fraction | 0x0080_0000, biased_exp - 150)
    };
    // Widen m so the exponent left over is even and the integer root has 25 bits: the 24 we keep
    // plus one rounding bit. A root that lands exactly halfway would need an odd perfect square,
    // and `wide` is even, so rounding half up is round-to-nearest.
    let shift = if (e - 25) & 1 == 0 { 25 } else { 26 };
    let wide = u64::from(m) << shift;
    let mantissa = (wide.isqrt() + 1) >> 1;
    // The result exponent lies in [-98, 40], so the scale is always a normal f32.
    let exp = (e - shift) / 2 + 1;
    mantissa as f32 * f32::from_bits(((exp + 127) as u32) << 23)
}

/// Returns `(sin(x), cos(x))`, accurate to about one ulp for every finite `x`.
///
/// Returns `(NaN, NaN)` for NaN and infinite inputs.
#[must_use]
pub fn sin_cos(x: f32) -> (f32, f32) {
    let (sin, cos) = sin_cos_f64(x);
    (sin as f32, cos as f32)
}

/// Returns `tan(x)`, accurate to about one ulp for every finite `x`.
///
/// Returns NaN for NaN and infinite inputs.
#[must_use]
pub fn tan(x: f32) -> f32 {
    let (sin, cos) = sin_cos_f64(x);
    (sin / cos) as f32
}

/// Returns `acos(x)` in `[0, π]`.
///
/// Returns NaN for NaN and for inputs outside `[-1, 1]`.
#[must_use]
pub fn acos(x: f32) -> f32 {
    let x = f64::from(x);
    // Each branch keeps the series argument within [-0.5, 0.5], where it converges fast.
    let result = if (-0.5..=0.5).contains(&x) {
        FRAC_PI_2 - asin_series(x)
    } else if (0.5..=1.0).contains(&x) {
        2.0 * asin_series(sqrt_f64((1.0 - x) * 0.5))
    } else if (-1.0..=-0.5).contains(&x) {
        PI - 2.0 * asin_series(sqrt_f64((1.0 + x) * 0.5))
    } else {
        f64::NAN
    };
    result as f32
}

fn sin_cos_f64(x: f32) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let abs = x.abs();
    let (quadrant, r) = if abs < core::f32::consts::FRAC_PI_4 {
        (0, f64::from(abs))
    } else {
        reduce(abs)
    };
    let (sin, cos) = (sin_poly(r), cos_poly(r));
    let (sin, cos) = match quadrant {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    };
    // sin is odd and cos is even; this also keeps sin(-0.0) == -0.0.
    (if x.is_sign_negative() { -sin } else { sin }, cos)
}

/// The leading 256 bits of 2/π, enough to reduce any finite `f32`.
const TWO_OVER_PI: [u32; 8] = [
    0xA2F9_836E,
    0x4E44_1529,
    0xFC27_57D1,
    0xF534_DDC0,
    0xDB62_9599,
    0x3C43_9041,
    0xFE51_63AB,
    0xDEBB_C561,
];

/// Splits `x` (finite, `x >= π/4`) into `(q, r)` with `x ≡ q·π/2 + r (mod 2π)`, `q` in `0..4`,
/// and `|r| <= π/4`.
///
/// This is Payne–Hanek reduction: it multiplies the exact `x` by just the bits of 2/π that can
/// affect `x·2/π mod 4`, so the result stays accurate even for `f32::MAX`.
fn reduce(x: f32) -> (u32, f64) {
    let bits = x.to_bits();
    // x = m · 2^e with m a 24-bit integer; x >= π/4 rules out subnormals.
    let m = u128::from((bits & 0x007F_FFFF) | 0x0080_0000);
    let e = (bits >> 23) as i32 - 150;
    // x·2/π = Σ_j m·b_j·2^(e-j) over the bits b_j of 2/π (j >= 1). Terms with e - j >= 2 are
    // multiples of 4, a whole turn, so the sum may start at bit j = e - 1. A 96-bit window of
    // bits truncates the sum by less than 2^-70 and keeps m · window below 2^120.
    let first = (e - 2).max(0) as usize; // 0-based index of the window's first bit
    let (word, offset) = (first / 32, first % 32);
    let leading = TWO_OVER_PI[word..word + 4]
        .iter()
        .fold(0u128, |acc, &w| (acc << 32) | u128::from(w));
    let next = u128::from(TWO_OVER_PI[word + 4]);
    let window = ((leading << offset) | (next >> (32 - offset))) >> 32;
    // `product · 2^-shift` is x·2/π mod 4, with shift in [94, 120].
    let product = m * window;
    let shift = first as i32 + 96 - e;
    let quadrant = (product >> shift) as u32;
    // The fraction bits, read as a signed fixed-point value in [-1/2, 1/2): a set top bit rounds
    // the quadrant up so that |r| <= π/4.
    let fraction = (product << (128 - shift)) as i128;
    let quadrant = quadrant + u32::from(fraction < 0);
    let r = fraction as f64 * f64::from_bits((1023 - 128) << 52) * FRAC_PI_2;
    (quadrant & 3, r)
}

/// Taylor series of sin for `|r| <= π/4`; the first omitted term is below 5e-17.
fn sin_poly(r: f64) -> f64 {
    const C: [f64; 7] = [
        -1.0 / 6.0,
        1.0 / 120.0,
        -1.0 / 5_040.0,
        1.0 / 362_880.0,
        -1.0 / 39_916_800.0,
        1.0 / 6_227_020_800.0,
        -1.0 / 1_307_674_368_000.0,
    ];
    let r2 = r * r;
    let p = C.iter().rev().fold(0.0, |acc, &c| acc * r2 + c);
    r + r * r2 * p
}

/// Taylor series of cos for `|r| <= π/4`; the first omitted term is below 3e-18.
fn cos_poly(r: f64) -> f64 {
    const C: [f64; 8] = [
        -1.0 / 2.0,
        1.0 / 24.0,
        -1.0 / 720.0,
        1.0 / 40_320.0,
        -1.0 / 3_628_800.0,
        1.0 / 479_001_600.0,
        -1.0 / 87_178_291_200.0,
        1.0 / 20_922_789_888_000.0,
    ];
    let r2 = r * r;
    let p = C.iter().rev().fold(0.0, |acc, &c| acc * r2 + c);
    1.0 + r2 * p
}

/// Taylor series of asin for `|t| <= 0.5`; the first omitted term is below 1e-12.
fn asin_series(t: f64) -> f64 {
    const TERMS: usize = 16;
    // a_0 = 1, a_{n+1} = a_n · (2n+1)² / ((2n+2)(2n+3)).
    const C: [f64; TERMS] = {
        let mut c = [1.0; TERMS];
        let mut n = 0;
        while n + 1 < TERMS {
            let k = (2 * n + 1) as f64;
            c[n + 1] = c[n] * k * k / ((k + 1.0) * (k + 2.0));
            n += 1;
        }
        c
    };
    let t2 = t * t;
    t * C.iter().rev().fold(0.0, |acc, &c| acc * t2 + c)
}

/// Square root of a non-negative `v` to about 48 bits: one Newton step from the `f32` root.
///
/// `v` must be zero, infinite, NaN, or normal, which covers every sum of squares of finite `f32`s.
pub(super) fn sqrt_f64(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    // Write v = m · 4^k with m in [1, 4), so the `f32` seed neither overflows nor underflows.
    let pow2 = |n: i64| f64::from_bits(((1023 + n) as u64) << 52);
    let k = (((v.to_bits() >> 52) as i64) - 1023) >> 1;
    let m = v * pow2(-2 * k);
    let s = f64::from(sqrt(m as f32));
    0.5 * (s + m / s) * pow2(k)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use core::f32::consts::{FRAC_PI_2, PI};

    /// Distance between two finite floats in units in the last place.
    fn ulps(a: f32, b: f32) -> u32 {
        let key = |x: f32| {
            let bits = x.to_bits() as i32;
            if bits < 0 { i32::MIN - bits } else { bits }
        };
        key(a).abs_diff(key(b))
    }

    fn assert_close(what: &str, x: f32, got: f32, want: f32) {
        if want.is_nan() {
            assert!(got.is_nan(), "{what}({x:e}) = {got:e}, want NaN");
            return;
        }
        assert!(
            (got - want).abs() < 1e-6 * want.abs().max(1.0) && ulps(got, want) <= 1,
            "{what}({x:e}) = {got:e}, want {want:e} ({} ulps)",
            ulps(got, want)
        );
    }

    const EDGES: [f32; 19] = [
        0.0,
        -0.0,
        FRAC_PI_2,
        -FRAC_PI_2,
        PI,
        -PI,
        1.0,
        -1.0,
        0.5,
        1e-30,
        1e4,
        -1e5,
        1e6,
        12_345_678.0,
        1e20,
        -3e30,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::NAN,
    ];

    fn sweep() -> impl Iterator<Item = f32> {
        let dense = (-200_000..=200_000).map(|i| i as f32 * 5e-4);
        // Every 2^20th bit pattern covers all exponents, including huge angles.
        let spread = (0..=u32::MAX).step_by(1 << 20).map(f32::from_bits);
        EDGES
            .into_iter()
            .chain([f32::INFINITY, f32::NEG_INFINITY])
            .chain(dense)
            .chain(spread)
    }

    #[test]
    fn sqrt_is_correctly_rounded() {
        let inputs = (0..=0x7F80_0000u32).step_by(997).map(f32::from_bits);
        for x in inputs.chain([f32::MAX, f32::from_bits(1), 2.0, 4.0]) {
            assert_eq!(
                sqrt(x).to_bits(),
                std::primitive::f32::sqrt(x).to_bits(),
                "sqrt({x:e})"
            );
        }
        assert_eq!(sqrt(-0.0).to_bits(), (-0.0f32).to_bits());
        assert!(sqrt(-1.0).is_nan());
        assert!(sqrt(-f32::MIN_POSITIVE).is_nan());
        assert!(sqrt(f32::NAN).is_nan());
        assert_eq!(sqrt(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn sin_cos_matches_std() {
        for x in sweep() {
            let (sin, cos) = sin_cos(x);
            assert_close("sin", x, sin, std::primitive::f32::sin(x));
            assert_close("cos", x, cos, std::primitive::f32::cos(x));
        }
        assert_eq!(sin_cos(-0.0).0.to_bits(), (-0.0f32).to_bits());
        assert_eq!(sin_cos(0.0), (0.0, 1.0));
    }

    #[test]
    fn tan_matches_std() {
        for x in sweep() {
            assert_close("tan", x, tan(x), std::primitive::f32::tan(x));
        }
    }

    #[test]
    fn acos_matches_std() {
        let dense = (-100_000..=100_000).map(|i| i as f32 * 1e-5);
        let edges = [
            1.0,
            -1.0,
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0 - f32::EPSILON / 2.0,
            -1.0 + 1e-7,
        ];
        let outside = [
            1.000_001,
            -1.000_001,
            2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for x in dense.chain(edges).chain(outside) {
            assert_close("acos", x, acos(x), std::primitive::f32::acos(x));
        }
        assert_eq!(acos(1.0), 0.0);
        assert_eq!(acos(-1.0), PI);
    }

    #[test]
    fn sqrt_f64_covers_squares_of_all_finite_f32s() {
        let tiny = f64::from(f32::from_bits(1));
        let max = f64::from(f32::MAX);
        for x in [tiny, 1e-30, 0.3, 1.0, 7.0, 1e30, max] {
            let root = sqrt_f64(x * x);
            assert!((root - x).abs() <= x * 1e-14, "sqrt_f64({x:e}²) = {root:e}");
        }
        assert_eq!(sqrt_f64(4.0 * max * max), 2.0 * max);
        assert_eq!(sqrt_f64(0.0), 0.0);
        assert_eq!(sqrt_f64(f64::INFINITY), f64::INFINITY);
        assert!(sqrt_f64(f64::NAN).is_nan());
    }
}
