from typing import Callable, Tuple

import jax.numpy as jnp


SUPPORTED_PENFS = (
    "haa2205",
    "gaa2205",
    "111105",
    "111100",
    "111101",
    "gaa2200",
    "haa2200",
    "gaa2201",
    "haa2201",
)


def normalize_penf_name(penf: str) -> str:
    key = str(penf).strip().lower().replace("spf", "")
    if key not in SUPPORTED_PENFS:
        raise ValueError(f"Unsupported shapeak_penf: {penf}. Supported: {', '.join(SUPPORTED_PENFS)}")
    return key


def _validate_eta(eta: float):
    if eta < 0:
        raise ValueError("eta must be non-negative.")


def _validate_a_positive(a: float):
    if a <= 0:
        raise ValueError("Parameter 'a' must be > 0 for this SPF family.")


def _validate_a_gt_half(a: float):
    if a <= 0.5:
        raise ValueError("Parameter 'a' must be > 0.5 for haa2200.")


def spf_haa2205(x: jnp.ndarray, a: float) -> jnp.ndarray:
    left = 0.5 * (a * a - (x - a) ** 2)
    right = 0.5 * (a * a - (x + a - 1.0) ** 2)
    return jnp.sum(jnp.where(x <= 0.5, left, right), axis=-1)


def prox_spf_haa2205(z: jnp.ndarray, a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)
    _validate_a_positive(a)

    if eta > 0.5 / a:
        return (z >= 0.5).astype(z.dtype)

    eta1 = 1.0 - eta
    if abs(eta1) < 1e-12:
        eta1 = 1e-12

    tmp1 = z - a * eta
    tmp2 = z + (a - 1.0) * eta

    x = jnp.zeros_like(z)
    x = jnp.where((tmp1 > 0.0) & (z <= 0.5), tmp1 / eta1, x)
    x = jnp.where((z > 0.5) & (tmp2 < eta1), tmp2 / eta1, x)
    x = jnp.where(tmp2 >= eta1, 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_gaa2205(x: jnp.ndarray, a: float) -> jnp.ndarray:
    left = 0.5 * ((x + a) ** 2 - a * a)
    right = 0.5 * ((x - 1.0 - a) ** 2 - a * a)
    return jnp.sum(jnp.where(x <= 0.5, left, right), axis=-1)


def prox_spf_gaa2205(z: jnp.ndarray, a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)
    _validate_a_positive(a)

    if eta > 0.5 / a:
        return (z >= 0.5).astype(z.dtype)

    eta1 = 1.0 + eta
    if abs(eta1) < 1e-12:
        eta1 = 1e-12

    tmp1 = z - a * eta
    tmp2 = z + (1.0 + a) * eta

    x = jnp.zeros_like(z)
    x = jnp.where((tmp1 > 0.0) & (z <= 0.5), tmp1 / eta1, x)
    x = jnp.where((z > 0.5) & (tmp2 < eta1), tmp2 / eta1, x)
    x = jnp.where(tmp2 >= eta1, 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_111105(x: jnp.ndarray, _a: float) -> jnp.ndarray:
    return jnp.sum(0.5 - jnp.abs(x - 0.5), axis=-1)


def prox_spf_111105(z: jnp.ndarray, _a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)

    if eta > 0.5:
        return (z >= 0.5).astype(z.dtype)

    x = jnp.zeros_like(z)
    x = jnp.where((z > eta) & (z <= 0.5), z - eta, x)
    ze = z + eta
    x = jnp.where((z > 0.5) & (ze < 1.0), ze, x)
    x = jnp.where(ze >= 1.0, 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_111100(x: jnp.ndarray, _a: float) -> jnp.ndarray:
    g = jnp.where(x == 0.0, 0.0, 1.0 - x)
    return jnp.sum(g, axis=-1)


def prox_spf_111100(z: jnp.ndarray, _a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)

    if eta > 0.5:
        return (z >= 0.5).astype(z.dtype)

    ze = z + eta
    thr = jnp.sqrt(jnp.maximum(2.0 * eta, 0.0))
    x = jnp.zeros_like(z)
    x = jnp.where(ze > thr, ze, x)
    x = jnp.where(ze >= 1.0, 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_111101(x: jnp.ndarray, _a: float) -> jnp.ndarray:
    g = jnp.where(x == 1.0, 0.0, x)
    return jnp.sum(g, axis=-1)


def prox_spf_111101(z: jnp.ndarray, _a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)

    if eta > 0.5:
        return (z >= 0.5).astype(z.dtype)

    ze = z - eta
    x = jnp.zeros_like(z)
    x = jnp.where(ze > 0.0, ze, x)
    one_thr = 1.0 - jnp.sqrt(jnp.maximum(2.0 * eta, 0.0))
    x = jnp.where(ze >= one_thr, 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_gaa2200(x: jnp.ndarray, a: float) -> jnp.ndarray:
    _validate_a_positive(a)
    g = 0.5 * ((x - (1.0 + a)) ** 2 - a * a)
    return jnp.sum(jnp.where(x > 0.0, g, 0.0), axis=-1)


def prox_spf_gaa2200(z: jnp.ndarray, a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)
    _validate_a_positive(a)

    if eta > 0.5 / a:
        return (z >= 0.5).astype(z.dtype)

    eta1 = 1.0 + eta
    if abs(eta1) < 1e-12:
        eta1 = 1e-12

    ze = z + (1.0 + a) * eta
    thr = jnp.sqrt(jnp.maximum(eta * (eta + 1.0) * (2.0 * a + 1.0), 0.0))
    x = jnp.zeros_like(z)
    x = jnp.where(ze > thr, ze / eta1, x)
    x = jnp.where(ze >= eta1, 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_haa2200(x: jnp.ndarray, a: float) -> jnp.ndarray:
    _validate_a_gt_half(a)
    g = 0.5 * (a * a - (x + (a - 1.0)) ** 2)
    return jnp.sum(jnp.where(x > 0.0, g, 0.0), axis=-1)


def prox_spf_haa2200(z: jnp.ndarray, a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)
    _validate_a_gt_half(a)

    if eta > 0.5 / a:
        return (z >= 0.5).astype(z.dtype)

    eta1 = 1.0 - eta
    if abs(eta1) < 1e-12:
        eta1 = 1e-12

    ze = z + (a - 1.0) * eta
    thr = jnp.sqrt(jnp.maximum(eta * (1.0 - eta) * (2.0 * a - 1.0), 0.0))
    x = jnp.zeros_like(z)
    x = jnp.where(ze > thr, ze / eta1, x)
    x = jnp.where(ze >= eta1, 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_gaa2201(x: jnp.ndarray, a: float) -> jnp.ndarray:
    _validate_a_positive(a)
    g = 0.5 * ((x + a) ** 2 - a * a)
    return jnp.sum(jnp.where(x < 1.0, g, 0.0), axis=-1)


def prox_spf_gaa2201(z: jnp.ndarray, a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)
    _validate_a_positive(a)

    if eta > 0.5 / a:
        return (z >= 0.5).astype(z.dtype)

    eta1 = 1.0 + eta
    if abs(eta1) < 1e-12:
        eta1 = 1e-12

    ze = z - a * eta
    x = jnp.zeros_like(z)
    x = jnp.where(ze > 0.0, ze / eta1, x)

    thr = jnp.sqrt(jnp.maximum(eta * (eta + 1.0) * (2.0 * a + 1.0), 0.0))
    x = jnp.where(z >= (eta1 - thr), 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def spf_haa2201(x: jnp.ndarray, a: float) -> jnp.ndarray:
    _validate_a_gt_half(a)
    g = 0.5 * (a * a - (x - a) ** 2)
    return jnp.sum(jnp.where(x < 1.0, g, 0.0), axis=-1)


def prox_spf_haa2201(z: jnp.ndarray, a: float, eta: float) -> jnp.ndarray:
    _validate_eta(eta)
    _validate_a_gt_half(a)

    if eta > 0.5 / a:
        return (z >= 0.5).astype(z.dtype)

    eta1 = 1.0 - eta
    if abs(eta1) < 1e-12:
        eta1 = 1e-12

    ze = z - a * eta
    x = jnp.zeros_like(z)
    x = jnp.where(ze > 0.0, ze / eta1, x)

    thr = jnp.sqrt(jnp.maximum(eta * (1.0 - eta) * (2.0 * a - 1.0), 0.0))
    x = jnp.where(z >= (eta1 - thr), 1.0, x)
    return jnp.clip(x, 0.0, 1.0)


def get_penalty_ops(penf: str, a: float) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray, float], jnp.ndarray], str]:
    key = normalize_penf_name(penf)

    if key == "haa2205":
        return lambda w: spf_haa2205(w, a), lambda z, eta: prox_spf_haa2205(z, a, eta), key
    if key == "gaa2205":
        return lambda w: spf_gaa2205(w, a), lambda z, eta: prox_spf_gaa2205(z, a, eta), key
    if key == "111105":
        return lambda w: spf_111105(w, a), lambda z, eta: prox_spf_111105(z, a, eta), key
    if key == "111100":
        return lambda w: spf_111100(w, a), lambda z, eta: prox_spf_111100(z, a, eta), key
    if key == "111101":
        return lambda w: spf_111101(w, a), lambda z, eta: prox_spf_111101(z, a, eta), key
    if key == "gaa2200":
        return lambda w: spf_gaa2200(w, a), lambda z, eta: prox_spf_gaa2200(z, a, eta), key
    if key == "haa2200":
        return lambda w: spf_haa2200(w, a), lambda z, eta: prox_spf_haa2200(z, a, eta), key
    if key == "gaa2201":
        return lambda w: spf_gaa2201(w, a), lambda z, eta: prox_spf_gaa2201(z, a, eta), key
    if key == "haa2201":
        return lambda w: spf_haa2201(w, a), lambda z, eta: prox_spf_haa2201(z, a, eta), key

    raise ValueError(f"Unsupported shapeak_penf: {penf}")
