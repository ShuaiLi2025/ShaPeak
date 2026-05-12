import time
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from jax.experimental import sparse

from .shapeak_spf import get_penalty_ops


class SHAPEAK:
    """Step-JIT ShapeaK solver for QUBO/Max-Cut style binary quadratic models."""

    def __init__(
            self,
            n_vars: int,
            Q_indices: np.ndarray,
            Q_values: np.ndarray,
            c: np.ndarray,
            pen: float,
            primal_lr: float = 0.01,
            batch_size: int = 1,
            max_iters: int = 5000,
            seed: int = 0,
            preQ: int = 1,
            sigma: float = 0.5,
            it0: int = 100,
            penrt: float = 1.25,
            tol: Optional[float] = None,
            check: bool = False,
            penf: str = "haa2205",
            a: float = 2.5,
            optimizer_type: str = "rmsprop",
            verbose: bool = True,
            cg_tol: float = 1e-8,
            cg_iters: int = 6,
            n0: int = 6000,
            prob: str = "qubo",
            grate: Optional[float] = None,
            gfreq: Optional[int] = None,
            drate: Optional[float] = None,
            dfreq: Optional[int] = None,
            x0: Optional[np.ndarray] = None,
            y0: Optional[np.ndarray] = None,
            y_init: Optional[float] = None,
            y_random_scale: float = 0.0,
            H_precond: Optional[np.ndarray] = None,
    ):
        assert preQ in {0, 1, 2}, "preQ must be 0, 1, or 2."
        assert optimizer_type in {"rmsprop", "adam"}, "Invalid optimizer type."
        if y_random_scale < 0:
            raise ValueError("y_random_scale must be non-negative.")

        self.key = jax.random.PRNGKey(seed)
        self.n = int(n_vars)
        self.batch_size = int(batch_size)
        self.max_iters = int(max_iters)
        self.preQ = int(preQ)
        self.sigma = float(sigma)
        self.pen = float(pen)
        self.pen0 = float(pen)
        self.it0 = int(it0)
        self.penrt = float(penrt)
        self.check = bool(check)
        self.a = float(a)
        self.penfun, self.proxfun, self.penf = get_penalty_ops(penf, self.a)
        self.verbose = bool(verbose)
        self.cg_tol = float(cg_tol)
        self.cg_iters = int(cg_iters)
        self.n0 = int(n0)
        self.optimizer_type = optimizer_type
        self.primal_lr = float(primal_lr)
        self.y_init = None if y_init is None else float(y_init)
        self.y_random_scale = float(y_random_scale)
        self.tol = float(1e-5 * np.sqrt(n_vars) if tol is None else tol)

        self.grate, self.gfreq, self.drate, self.dfreq = self._set_schedule(prob, self.n)
        if grate is not None:
            self.grate = float(grate)
        if gfreq is not None:
            self.gfreq = int(gfreq)
        if drate is not None:
            self.drate = float(drate)
        if dfreq is not None:
            self.dfreq = int(dfreq)

        self.Q_indices = np.asarray(Q_indices, dtype=np.int32)
        self.Q_values = np.asarray(Q_values, dtype=np.float32)
        self.Q = sparse.BCOO(
            (self.Q_values, jnp.column_stack(self.Q_indices)),
            shape=(self.n, self.n),
        )

        q_t_indices = np.flip(self.Q_indices, axis=0)
        self.Q_T = sparse.BCOO(
            (self.Q_values, jnp.column_stack(q_t_indices)),
            shape=(self.n, self.n),
        )

        self.c = jnp.asarray(c, dtype=jnp.float32)

        self.H_precond_dense = None
        if self.preQ in {1, 2}:
            if H_precond is not None:
                h = np.asarray(H_precond, dtype=np.float32)
                if h.shape != (self.n, self.n):
                    raise ValueError(f"H_precond shape must be ({self.n}, {self.n}), got {h.shape}.")
                self.H_precond_dense = jnp.asarray(h, dtype=jnp.float32)
            elif self.n <= self.n0:
                self.H_precond_dense = self.Q.todense() + self.Q_T.todense()

        self.x, self.w, self.y = self._init_variables(x0, y0)
        self._setup_optimizer()

        self.objVal = self._obj_single(jnp.round(jnp.clip(self.w[0], 0.0, 1.0)))
        self.incumbent = jnp.round(jnp.clip(self.w[0], 0.0, 1.0))

        self.objVal_record = [float(self.objVal)]
        self.timing_record = [0.0]
        self.Error = []
        self.start_time = None
        self.solving_time = None

        self.mark = 0
        self._explicit_inq = self.preQ == 1 and self.H_precond_dense is not None and self.n <= self.n0
        self._A_inv = None
        if self._explicit_inq:
            self._refresh_preconditioner()
        if self.verbose:
            device_list = ", ".join(str(d) for d in jax.devices())
            print(f"[shapeakadmm] JAX backend: {jax.default_backend()} | devices: {device_list}")

        self._jit_step_fn = self._build_jit_step_fn()

    @staticmethod
    def _set_schedule(prob: str, n: int) -> Tuple[float, int, float, int]:
        if prob in {"recovery", "mimo", "1bitmimo"}:
            return 1.2, 10 if prob != "1bitmimo" else 5, 1.1, 100 if prob != "mimo" else 10
        if prob == "qubo":
            return 1.2, 10, 1.1, 100 if n <= 2000 else 10
        return 1.2, 10, 1.1, 10

    def _setup_optimizer(self):
        if self.preQ != 0:
            self.optimizer = None
            self.opt_state = None
            return

        if self.optimizer_type == "rmsprop":
            self.optimizer = optax.rmsprop(self.primal_lr, decay=0.98, eps=1e-8, momentum=0.91)
        else:
            self.optimizer = optax.adam(self.primal_lr, b1=0.9, b2=0.999, eps=1e-8)
        self.opt_state = self.optimizer.init(self.w)

    def _init_x_batch(self, x0: Optional[np.ndarray]) -> jnp.ndarray:
        key_x, _ = jax.random.split(self.key)
        if x0 is None:
            return jax.random.uniform(key_x, (self.batch_size, self.n), dtype=jnp.float32)

        x0_arr = np.asarray(x0, dtype=np.float32)
        if x0_arr.ndim == 1:
            if x0_arr.shape[0] != self.n:
                raise ValueError(f"x0 length must be {self.n}.")
            return jnp.tile(jnp.asarray(x0_arr)[jnp.newaxis, :], (self.batch_size, 1))

        if x0_arr.ndim == 2:
            if x0_arr.shape == (1, self.n):
                return jnp.tile(jnp.asarray(x0_arr, dtype=jnp.float32), (self.batch_size, 1))
            if x0_arr.shape == (self.batch_size, self.n):
                return jnp.asarray(x0_arr, dtype=jnp.float32)
            raise ValueError(
                f"x0 shape must be ({self.batch_size}, {self.n}) or (1, {self.n}), got {x0_arr.shape}."
            )

        raise ValueError("x0 must be a 1D vector or a 2D matrix.")

    def _format_y0(self, y0: np.ndarray) -> jnp.ndarray:
        y0_arr = np.asarray(y0, dtype=np.float32)
        if y0_arr.ndim == 0:
            return jnp.full((self.batch_size, self.n), float(y0_arr), dtype=jnp.float32)
        if y0_arr.ndim == 1:
            if y0_arr.shape[0] != self.n:
                raise ValueError(f"y0 length must be {self.n}.")
            return jnp.tile(jnp.asarray(y0_arr)[jnp.newaxis, :], (self.batch_size, 1))
        if y0_arr.ndim == 2:
            if y0_arr.shape == (self.batch_size, self.n):
                return jnp.asarray(y0_arr, dtype=jnp.float32)
            if y0_arr.shape == (1, self.n):
                return jnp.tile(jnp.asarray(y0_arr, dtype=jnp.float32), (self.batch_size, 1))
            raise ValueError(
                f"y0 shape must be ({self.batch_size}, {self.n}) or (1, {self.n}), got {y0_arr.shape}."
            )
        raise ValueError("y0 must be a scalar, a 1D vector, or a 2D matrix.")

    def _init_variables(self, x0: Optional[np.ndarray], y0: Optional[np.ndarray]):
        _, key_y = jax.random.split(self.key)
        x = self._init_x_batch(x0)
        w = x

        if y0 is not None:
            y = self._format_y0(y0)
        elif self.y_init is None:
            y = -self._grad_f_batch(w)
        elif self.y_random_scale > 0:
            y = self.y_init + self.y_random_scale * jax.random.normal(
                key_y, (self.batch_size, self.n), dtype=jnp.float32
            )
        else:
            y = jnp.full((self.batch_size, self.n), self.y_init, dtype=jnp.float32)
        return x, w, y

    def _hessian_mv(self, v: jnp.ndarray) -> jnp.ndarray:
        return self.Q @ v + self.Q_T @ v

    def _precond_mv(self, v: jnp.ndarray) -> jnp.ndarray:
        if self.H_precond_dense is not None:
            return self.H_precond_dense @ v
        return self._hessian_mv(v)

    def _grad_f_batch(self, w: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda wi: self._hessian_mv(wi) + self.c)(w)

    def _obj_single(self, x: jnp.ndarray) -> jnp.ndarray:
        qx = self.Q @ x
        return jnp.dot(x, qx) + jnp.dot(self.c, x)

    def _batch_obj(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self._obj_single)(x)

    def _penfun(self, w: jnp.ndarray) -> jnp.ndarray:
        return self.penfun(w)

    def _proxfun(self, z: jnp.ndarray, eta: float) -> jnp.ndarray:
        eta = max(float(eta), 0.0)
        return self.proxfun(z, eta)

    def _refresh_preconditioner(self):
        A = self.H_precond_dense + self.sigma * jnp.eye(self.n, dtype=jnp.float32)
        self._A_inv = jnp.linalg.inv(A)

    def _cg_single_sigma(self, b: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        def solve(_):
            def cond_fun(state):
                i, _, _, _, e, t = state
                return (i < self.cg_iters) & (e > self.cg_tol * t)

            def body_fun(state):
                i, x, r, p, e, t = state
                w = sigma * p + self._precond_mv(p)
                denom = jnp.dot(p, w) + 1e-12
                alpha = e / denom
                x_new = x + alpha * p
                r_new = r - alpha * w
                e_new = jnp.dot(r_new, r_new)
                beta = e_new / (e + 1e-12)
                p_new = r_new + beta * p
                return i + 1, x_new, r_new, p_new, e_new, t

            x0 = jnp.zeros_like(b)
            r0 = b
            e0 = jnp.dot(r0, r0)
            state0 = (0, x0, r0, r0, e0, e0)
            _, x, _, _, _, _ = lax.while_loop(cond_fun, body_fun, state0)
            return x

        return lax.cond(
            jnp.dot(b, b) <= 1e-20,
            lambda _: jnp.zeros_like(b),
            solve,
            operand=None,
        )

    def _check_opt(self, x_bin: jnp.ndarray) -> jnp.ndarray:
        x = np.array(x_bin, dtype=np.float32)
        fx = float(self._obj_single(jnp.asarray(x)))

        ones = np.where(x == 1)[0]
        zeros = np.where(x == 0)[0]
        if ones.size > zeros.size:
            ones, zeros = zeros, ones

        for idx in ones:
            y = x.copy()
            y[idx] = 1.0 - y[idx]
            fy = float(self._obj_single(jnp.asarray(y)))
            if fy < fx:
                return jnp.asarray(y)

        for idx in zeros:
            y = x.copy()
            y[idx] = 1.0 - y[idx]
            fy = float(self._obj_single(jnp.asarray(y)))
            if fy < fx:
                return jnp.asarray(y)

        return jnp.asarray(x)

    def _finish_step(
            self,
            w: jnp.ndarray,
            y: jnp.ndarray,
            x: jnp.ndarray,
            incumbent: jnp.ndarray,
            sigma: jnp.ndarray,
            eta: float,
    ):
        w_prev = w
        w_new = self._proxfun(x + y / sigma, eta)
        xy = x - w_new
        y_new = y + sigma * xy

        sx_batch = self._penfun(w_new)
        sx = jnp.min(sx_batch)

        err_xy = jnp.sum(xy * xy, axis=1)
        err_w = jnp.sum((w_new - w_prev) ** 2, axis=1)
        norm_w = jnp.sum(w_new * w_new, axis=1)
        err_batch = jnp.maximum(err_xy, err_w) / (1.0 + norm_w)
        err = jnp.max(err_batch)

        int_w = jnp.round(jnp.clip(w_new, 0.0, 1.0))
        cand = jnp.concatenate([int_w, incumbent[jnp.newaxis, :]], axis=0)
        objs = self._batch_obj(cand)
        best_idx = jnp.argmin(objs)
        best_obj = objs[best_idx]
        best_sol = cand[best_idx]
        return x, w_new, y_new, best_obj, best_sol, sx, err

    def _build_jit_step_fn(self):
        if self.preQ == 0:
            @partial(jax.jit, static_argnames=("eta",))
            def step(w, y, incumbent, opt_state, sigma, eta: float):
                gyy = self._grad_f_batch(w) + y
                grad_scaled = gyy / jnp.maximum(sigma, 1e-8)
                updates, opt_state_new = self.optimizer.update(grad_scaled, opt_state, w)
                x = optax.apply_updates(w, updates)
                x, w_new, y_new, best_obj, best_sol, sx, err = self._finish_step(
                    w, y, x, incumbent, sigma, eta
                )
                return x, w_new, y_new, opt_state_new, best_obj, best_sol, sx, err

            return step

        if self._explicit_inq:
            @partial(jax.jit, static_argnames=("eta",))
            def step(w, y, incumbent, a_inv, sigma, eta: float):
                gyy = self._grad_f_batch(w) + y
                d = gyy @ a_inv.T
                x = w - d
                return self._finish_step(w, y, x, incumbent, sigma, eta)

            return step

        @partial(jax.jit, static_argnames=("eta",))
        def step(w, y, incumbent, sigma, eta: float):
            gyy = self._grad_f_batch(w) + y
            d = jax.vmap(lambda bi: self._cg_single_sigma(bi, sigma))(gyy)
            x = w - d
            return self._finish_step(w, y, x, incumbent, sigma, eta)

        return step

    def optimize(self):
        self.start_time = time.perf_counter()
        no_improve_patience = 1500
        no_improve_steps = 0

        for iter_idx in range(1, self.max_iters + 1):
            sigma = jnp.asarray(self.sigma, dtype=jnp.float32)
            eta = max(float(self.pen / max(self.sigma, 1e-8)), 0.0)

            if self.preQ == 0:
                (
                    self.x,
                    self.w,
                    self.y,
                    self.opt_state,
                    best_obj,
                    best_sol,
                    sx,
                    err,
                ) = self._jit_step_fn(self.w, self.y, self.incumbent, self.opt_state, sigma, eta)
            elif self._explicit_inq:
                self.x, self.w, self.y, best_obj, best_sol, sx, err = self._jit_step_fn(
                    self.w, self.y, self.incumbent, self._A_inv, sigma, eta
                )
            else:
                self.x, self.w, self.y, best_obj, best_sol, sx, err = self._jit_step_fn(
                    self.w, self.y, self.incumbent, sigma, eta
                )

            best_obj = float(best_obj)
            sx = float(sx)
            err = float(err)
            self.Error.append(err)

            if best_obj < self.objVal_record[-1]:
                self.objVal_record.append(best_obj)
                self.timing_record.append(time.perf_counter() - self.start_time)
                self.incumbent = best_sol
                self.objVal = jnp.asarray(best_obj)
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if self.verbose and (iter_idx < 10 or iter_idx % 10 == 0):
                cur_best = float(self.objVal_record[-1])
                print(
                    f"{iter_idx:4d}  best:{cur_best:.6f}  err:{err:.2e}  "
                    f"bin:{sx:.2e}  time:{time.perf_counter() - self.start_time:.3f}"
                )

            if no_improve_steps >= no_improve_patience:
                if self.verbose:
                    print(
                        f"Early stop at iter {iter_idx}: "
                        f"best obj unchanged for {no_improve_patience} iterations."
                    )
                break

            if sx <= 0.0 and err < self.tol:
                if self.check and iter_idx > 2000 and self.n <= 10_000:
                    z = self._check_opt(self.incumbent)
                    self.pen = self.pen0 / max(np.log(iter_idx), 1.0)
                    if int(jnp.sum(jnp.abs(z - self.incumbent))) == 0:
                        break
                    self.incumbent = z
                    self.w = jnp.tile(z[jnp.newaxis, :], (self.batch_size, 1))
                    self.x = self.w
                else:
                    break

            self.mark += int(sx <= 0.0)

            if iter_idx % self.it0 == 0 and sx > 0.0:
                self.pen = self.pen * self.penrt

            change = False
            if self.mark > 5 and err > self.tol and iter_idx % self.gfreq == 0:
                self.sigma = min(1e4, self.sigma * self.grate)
                change = True
            elif sx > 0.0 and err < self.tol and iter_idx % self.dfreq == 0:
                self.sigma = max(1e-3, self.sigma / self.drate)
                self.pen = self.pen * self.penrt
                change = True

            if change and self._explicit_inq:
                self._refresh_preconditioner()

        self.solving_time = time.perf_counter() - self.start_time
