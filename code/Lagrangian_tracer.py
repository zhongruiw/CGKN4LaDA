import numpy as np
from mode_truc import truncate, inv_truncate

from numba import jit, prange
@jit(nopython=True, parallel=True, fastmath=True)
def for_loop_tracer(K, ens, L, N, dt, x, y, KX_flat, KY_flat, psi_hat_flat, sigma_xy):
    KX_flat_c = np.ascontiguousarray(KX_flat[None, :])
    KY_flat_c = np.ascontiguousarray(KY_flat[None, :])
    for e in prange(ens):  # Parallelize over ensemble members
        for i in range(1, N+1):
            exp_term = np.exp(1j * x[e, :, i-1][:, None] @ KX_flat_c + 1j * y[e, :, i-1][:, None] @ KY_flat_c)
            uk = (psi_hat_flat[e, i-1, :] * (1j) * KY_flat)
            vk = (psi_hat_flat[e, i-1, :] * (-1j) * KX_flat)

            # # ensure conjugate symmetric
            # if style == 'square' and psi_hat.shape[0] == K:
            #     uk[K//2::K] = 0; uk[K*K//2:K*(K//2 + 1)] = 0
            #     vk[K//2::K] = 0; vk[K*K//2:K*(K//2 + 1)] = 0 
            
            u = np.real(exp_term @ uk) / K**2
            v = np.real(exp_term @ vk) / K**2
            
            x[e, :, i] = x[e, :, i-1] + u * dt + np.random.randn(L) * sigma_xy * np.sqrt(dt)
            y[e, :, i] = y[e, :, i-1] + v * dt + np.random.randn(L) * sigma_xy * np.sqrt(dt)
            x[e, :, i] = np.mod(x[e, :, i], 2*np.pi)  # Periodic boundary conditions
            y[e, :, i] = np.mod(y[e, :, i], 2*np.pi)  # Periodic boundary conditions

    return x, y

# ##################### faster tracer loop with jax #################
# import jax
# import jax.numpy as jnp
# from jax_finufft import nufft2
# jax.config.update("jax_enable_x64", True) # Enable float64 in JAX for tight agreement
# def for_loop_tracer_jax(K, ens, L, N, dt, x, y, psi_hat, sigma_xy, key, eps=1e-9):
#     # decide real dtype from complex dtype
#     ctype = jnp.asarray(psi_hat).dtype
#     rtype = jnp.float64 if ctype == jnp.complex128 else jnp.float32

#     # JAX scalars
#     dt_r   = jnp.asarray(dt, dtype=rtype)
#     sig_r  = jnp.asarray(sigma_xy, dtype=rtype)
#     twopi  = jnp.asarray(2 * jnp.pi, dtype=rtype)
#     invK2  = jnp.asarray(1.0 / (K * K), dtype=rtype)
#     sdt    = jnp.sqrt(dt_r) * sig_r

#     # Initial positions
#     x0 = jnp.asarray(x[:, :, 0], dtype=rtype)
#     y0 = jnp.asarray(y[:, :, 0], dtype=rtype)

#     # Wavenumbers in FFT order (JAX fftfreq)
#     k_fft = (jnp.fft.fftfreq(K) * K).astype(rtype)
#     k_nat = jnp.fft.fftshift(k_fft)     
#     kx, ky = jnp.meshgrid(k_nat, k_nat)

#     # Time-major psi
#     psi_seq = jnp.swapaxes(jnp.asarray(psi_hat), 0, 1) # (N, ens, K, K)

#     def uv_one(psi_e, x_e, y_e):
#         psi_e = jnp.fft.fftshift(psi_e, axes=(0, 1))
#         uhat = 1j * ky * psi_e
#         vhat = -1j * kx * psi_e
#         xx = x_e % (2 * jnp.pi)
#         yy = y_e % (2 * jnp.pi)
#         u = nufft2(uhat, yy, xx, iflag=1, eps=eps).real * invK2
#         v = nufft2(vhat, yy, xx, iflag=1, eps=eps).real * invK2
#         return u, v

#     uv_batched = jax.vmap(uv_one, in_axes=(0, 0, 0), out_axes=(0, 0))

#     def step(carry, psi_t):
#         x_t, y_t, key_t = carry
#         u_t, v_t = uv_batched(psi_t, x_t, y_t)
#         key_t, kxkey, kykey = jax.random.split(key_t, 3)
#         xi_x = jax.random.normal(kxkey, shape=x_t.shape, dtype=rtype)
#         xi_y = jax.random.normal(kykey, shape=y_t.shape, dtype=rtype)
#         x_new = (x_t + dt_r * u_t + sdt * xi_x) % twopi
#         y_new = (y_t + dt_r * v_t + sdt * xi_y) % twopi
#         return (x_new, y_new, key_t), (x_new, y_new)

#     carry0 = (x0, y0, key)
#     (xf, yf, _), (xs_hist, ys_hist) = jax.lax.scan(step, carry0, psi_seq[:-1])
#     xs_hist = jnp.transpose(xs_hist, (1, 2, 0)) 
#     ys_hist = jnp.transpose(ys_hist, (1, 2, 0))
#     x_out = jnp.concatenate([x0[:, :, None], xs_hist], axis=-1)
#     y_out = jnp.concatenate([y0[:, :, None], ys_hist], axis=-1)
#     return x_out, y_out
# for_loop_tracer_jax = jax.jit(for_loop_tracer_jax, static_argnums=(0, 1, 2, 3), static_argnames=("eps",))


class Lagrange_tracer_model:
    """
    math of the model:
    v(x, t) =\sum_k{psi_hat_{1,k}(t) e^(ik·x) r_k}
    r_k=(ik_2,-ik_1). 

    """
    def __init__(self, K, sigma_xy, style='square'):
        """
        Parameters:
        - K: number of modes
        - sigma_xy: float, standard deviation of the noise
        - style: truncation style
        """
        Kx = np.fft.fftfreq(K) * K
        Ky = np.fft.fftfreq(K) * K
        KX, KY = np.meshgrid(Kx, Ky)
        KX_flat = KX.flatten()
        KY_flat = KY.flatten()
        KX_flat = KX_flat.astype(np.complex128)
        KY_flat = KY_flat.astype(np.complex128)

        self.K = K
        self.sigma_xy = sigma_xy
        self.style = style
        self.KX = KX
        self.KY = KY
        self.KX_flat = KX_flat
        self.KY_flat = KY_flat

    def forward(self, L, N, dt, x0, y0, psi_hat, interv=1, t_interv=None):
        """
        Integrates tracer locations using forward Euler method.
        
        There are to ways to get u,v from psi_hat.
        1) write the modified Fourier coefficients as psi_hat_{1,k} r_k, do ifft2. Then interpolate velocity on the grid points to (x,y).
        2) use (x,y) when computing the exponential components, then manually sum up.

        Parameters:
        - N: int, total number of steps
        - L: int, number of tracers
        - psi_hat: np.array of shape (K, K, N) or truncated, Fourier time series of the upper layer stream function.
        - dt: float, time step
        - x0:  Initial tracer locations in x of shape (L,)
        - y0:  Initial tracer locations in y of shape (L,)
        - interv:  int, wave number inverval for calculating u, v field 
        - t_interv: int, time interval for calculate and save u, v field
        """
        K = self.K
        KX = self.KX
        KY = self.KY
        style = self.style
        x = np.zeros((L, N+1))
        y = np.zeros((L, N+1))
        x[:,0] = x0
        y[:,0] = y0
        ut = np.zeros((K//interv, K//interv, N//t_interv))  
        vt = np.zeros((K//interv, K//interv, N//t_interv))

        if psi_hat.shape[0] < K and style == 'square':
            r_cut = (psi_hat.shape[0] - 1) // 2
            KX_flat = truncate(KX, r_cut, style=style)
            KY_flat = truncate(KY, r_cut, style=style)
            psi_hat_flat = np.reshape(psi_hat, (psi_hat.shape[0]**2, -1), order='F')
        elif psi_hat.shape[0] == K:
            KX_flat = KX.flatten(order='F')
            KY_flat = KY.flatten(order='F')
            psi_hat_flat = np.reshape(psi_hat, (K**2, -1), order='F')
        else:
            raise Exception("unknown truncation style")

        l = 0
        for i in range(1, N+1):
            exp_term = np.exp(1j * x[:, i-1][:,None] @ KX_flat[None,:] + 1j * y[:, i-1][:,None] @ KY_flat[None,:])
            uk = (psi_hat_flat[:, i-1] * (1j) * KY_flat)
            vk = (psi_hat_flat[:, i-1] * (-1j) * KX_flat)

            # ensure conjugate symmetric
            if style == 'square' and psi_hat.shape[0] == K:
                uk[K//2::K] = 0; uk[K*K//2:K*(K//2 + 1)] = 0
                vk[K//2::K] = 0; vk[K*K//2:K*(K//2 + 1)] = 0 

            u = np.squeeze(exp_term @ uk[:,None]) / K**2
            v = np.squeeze(exp_term @ vk[:,None]) / K**2
            u = np.real(u)
            v = np.real(v)

            # max_imag_abs = max(np.max(np.abs(np.imag(u))), np.max(np.abs(np.imag(v))))
            # if max_imag_abs > 1e-10:
            #     raise Exception("get significant imaginary parts, check the ifft2")
            # else:
            #     u = np.real(u)
            #     v = np.real(v)
            
            x[:, i] = x[:, i-1] + u * dt + np.random.randn(L) * self.sigma_xy * np.sqrt(dt)
            y[:, i] = y[:, i-1] + v * dt + np.random.randn(L) * self.sigma_xy * np.sqrt(dt)
            x[:, i] = np.mod(x[:, i], 2*np.pi)  # Periodic boundary conditions
            y[:, i] = np.mod(y[:, i], 2*np.pi)  # Periodic boundary conditions

            if np.mod(i,t_interv) == 0:
                if psi_hat.shape[0] == K:
                    psi_hat_KK = psi_hat[:, :, i-1]
                else:
                    psi_hat_KK = inv_truncate(psi_hat_flat[:, i-1][:,None], r_cut, K, style)[:,:,0]

                # using built-in ifft2
                u_ifft = np.fft.ifft2(psi_hat_KK * 1j * KY)
                ut[:,:,l] = u_ifft[::interv, ::interv] # only save the sparsely sampled grids
                v_ifft = np.fft.ifft2(psi_hat_KK * (-1j) * KX)
                vt[:,:,l] = v_ifft[::interv, ::interv] # only save the sparsely sampled grids
                        
                l += 1

        return x, y, ut, vt

    def forward_ens(self, ens, L, N, dt, x0, y0, psi_hat, eps_nufft=1e-9):
        '''
        assuming all necessary mode truncations, if there any, are made before pass to this function

        Parameters:
        - N: int, total number of time steps
        - L: int, number of tracers
        - psi_hat: np.array of shape (ens, N+1, K, K), Fourier time series of the upper layer stream function.
        - dt: float, time step
        - x0:  Initial tracer locations in x of shape (ens, L)
        - y0:  Initial tracer locations in y of shape (ens, L)
        '''
        K = self.K
        KX_flat = self.KX_flat
        KY_flat = self.KY_flat
        sigma_xy = self.sigma_xy
        x = np.zeros((ens, L, N+1))
        y = np.zeros((ens, L, N+1))
        x[:,:,0] = x0
        y[:,:,0] = y0

        psi_hat_flat = np.reshape(psi_hat, (ens, N+1, -1))
        x, y = for_loop_tracer(K, ens, L, N, dt, x, y, KX_flat, KY_flat, psi_hat_flat, sigma_xy)

        # key = jax.random.PRNGKey(np.random.randint(-2147483648, 2147483648))
        # xJ, yJ = for_loop_tracer_jax(K, ens, L, N, dt, x, y, psi_hat, sigma_xy, key, eps=eps_nufft)
        # x = np.array(xJ); y = np.array(yJ)

        return x, y