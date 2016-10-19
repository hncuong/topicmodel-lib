from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
ctypedef np.uint32_t uitype_t

@cython.boundscheck(False)
@cython.wraparound(False)

def sampling ( np.ndarray[uitype_t, ndim=2] Ndk,
               np.ndarray[dtype_t, ndim=2] Nkw_mean,
	       np.ndarray[dtype_t, ndim=2] Ndk_mean,
               np.ndarray[dtype_t, ndim=2] expElogbeta,
               np.ndarray[dtype_t, ndim=1] uni_rvs,
	       list z,
               list wordtks,
	       list lengths,
	       double alpha,
	       double update_unit,
               int num_sim,
               int burn_in ):

    #if not phi.flags.f_contiguous: phi = phi.copy('F')
    #if not Adk.flags.c_contiguous: phi = phi.copy('C')
    ##if not Bkw.flags.f_contiguous: phi = phi.copy('F')

    cdef Py_ssize_t D = Ndk.shape[0]
    cdef Py_ssize_t K = Ndk.shape[1]
    cdef Py_ssize_t W = Nkw_mean.shape[1]
    cdef Py_ssize_t d, w, k, sim, zInit, zOld, zNew
    cdef Py_ssize_t rc_start = 0, rc_mid, rc_stop = K
    cdef double prob_sum, uni_rv
    cdef Py_ssize_t uni_idx = 0, tks_idx = 0
    cdef np.ndarray[dtype_t, ndim=1] cumprobs = np.linspace(0,1,K+1)[0:K]
    cdef np.ndarray[uitype_t, ndim=1] zd

    # Make sure the counts are initialised to zero
    # Ndk.fill(0)
    # Nkw_mean.fill(0)
    # Initialise the z_id for each document in the batch
    for d in range(D):
        zd = np.zeros(lengths[d], dtype=np.uint32)
        tks_idx = 0
        for w in wordtks[d]:
            uni_rv = uni_rvs[uni_idx] #np.random.rand() * prob_sum
            uni_idx += 1
            rc_start = 0
            rc_stop  = K
            while rc_start < rc_stop - 1:
                rc_mid = (rc_start + rc_stop) // 2
                if cumprobs[rc_mid] <= uni_rv:
                    rc_start = rc_mid
                else:
                    rc_stop = rc_mid
            #while uni_rv > cumprobs[rc_start]:
            #    rc_start += 1
            zInit    = rc_start
            Ndk[d,zInit] += 1
            zd[tks_idx] = zInit
            tks_idx += 1
        z[d] = zd

    # Draw samples from the posterior on z_ids using Gibbs sampling

    # burn-in phase
    for sim in range(burn_in):
        for d in range(D):
            tks_idx = 0
            for w in wordtks[d]:
                zOld = z[d][tks_idx]
                Ndk[d,zOld] -= 1
                prob_sum = 0
                # Faster than using numpy elt product
                for k in range(K):
                    cumprobs[k] = prob_sum
                    prob_sum +=  (alpha + Ndk[d,k]) * expElogbeta[k,w]
                uni_rv = prob_sum * uni_rvs[uni_idx]
                uni_idx += 1
		# inline randcat function call
                rc_start = 0
                rc_stop  = K
                while rc_start < rc_stop - 1:
                    rc_mid = (rc_start + rc_stop) // 2
                    if cumprobs[rc_mid] <= uni_rv:
                        rc_start = rc_mid
                    else:
                        rc_stop = rc_mid
                zNew = rc_start
                z[d][tks_idx] = zNew
                tks_idx += 1
                Ndk[d,zNew] += 1

    # sampling phase
    for sim in range(num_sim):
        for d in range(D):
            tks_idx = 0
            for w in wordtks[d]:
                zOld = z[d][tks_idx]
                Ndk[d,zOld] -= 1
                prob_sum = 0
                # Faster than using numpy elt product
                for k in range(K):
                    cumprobs[k] = prob_sum
                    prob_sum +=  (alpha + Ndk[d,k]) * expElogbeta[k,w]
                uni_rv = prob_sum * uni_rvs[uni_idx]
                uni_idx += 1
		# inline randcat function call
                rc_start = 0
                rc_stop  = K
                while rc_start < rc_stop - 1:
                    rc_mid = (rc_start + rc_stop) // 2
                    if cumprobs[rc_mid] <= uni_rv:
                        rc_start = rc_mid
                    else:
                        rc_stop = rc_mid
                zNew = rc_start
                z[d][tks_idx] = zNew
                tks_idx += 1
                Ndk[d,zNew] += 1
                Ndk_mean[d,zNew] += update_unit
                Nkw_mean[zNew,w] += update_unit
