"""
Description:
------------
244-coefficient truncated Volterra series ALES model static test program

Notes:
------
run `mpiexec -n 1 python ales244_static_test.py -h` for help

Authors:
--------
Colin Towery, colin.towery@colorado.edu

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
https://github.com/teslacu/teslapy.git
https://github.com/teslacu/spectralLES.git
"""

from mpi4py import MPI
import numpy as np
import os
import sys
import time
from math import sqrt
import argparse
from spectralLES import spectralLES
from teslacu import mpiWriter
from teslacu.fft import rfft3, irfft3, shell_average
from teslacu.stats import psum
import Smith5Function
import Selective_limiting_k
import Selective_amp
import matplotlib as mpl
mpl.use('agg')
import pylab as plt
from subprocess import call
import json
from differentiation    import spec_diff
#-------------------------------------------------------------------------#
# JSON dump                                                               #
#-------------------------------------------------------------------------#
def dump_run_config_json(path, comm, pp, sp, solver, extra=None):
    """
    Dump run configuration (pp, sp, and key solver info) to a JSON file.
    Rank 0 only.
    """
    if comm.rank != 0:
        return

    def to_jsonable(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    pp_dict = {k: to_jsonable(v) for k, v in vars(pp).items()}
    sp_dict = {k: to_jsonable(v) for k, v in vars(sp).items()}

    solver_dict = {
        "nx": to_jsonable(solver.nx),
        "L": to_jsonable(solver.L),
        "dx": to_jsonable(solver.dx),
        "nu": to_jsonable(solver.nu),
        "epsilon": to_jsonable(solver.epsilon),

        "k_dealias": to_jsonable(getattr(solver, "k_dealias", None)),
        "k_les": to_jsonable(getattr(solver, "k_les", None)),
        "D_les": to_jsonable(getattr(solver, "D_les", None)),

        "kfLow": to_jsonable(getattr(solver, "kfLow", None)),
        "kfHigh": to_jsonable(getattr(solver, "kfHigh", None)),
        "k_test": to_jsonable(getattr(solver, "k_test", None)),
    }

    out = {
        "problem_params": pp_dict,
        "solver_params": sp_dict,
        "solver_info": solver_dict,
    }

    if extra:
        out["extra"] = {k: to_jsonable(v) for k, v in extra.items()}

    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)




#-------------------------------------------------------------------------#
# User packages                                                           #
#-------------------------------------------------------------------------#

comm = MPI.COMM_WORLD


def timeofday():
    return time.strftime("%H:%M:%S")


###############################################################################
# Extend the spectralLES class
###############################################################################
class ales244_solver(spectralLES):
    """
    Just adding extra memory and the ales244 SGS model. By using the
    spectralLES class as a super-class and defining a subclass for each
    SGS model we want to test, spectralLES doesn't get cluttered with
    an excess of models over time.
    """

    # Class Constructor -------------------------------------------------------
    def __init__(self, comm, N, L, nu, epsilon, Gtype, **kwargs):
        """
        Empty Docstring!
        """

        super().__init__(comm, N, L, nu, epsilon, Gtype, **kwargs)

        self.t_ds           = 0.0
        self.tau_hat        = np.empty((9, *self.nnk), dtype=complex)
        self.UU_hat         = np.empty_like(self.tau_hat)
        self.tau            = np.empty((9, *self.nnx))          # subgrid stress
        self.Sij            = np.empty((6, *self.nnx))          # subgrid stress
        self.k_test         = 15
        self.test_filter    = self.filter_kernel(self.k_test, Gtype)

    # Instance Methods --------------------------------------------------------
    def computeSource_ales244_SGS(self, **ignored):
        """
        sparse tensor indexing for ales244_solver UU_hat and tau_hat:
        m == 0 -> ij == 22
        m == 1 -> ij == 21
        m == 2 -> ij == 20
        m == 3 -> ij == 11
        m == 4 -> ij == 10
        m == 5 -> ij == 00

        """
        dim = self.nx[0].min()
        C_S5 = 1.0
        t_ds        = self.t_ds             # simulation time
        #-----------------------------------------------------------------#
        # Preallocating variables                                         #
        #-----------------------------------------------------------------#
        tau_hat     = self.tau_hat
        tau         = self.tau
        Sij         = self.Sij
        W_hat       = self.W_hat
        comm        = self.comm
        self.tau[:] = 0.0
        #-----------------------------------------------------------------#
        # U hat                                                           #
        #-----------------------------------------------------------------#
        W_hat[:] = self.les_filter*self.U_hat
        irfft3(self.comm, W_hat[0], self.W[0])
        irfft3(self.comm, W_hat[1], self.W[1])
        irfft3(self.comm, W_hat[2], self.W[2])
        #-----------------------------------------------------------------#
        # More preallocation                                              #
        #-----------------------------------------------------------------#
        t0          = time.time()
        Tij         = np.empty((6, *self.nnx), dtype=np.float64)
        Uhat        = np.empty((3, *self.nnx), dtype=np.float64)
        Uhat_hat    = np.empty((3, *self.nnk), dtype=complex)
        Uijhat      = np.empty((6, *self.nnx), dtype=np.float64)

        tau_S5      = np.zeros((9, *self.nnx), dtype=np.float64)
        tau_out     = np.zeros((6, *self.nnx), dtype=np.float64)

        St          = np.empty((6, *self.nnx), dtype=np.float64)
        Rt          = np.empty((3, *self.nnx), dtype=np.float64)
        Sh          = np.empty((6, *self.nnx), dtype=np.float64)
        Rh          = np.empty((3, *self.nnx), dtype=np.float64)

        Pij_S5      = np.empty((9, *self.nnx),  dtype = np.float64)
        P_S5        = np.empty([*self.nnx],     dtype = np.float64)

        S           = np.empty([*self.nnx],         dtype = np.float64)
        Sij_colin   = np.empty((3, 3, *self.nnx),   dtype = np.float64)
        S_colin     = np.empty([*self.nnx],         dtype = np.float64)

        Pij_ke      = np.empty((9, *self.nnx), dtype = np.float64)
        Cij_ke      = np.empty((9, *self.nnx), dtype = np.float64)

        P_ke        = np.empty((1, *self.nnx), dtype = np.float64)
        C_ke        = np.empty((1, *self.nnx), dtype = np.float64)

        #-----------------------------------------------------------------#
        # Filter Velocities to find LES and Test filtered fields          #
        #-----------------------------------------------------------------#
        Uhat_hat[:] = self.test_filter*self.W_hat
        irfft3(self.comm, Uhat_hat[0], Uhat[2])
        irfft3(self.comm, Uhat_hat[1], Uhat[1])
        irfft3(self.comm, Uhat_hat[2], Uhat[0])

        Uijhat[0] = irfft3(comm,self.test_filter*rfft3(comm,self.W[2]*self.W[2])).real
        Uijhat[1] = irfft3(comm,self.test_filter*rfft3(comm,self.W[2]*self.W[1])).real
        Uijhat[2] = irfft3(comm,self.test_filter*rfft3(comm,self.W[2]*self.W[0])).real
        Uijhat[3] = irfft3(comm,self.test_filter*rfft3(comm,self.W[1]*self.W[1])).real
        Uijhat[4] = irfft3(comm,self.test_filter*rfft3(comm,self.W[1]*self.W[0])).real
        Uijhat[5] = irfft3(comm,self.test_filter*rfft3(comm,self.W[0]*self.W[0])).real

        Tij[0] = Uijhat[0] - Uhat[0] * Uhat[0]
        Tij[1] = Uijhat[1] - Uhat[0] * Uhat[1]
        Tij[2] = Uijhat[2] - Uhat[0] * Uhat[2]
        Tij[3] = Uijhat[3] - Uhat[1] * Uhat[1]
        Tij[4] = Uijhat[4] - Uhat[1] * Uhat[2]
        Tij[5] = Uijhat[5] - Uhat[2] * Uhat[2]

        St[0]           = 0.5*irfft3(self.comm,1j*self.K[2]*W_hat[2]+1j*self.K[2]*W_hat[2])
        St[1]           = 0.5*irfft3(self.comm,1j*self.K[2]*W_hat[1]+1j*self.K[1]*W_hat[2])
        St[2]           = 0.5*irfft3(self.comm,1j*self.K[2]*W_hat[0]+1j*self.K[0]*W_hat[2])
        St[3]           = 0.5*irfft3(self.comm,1j*self.K[1]*W_hat[1]+1j*self.K[1]*W_hat[1])
        St[4]           = 0.5*irfft3(self.comm,1j*self.K[1]*W_hat[0]+1j*self.K[0]*W_hat[1])
        St[5]           = 0.5*irfft3(self.comm,1j*self.K[0]*W_hat[0]+1j*self.K[0]*W_hat[0])
        Sij[:]          = St[:]
        self.Sij[:]     = St[:]

        Rt[0] = 0.5*irfft3(self.comm,1j*self.K[1]*W_hat[2]-1j*self.K[2]*W_hat[1])
        Rt[1] = 0.5*irfft3(self.comm,1j*self.K[0]*W_hat[1]-1j*self.K[1]*W_hat[0])
        Rt[2] = 0.5*irfft3(self.comm,1j*self.K[2]*W_hat[0]-1j*self.K[0]*W_hat[2])

        Sh[0] = irfft3(comm,self.test_filter*rfft3(comm,St[0])).real
        Sh[1] = irfft3(comm,self.test_filter*rfft3(comm,St[1])).real
        Sh[2] = irfft3(comm,self.test_filter*rfft3(comm,St[2])).real
        Sh[3] = irfft3(comm,self.test_filter*rfft3(comm,St[3])).real
        Sh[4] = irfft3(comm,self.test_filter*rfft3(comm,St[4])).real
        Sh[5] = irfft3(comm,self.test_filter*rfft3(comm,St[5])).real

        Rh[0] = irfft3(comm,self.test_filter*rfft3(comm,Rt[0])).real
        Rh[1] = irfft3(comm,self.test_filter*rfft3(comm,Rt[1])).real
        Rh[2] = irfft3(comm,self.test_filter*rfft3(comm,Rt[2])).real

        # print('Running Autonomic Closure')
        # Define domain dimensions and autonomic closure parameters
        shift = 3
        gc_size = dim + shift + shift

        # Create arrays with required ghost cells
        Tijgc = np.empty((6, gc_size, gc_size, gc_size), dtype=np.float64)
        Stgc  = np.empty((6, gc_size, gc_size, gc_size), dtype=np.float64)
        Rtgc  = np.empty((3, gc_size, gc_size, gc_size), dtype=np.float64)
        Shgc  = np.empty((6, gc_size, gc_size, gc_size), dtype=np.float64)
        Rhgc  = np.empty((3, gc_size, gc_size, gc_size), dtype=np.float64)

        # Fill U into physical part of domain
        Tijgc[:, shift:dim + shift, shift:dim + shift, shift:dim + shift]  = Tij
        Stgc[ :, shift:dim + shift, shift:dim + shift, shift:dim + shift]  = St
        Shgc[ :, shift:dim + shift, shift:dim + shift, shift:dim + shift]  = Sh
        Rtgc[ :, shift:dim + shift, shift:dim + shift, shift:dim + shift]  = Rt
        Rhgc[ :, shift:dim + shift, shift:dim + shift, shift:dim + shift]  = Rh

        # Add U into ghost cells off the front and back xy plane
        Tijgc[:, :, :, 0:shift] = Tijgc[:, :, :, dim:dim + shift]
        Stgc[ :, :, :, 0:shift] = Stgc[ :, :, :, dim:dim + shift]
        Shgc[ :, :, :, 0:shift] = Shgc[ :, :, :, dim:dim + shift]
        Rtgc[ :, :, :, 0:shift] = Rtgc[ :, :, :, dim:dim + shift]
        Rhgc[ :, :, :, 0:shift] = Rhgc[ :, :, :, dim:dim + shift]
        Tijgc[:, :, :, dim + shift:gc_size] = Tijgc[:, :, :, shift:shift + shift]
        Stgc[ :, :, :, dim + shift:gc_size] = Stgc[ :, :, :, shift:shift + shift]
        Shgc[ :, :, :, dim + shift:gc_size] = Shgc[ :, :, :, shift:shift + shift]
        Rtgc[ :, :, :, dim + shift:gc_size] = Rtgc[ :, :, :, shift:shift + shift]
        Rhgc[ :, :, :, dim + shift:gc_size] = Rhgc[ :, :, :, shift:shift + shift]

        # Add U into ghost cells off the top and bottom xz plane
        Tijgc[:, :, 0:shift, :] = Tijgc[:, :, dim:dim + shift, :]
        Stgc[ :, :, 0:shift, :] = Stgc[ :, :, dim:dim + shift, :]
        Shgc[ :, :, 0:shift, :] = Shgc[ :, :, dim:dim + shift, :]
        Rtgc[ :, :, 0:shift, :] = Rtgc[ :, :, dim:dim + shift, :]
        Rhgc[ :, :, 0:shift, :] = Rhgc[ :, :, dim:dim + shift, :]
        Tijgc[:, :, dim + shift:gc_size, :] = Tijgc[:, :, shift:shift + shift, :]
        Stgc[ :, :, dim + shift:gc_size, :] = Stgc[ :, :, shift:shift + shift, :]
        Shgc[ :, :, dim + shift:gc_size, :] = Shgc[ :, :, shift:shift + shift, :]
        Rtgc[ :, :, dim + shift:gc_size, :] = Rtgc[ :, :, shift:shift + shift, :]
        Rhgc[ :, :, dim + shift:gc_size, :] = Rhgc[ :, :, shift:shift + shift, :]

        # Add U into ghost cells off the left and right yz plane
        Tijgc[:, 0:shift, :, :] = Tijgc[:, dim:dim + shift, :, :]
        Stgc[ :, 0:shift, :, :] = Stgc[ :, dim:dim + shift, :, :]
        Shgc[ :, 0:shift, :, :] = Shgc[ :, dim:dim + shift, :, :]
        Rtgc[ :, 0:shift, :, :] = Rtgc[ :, dim:dim + shift, :, :]
        Rhgc[ :, 0:shift, :, :] = Rhgc[ :, dim:dim + shift, :, :]
        Tijgc[:, dim + shift:gc_size, :, :] = Tijgc[:, shift:shift + shift, :, :]
        Stgc[ :, dim + shift:gc_size, :, :] = Stgc[ :, shift:shift + shift, :, :]
        Shgc[ :, dim + shift:gc_size, :, :] = Shgc[ :, shift:shift + shift, :, :]
        Rtgc[ :, dim + shift:gc_size, :, :] = Rtgc[ :, shift:shift + shift, :, :]
        Rhgc[ :, dim + shift:gc_size, :, :] = Rhgc[ :, shift:shift + shift, :, :]
        #-----------------------------------------------------------------#
        # TF5 method                                                      #
        #-----------------------------------------------------------------#
        tau_out[:]  = 0.0
        tau_out[:]  = Smith5Function.smith5(Stgc,Rtgc,Shgc,Rhgc,Tijgc,dim,shift,tau_out,1)
        tau_S5[0]   = tau_out[0]    # 11
        tau_S5[1]   = tau_out[1]    # 12
        tau_S5[2]   = tau_out[2]    # 13
        tau_S5[3]   = tau_out[1]    # 21
        tau_S5[4]   = tau_out[3]    # 22
        tau_S5[5]   = tau_out[4]    # 23
        tau_S5[6]   = tau_out[2]    # 31
        tau_S5[7]   = tau_out[4]    # 32
        tau_S5[8]   = tau_out[5]    # 33
        #-----------------------------------------------------------------#
        # Total subgrid stress                                            #
        #-----------------------------------------------------------------#
        tau[:]      = tau_S5[:]
        ##-----------------------------------------------------------------#
        ## Find transport terms                                            #
        ##-----------------------------------------------------------------#
        #[Pij_ke[:], P_ke[:]]    = self.P_KE(tau, Sij)
        #Pk_before               = np.mean(np.mean(np.mean(P_ke)))
        ##-----------------------------------------------------------------#
        ## Selective limiting                                              #
        ##-----------------------------------------------------------------#
        #CA                      = 1.5
        #tau[:]                  = Selective_amp.selective_amp(CA, Pij_ke, tau)
        [Pij_ke[:], P_ke[:]]    = self.P_KE(tau, Sij)
        #[Cij_ke[:], C_ke[:]]    = self.C_KE(tau)
        Pk_before               = np.mean(np.mean(np.mean(P_ke)))
        #-----------------------------------------------------------------#
        # Selective limiting                                              #
        #-----------------------------------------------------------------#
        CL          = 0.5
        tau[:]      = Selective_limiting_k.selective_limiting(\
                            CL, Pij_ke, P_ke[0], tau)
        P_ke[:]     = self.P_KE(tau, Sij)[1]
        Pk_after    = np.mean(np.mean(np.mean(P_ke)))
        Pk_max      = np.amax(P_ke)
        #-----------------------------------------------------------------#
        # Print statement                                                 #
        #-----------------------------------------------------------------#
        print('ke time = %8.3e\tPk_before = %8.3e\tPk_after = %8.3e\tPk_max = %8.3e'\
                    %(t_ds,Pk_before, Pk_after, Pk_max))
        #-----------------------------------------------------------------#
        # Global subgrid stress                                           #
        #-----------------------------------------------------------------#
        self.tau[:] = tau[:]
        #-----------------------------------------------------------------#
        # SGS transform                                                   #
        #-----------------------------------------------------------------#
        tau_hat[0] = rfft3(comm, self.tau[0])       # 11
        tau_hat[1] = rfft3(comm, self.tau[1])       # 12
        tau_hat[2] = rfft3(comm, self.tau[2])       # 31
        tau_hat[3] = rfft3(comm, self.tau[3])       # 21
        tau_hat[4] = rfft3(comm, self.tau[4])       # 22
        tau_hat[5] = rfft3(comm, self.tau[5])       # 23
        tau_hat[6] = rfft3(comm, self.tau[6])       # 31
        tau_hat[7] = rfft3(comm, self.tau[7])       # 32
        tau_hat[8] = rfft3(comm, self.tau[8])       # 33
        #-----------------------------------------------------------------#
        # Add to RHS                                                      #
        #-----------------------------------------------------------------#
        m = 0
        for i in range(2, -1, -1):
            for j in range(2, -1, -1):
                self.dU[i] -= 1j*self.K[j]*tau_hat[m]
                m   += 1

        t1 = time.time()

        return
    #--------------------------------------------------------------------------#
    # Subroutine for the time                                                  #
    #--------------------------------------------------------------------------#
    def get_time(self, time, **ignored):
        self.t_ds   = time

        return
    #--------------------------------------------------------------------------#
    # A term calculation                                                       #
    #--------------------------------------------------------------------------#
    def A_enstrophy_transport(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this script is to compute the A term in the enstrophy
            transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        Sij         = self.Sij
        omega       = self.omega
        #----------------------------------------------------------------------#
        # Calculating A term                                                   #
        #----------------------------------------------------------------------#
        A           =  omega[2]*Sij[0]*omega[2] + 2.0*omega[2]*Sij[1]*omega[1]\
                    + 2.0*omega[2]*Sij[2]*omega[0] + omega[1]*Sij[3]*omega[1]\
                    + 2.0*omega[1]*Sij[4]*omega[0] + omega[0]*Sij[5]*omega[0]

        return A
    #--------------------------------------------------------------------------#
    # B term calculation                                                       #
    #--------------------------------------------------------------------------#
    def B_enstrophy_transport(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the B term in the
            enstrophy transport equation.

            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        nu          = self.nu
        #----------------------------------------------------------------------#
        # Calculating enstrophy                                                #
        #----------------------------------------------------------------------#
        enst        = 0.5*(omega[0]**2.0 + omega[1]**2.0 + omega[2]**2.0)
        enst_hat    = rfft3(comm, enst)
        #----------------------------------------------------------------------#
        # Calculating B term                                                   #
        #----------------------------------------------------------------------#
        Ksq     = self.Ksq
        B_hat   = -nu*Ksq*enst_hat
        B       = irfft3(comm, B_hat).real

        return B
    #--------------------------------------------------------------------------#
    # D term calculation                                                       #
    #--------------------------------------------------------------------------#
    def D_enstrophy_transport(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the D term in the
            enstrophy transport equation.

            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        nu          = self.nu
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega_hat_2 = rfft3(self.comm, omega[2])
        omega_hat_1 = rfft3(self.comm, omega[1])
        omega_hat_0 = rfft3(self.comm, omega[0])
        #----------------------------------------------------------------------#
        # Calculating D term                                                   #
        #----------------------------------------------------------------------#
        D       = np.square(irfft3(self.comm, 1j*self.K[2]*omega_hat_2))
        D       += np.square(irfft3(self.comm, 1j*self.K[1]*omega_hat_2))
        D       += np.square(irfft3(self.comm, 1j*self.K[0]*omega_hat_2))
        D       += np.square(irfft3(self.comm, 1j*self.K[2]*omega_hat_1))
        D       += np.square(irfft3(self.comm, 1j*self.K[1]*omega_hat_1))
        D       += np.square(irfft3(self.comm, 1j*self.K[0]*omega_hat_1))
        D       += np.square(irfft3(self.comm, 1j*self.K[2]*omega_hat_0))
        D       += np.square(irfft3(self.comm, 1j*self.K[1]*omega_hat_0))
        D       += np.square(irfft3(self.comm, 1j*self.K[0]*omega_hat_0))
        D       *= -nu

        return D
    #--------------------------------------------------------------------------#
    # SGS transport term calculation                                           #
    #--------------------------------------------------------------------------#
    def C_enstrophy_transport(self, Tau, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the SGS transport
            term in the enstrophy transport equation.

            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.

        Author:
            Emilio Torres
        ====================================================================="""
        #-----------------------------------------------------------------#
        # Preallocating                                                   #
        #-----------------------------------------------------------------#
        omega   = np.empty([3, 64, 64, 64], dtype=np.float64)
        cij     = np.empty([9, 64, 64, 64], dtype=np.float64)
        C       = np.empty([1, 64, 64, 64], dtype=np.float64)
        #-----------------------------------------------------------------#
        # Velocity variables                                              #
        #-----------------------------------------------------------------#
        U_hat   = self.U_hat
        U       = self.U
        K       = self.K
        #-----------------------------------------------------------------#
        # Take curl of velocity and inverse transform to get vorticity    #
        #-----------------------------------------------------------------#
        irfft3(comm, 1j*(-K[1]*U_hat[2] + K[2]*U_hat[1]), omega[0])
        irfft3(comm, 1j*(-K[2]*U_hat[0] + K[0]*U_hat[2]), omega[1])
        irfft3(comm, 1j*(-K[0]*U_hat[1] + K[1]*U_hat[0]), omega[2])
        #-----------------------------------------------------------------#
        # C_11, C_12, C13                                                 #
        #-----------------------------------------------------------------#
        cij[0]  = -spec_diff(omega[1]*spec_diff(Tau[0], 0), 2) +\
                    spec_diff(omega[0]*spec_diff(Tau[0], 1), 2)     # C_11

        cij[1]  = -spec_diff(omega[1]*spec_diff(Tau[1], 0), 1) +\
                    spec_diff(omega[0]*spec_diff(Tau[1], 1), 1)     # C_12

        cij[2]  = -spec_diff(omega[1]*spec_diff(Tau[2], 0), 0) +\
                    spec_diff(omega[0]*spec_diff(Tau[2], 1), 0)     # C_13
        #-----------------------------------------------------------------#
        # C_21, C_22, C23                                                 #
        #-----------------------------------------------------------------#
        cij[3]  = -spec_diff(omega[0]*spec_diff(Tau[3], 2), 2) +\
                    spec_diff(omega[2]*spec_diff(Tau[3], 0), 2)     # C_21

        cij[4]  = -spec_diff(omega[0]*spec_diff(Tau[4], 2), 1) +\
                    spec_diff(omega[2]*spec_diff(Tau[4], 0), 1)     # C_22

        cij[5]  = -spec_diff(omega[0]*spec_diff(Tau[5], 2), 0) +\
                    spec_diff(omega[2]*spec_diff(Tau[5], 0), 0)     # C_23
        #-----------------------------------------------------------------#
        # C_31, C_32, C33                                                 #
        #-----------------------------------------------------------------#
        cij[6]  = -spec_diff(omega[2]*spec_diff(Tau[6], 1), 2) +\
                    spec_diff(omega[1]*spec_diff(Tau[6], 2), 2)     # C_31

        cij[7]  = -spec_diff(omega[2]*spec_diff(Tau[7], 1), 1) +\
                    spec_diff(omega[1]*spec_diff(Tau[7], 2), 1)     #C_32

        cij[8]  = -spec_diff(omega[2]*spec_diff(Tau[8], 1), 0) +\
                    spec_diff(omega[1]*spec_diff(Tau[8], 2), 0)     # C_33
        #-----------------------------------------------------------------#
        # Total C_Omega                                                   #
        #-----------------------------------------------------------------#
        C[0]    = cij[0] + cij[1] + cij[3] + cij[4] + cij[5] + cij[6] +\
                    cij[7] + cij[8]

        return (cij, C)
    #---------------------------------------------------------------------#
    # SGS production term calculation                                     #
    #---------------------------------------------------------------------#
    def P_enstrophy_transport(self, Tau, **ignored):
        """================================================================
        Purpose:
            The purpose of this subroutine is to calculate the SGS production
            term in the enstrophy transport equation.

            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.

        Author:
            Emilio Torres
        ================================================================"""
        #-----------------------------------------------------------------#
        # Preallocating                                                   #
        #-----------------------------------------------------------------#
        omega   = np.empty([3, 64, 64, 64], dtype=np.float64)
        P       = np.empty((1, *self.nnx),  dtype = np.float64)
        pij     = np.empty([9, 64, 64, 64], dtype=np.float64)
        #-----------------------------------------------------------------#
        # Velocity variables                                              #
        #-----------------------------------------------------------------#
        K       = self.K
        U_hat   = self.U_hat
        U       = self.U
        comm    = self.comm
        #-----------------------------------------------------------------#
        # Take curl of velocity and inverse transform to get vorticity    #
        #-----------------------------------------------------------------#
        irfft3(comm, 1j*(-K[1]*U_hat[2] + K[2]*U_hat[1]), omega[0])
        irfft3(comm, 1j*(-K[2]*U_hat[0] + K[0]*U_hat[2]), omega[1])
        irfft3(comm, 1j*(-K[0]*U_hat[1] + K[1]*U_hat[0]), omega[2])
        #-----------------------------------------------------------------#
        # P_11, P_12, P_13                                                #
        #-----------------------------------------------------------------#
        pij[0]  = spec_diff(Tau[0], 0)*spec_diff(omega[1], 2) -\
                        spec_diff(Tau[0], 1)*spec_diff(omega[0], 2)

        pij[1]  = spec_diff(Tau[1], 0)*spec_diff(omega[1], 1) -\
                        spec_diff(Tau[1], 1)*spec_diff(omega[0], 1)

        pij[2]  = spec_diff(Tau[2], 0)*spec_diff(omega[1], 0) -\
                        spec_diff(Tau[2], 1)*spec_diff(omega[0], 0)
        #-----------------------------------------------------------------#
        # P_21, P_22, P_23                                                #
        #-----------------------------------------------------------------#
        pij[3]  = spec_diff(Tau[3], 2)*spec_diff(omega[0], 2) -\
                        spec_diff(Tau[3], 0)*spec_diff(omega[2], 2)

        pij[4]  = spec_diff(Tau[4], 2)*spec_diff(omega[0], 1) -\
                        spec_diff(Tau[4], 0)*spec_diff(omega[2], 1)

        pij[5]  = spec_diff(Tau[5], 2)*spec_diff(omega[0], 0) -\
                        spec_diff(Tau[5], 0)*spec_diff(omega[2], 0)
        #-----------------------------------------------------------------#
        # P_31, P_32, P_33                                                #
        #-----------------------------------------------------------------#
        pij[6]  = spec_diff(Tau[6], 1)*spec_diff(omega[2], 2) -\
                        spec_diff(Tau[6], 2)*spec_diff(omega[1], 2)

        pij[7]  = spec_diff(Tau[7], 1)*spec_diff(omega[2], 1) -\
                        spec_diff(Tau[7], 2)*spec_diff(omega[1], 1)

        pij[8]  = spec_diff(Tau[8], 1)*spec_diff(omega[2], 0) -\
                        spec_diff(Tau[8], 2)*spec_diff(omega[1], 0)
        #-----------------------------------------------------------------#
        # Total C_Omega                                                   #
        #-----------------------------------------------------------------#
        P[0]    = pij[0] + pij[1] + pij[3] + pij[4] + pij[5] + pij[6] +\
                    pij[7] + pij[8]

        return (pij, P)
    #--------------------------------------------------------------------------#
    # Calculating the enstrophy                                                #
    #--------------------------------------------------------------------------#
    def enstrophy(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the enstrophy in
            order to compare it to the results from the extraction.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        #----------------------------------------------------------------------#
        # Calculating enstrophy                                                #
        #----------------------------------------------------------------------#
        enst        = 0.5*(omega[0]**2.0 + omega[1]**2.0 + omega[2]**2.0)

        return enst
    #--------------------------------------------------------------------------#
    # Calculating the kinetic energy                                           #
    #--------------------------------------------------------------------------#
    def kinetic_energy(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the kinetic energy
            in order to compare it to the results from the extraction.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        U   = self.U
        #----------------------------------------------------------------------#
        # Calculating enstrophy                                                #
        #----------------------------------------------------------------------#
        ke  = 0.5*(U[0]**2.0 + U[1]**2.0 + U[2]**2.0)

        return ke
    #--------------------------------------------------------------------------#
    # Calculating the A term in the  kinetic energy transport equation         #
    #--------------------------------------------------------------------------#
    def A_KE(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the A term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        press   = self.compute_pressure()[0]
        vel     = self.U
        Ke      = self.kinetic_energy()
        #----------------------------------------------------------------------#
        # Calculating A                                                        #
        #----------------------------------------------------------------------#
        A   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,press*vel[2])).real
        A   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,press*vel[1])).real
        A   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,press*vel[0])).real
        A   *= -1.0

        return A
    #---------------------------------------------------------------------#
    # Calculating the C term in the  kinetic energy transport equation    #
    #---------------------------------------------------------------------#
    def C_KE(self, Tau, **ignored):
        """================================================================
        Purpose:
            The purpose of this subroutine is to calculate the B term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ================================================================"""
        #-----------------------------------------------------------------#
        # Preallocating                                                   #
        #-----------------------------------------------------------------#
        cij     = np.empty([9, 64, 64, 64], dtype = np.float64)
        C       = np.empty([1, 64, 64, 64], dtype = np.float64)
        #-----------------------------------------------------------------#
        # Global variables                                                #
        #-----------------------------------------------------------------#
        U       = self.U
        #-----------------------------------------------------------------#
        # C_11-C_33                                                       #
        #-----------------------------------------------------------------#
        cij[0]  = -spec_diff(U[2]*Tau[0], 2)
        cij[1]  = -spec_diff(U[2]*Tau[1], 1)
        cij[2]  = -spec_diff(U[2]*Tau[2], 0)
        cij[3]  = -spec_diff(U[1]*Tau[3], 2)
        cij[4]  = -spec_diff(U[1]*Tau[4], 1)
        cij[5]  = -spec_diff(U[1]*Tau[5], 0)
        cij[6]  = -spec_diff(U[0]*Tau[6], 2)
        cij[7]  = -spec_diff(U[0]*Tau[7], 1)
        cij[8]  = -spec_diff(U[0]*Tau[8], 0)
        #-----------------------------------------------------------------#
        # C_k production                                                  #
        #-----------------------------------------------------------------#
        C[0]    = cij[0] + cij[1] + cij[2] + cij[3] + cij[4] + cij[5] +\
                    cij[6] + cij[7] + cij[8]

        return (cij, C)
    #--------------------------------------------------------------------------#
    # Calculating the C term in the kinetic energy transport equation          #
    #--------------------------------------------------------------------------#
    def B_KE(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the C term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        nu  = self.nu
        Sij = self.Sij
        vel = self.U
        Ke  = self.kinetic_energy()
        #----------------------------------------------------------------------#
        # Calculating B                                                        #
        #----------------------------------------------------------------------#
        B   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,Sij[0]*vel[2])).real
        B   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,Sij[1]*vel[2])).real
        B   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,Sij[2]*vel[2])).real
        B   += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,Sij[1]*vel[1])).real
        B   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,Sij[3]*vel[1])).real
        B   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,Sij[4]*vel[1])).real
        B   += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,Sij[2]*vel[0])).real
        B   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,Sij[4]*vel[0])).real
        B   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,Sij[5]*vel[0])).real
        B   *= 2.0*nu

        return B
    #--------------------------------------------------------------------------#
    # Calculating the D term in the kinetic energy transport equation          #
    #--------------------------------------------------------------------------#
    def D_KE(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the D term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        nu  = self.nu
        Sij = self.Sij
        #----------------------------------------------------------------------#
        # Calculating D                                                        #
        #----------------------------------------------------------------------#
        D   = Sij[0]*Sij[0]
        D   += 2.0*Sij[1]*Sij[1]
        D   += 2.0*Sij[2]*Sij[2]
        D   += Sij[3]*Sij[3]
        D   += 2.0*Sij[4]*Sij[4]
        D   += Sij[5]*Sij[5]
        D   *= -2.0*nu

        return D
    #---------------------------------------------------------------------#
    # Calculating the P term in the kinetic energy transport equation     #
    #---------------------------------------------------------------------#
    def P_KE(self, Tau, sij, **ignored):
        """================================================================
        Purpose:
            The purpose of this subroutine is to calculate the P term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ================================================================"""
        #-----------------------------------------------------------------#
        # Preallocate                                                     #
        #-----------------------------------------------------------------#
        pij     = np.empty([9, 64, 64, 64], dtype = np.float64)
        P       = np.empty([1, 64, 64, 64], dtype = np.float64)


        #print('Inside Subroutine')
        #print('$$$$$$$$$$')
        #print(Tau[1,12:18,32,48])
        #print('^^^^^^^^^^')
        #print(sij[1,12:18,32,48])


        #-----------------------------------------------------------------#
        # P_11-P_33                                                       #
        #-----------------------------------------------------------------#
        pij[0]  = Tau[0]*sij[0]
        pij[1]  = Tau[1]*sij[1]
        pij[2]  = Tau[2]*sij[2]
        pij[3]  = Tau[3]*sij[1]
        pij[4]  = Tau[4]*sij[3]
        pij[5]  = Tau[5]*sij[4]
        pij[6]  = Tau[6]*sij[2]
        pij[7]  = Tau[7]*sij[4]
        pij[8]  = Tau[8]*sij[5]
        #-----------------------------------------------------------------#
        # Subgrid production                                              #
        #-----------------------------------------------------------------#
        P[0]    = pij[0] + pij[1] + pij[3] + pij[4] + pij[5] + pij[6] +\
                    pij[7] + pij[8]

        return (pij, P)
###############################################################################
# Define the problem ("main" function)
###############################################################################
def ales244_static_les_test(pp=None, sp=None):
    """
    Arguments:
    ----------
    pp: (optional) program parameters, parsed by argument parser
        provided by this file
    sp: (optional) solver parameters, parsed by spectralLES.parser
    """

    if comm.rank == 0:
        print("\n----------------------------------------------------------")
        print("MPI-parallel Python spectralLES simulation of problem \n"
              "`Homogeneous Isotropic Turbulence' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))
        print("----------------------------------------------------------")

    # ------------------------------------------------------------------
    # Get the problem and solver parameters and assert compliance
    if pp is None:
        pp = hit_parser.parse_known_args()[0]

    if sp is None:
        sp = spectralLES.parser.parse_known_args()[0]

    if comm.rank == 0:
        print('\nProblem Parameters:\n-------------------')
        for k, v in vars(pp).items():
            print(k, v)
        print('\nSpectralLES Parameters:\n-----------------------')
        for k, v in vars(sp).items():
            print(k, v)
        print("\n----------------------------------------------------------\n")

    assert len(set(pp.N)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal mesh dimensions')
    N = pp.N[0]
    assert len(set(pp.L)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal domain dimensions')
    L = pp.L[0]

    if N % comm.size > 0:
        if comm.rank == 0:
            print('Error: job started with improper number of MPI tasks for '
                  'the size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Configure the LES solver
    solver = ales244_solver(comm, **vars(sp))
    #---------------------------------------------------------------------#
    # Dump config once so you can inspect it later                        #
    #---------------------------------------------------------------------#
    dump_run_config_json(
        path_name + "/python/run_config.json",
        comm,
        pp,
        sp,
        solver,
        extra={
            "ke_field_write": ke_field_write,
            "enst_field_write": enst_field_write,
            "enst_terms_write": enst_terms_write,
            "ke_terms_write": ke_terms_write,
            "vel_write": vel_write,
            "tau_write": tau_write,
            "vorticity_write": vorticity_write,
            "S_write": S_write,
            "path_name": path_name,
        }
    )

    solver.computeAD = solver.computeAD_vorticity_form
    Sources = [solver.computeSource_linear_forcing,
               solver.computeSource_ales244_SGS]

    kwargs = {'dvScale': None}

    U_hat   = solver.U_hat
    U       = solver.U
    omega   = solver.omega                              # physical vorticity
    Kmod    = np.floor(np.sqrt(solver.Ksq)).astype(int)
    tau     = solver.tau                                # physical stress
    Sij     = solver.Sij                                # physical strain rate

    # ------------------------------------------------------------------
    # form HIT initial conditions from either user-defined values or
    # physics-based relationships
    Urms = 1.083*(pp.epsilon*L)**(1./3.)             # empirical coefficient
    Einit= getattr(pp, 'Einit', None) or Urms**2   # == 2*KE_equilibrium
    kexp = getattr(pp, 'kexp', None) or -1./3.     # -> E(k) ~ k^(-2./3.)
    kpeak= getattr(pp, 'kpeak', None) or N//4      # ~ kmax/2

    # currently using a fixed random seed for testing
    solver.initialize_HIT_random_spectrum(Einit, kexp, kpeak, rseed=comm.rank)

    # ------------------------------------------------------------------
    # Configure a spatial field writer
    writer = mpiWriter(comm, odir=pp.odir, N=N)
    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # -------------------------------------------------------------------------
    # Setup the various time and IO counters
    tauK = sqrt(pp.nu/pp.epsilon)           # Kolmogorov time-scale
    taul = 0.11*sqrt(3)*L/Urms              # 0.11 is empirical coefficient

    if pp.tlimit == np.Inf:
        pp.tlimit = 200*taul

    dt_rst = getattr(pp, 'dt_rst', None) or taul
    dt_spec= getattr(pp, 'dt_spec', None) or 0.2*taul
    dt_drv = getattr(pp, 'dt_drv', None) or 0.25*tauK

    t_sim = t_rst = t_spec = t_drv = 0.0
    tstep = irst = ispec = 0
    tseries = []

    if comm.rank == 0:
        print('\ntau_ell = %.6e\ntau_K = %.6e\n' % (taul, tauK))

    # -------------------------------------------------------------------------
    # Run the simulation
    limiter = 0
    while t_sim < pp.tlimit+1.e-8:

        t2 = time.time()

        # -- Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_constant_nu(pp.cfl)
        t_test = t_sim + 0.5*dt

        # -- output/store a log every step if needed/wanted
        KE      = 0.5*comm.allreduce(psum(np.square(U)))/solver.Nx
        Omega   = 0.5*comm.allreduce(psum(np.square(omega)))/solver.Nx
        #-----------------------------------------------------------------#
        # Calculating enstrophy transport terms spectral tool             #
        #-----------------------------------------------------------------#
        #A_enst          = solver.A_enstrophy_transport()
        #B_enst          = solver.B_enstrophy_transport()
        #D_enst          = solver.D_enstrophy_transport()
        #C_enst          = solver.C_enstrophy_transport(tau)[1]
        #P_enst          = solver.P_enstrophy_transport(tau)[1]
        #enst            = solver.enstrophy()
        #P_Omega_bar     = comm.allreduce(psum(P_enst))/solver.Nx
        #-----------------------------------------------------------------#
        # Calculating kinetic energy transport terms                      #
        #-----------------------------------------------------------------#
        #A_ke            = solver.A_KE()
        #B_ke            = solver.B_KE()
        #D_ke            = solver.D_KE()
        #C_ke            = solver.C_KE(tau)[1]
        P_ke            = solver.P_KE(tau,Sij)[1]
        #ke              = solver.kinetic_energy()
        P_K_bar         = comm.allreduce(psum(P_ke))/solver.Nx
        tseries.append([tstep, t_sim, dt, KE, Omega, P_K_bar])
        # print('During Output:')
        # print('tau[:,15,24,32] = ',tau[:,15,24,32])
        # print('tau[:,32,32,32] = ',tau[:,32,32,32])
        # print(' ')

        # -- output KE and enstrophy spectra
        if t_test >= t_spec:
            #-------------------------------------------------------------#
            # Storing the time data                                       #
            #-------------------------------------------------------------#
            np.save(path_name + '/time/SimulationTime_%(a)3.3d' % {'a': ispec}, t_sim)
            #-------------------------------------------------------------#
            # Storing enstrophy and enstrophy transport terms (spectral)  #
            #-------------------------------------------------------------#
            if enst_terms_write is True:
                np.save(path_name + '/A-enst/A_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   A_enst)
                np.save(path_name + '/B-enst/B_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   B_enst)
                np.save(path_name + '/D-enst/D_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   D_enst)
                np.save(path_name + '/C-enst/C_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   C_enst)
                np.save(path_name + '/P-enst/P_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   P_enst)
            if enst_field_write is True:
                np.save(path_name + '/enst/enstrophy_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, enst)
            #-------------------------------------------------------------#
            # Storing kinetic energy and kinetic energy transport terms   #
            #-------------------------------------------------------------#
            if ke_terms_write is True:
                np.save(path_name + '/A-ke/A_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, A_ke)
                np.save(path_name + '/B-ke/B_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, B_ke)
                np.save(path_name + '/C-ke/C_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, C_ke)
                np.save(path_name + '/D-ke/D_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, D_ke)
                np.save(path_name + '/P-ke/P_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, P_ke)
            if ke_field_write is True:
                np.save(path_name + '/ke/ke_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, ke)
            #-------------------------------------------------------------#
            # Storing the velocity data                                   #
            #-------------------------------------------------------------#
            if vel_write is True:
                np.save(path_name + '/velocity1/Velocity1_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, U[2])
                np.save(path_name + '/velocity2/Velocity2_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, U[1])
                np.save(path_name + '/velocity3/Velocity3_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, U[0])
            #-------------------------------------------------------------#
            # Storing the subgrid stress data                             #
            #-------------------------------------------------------------#
            if tau_write is True:
                np.save(path_name + '/tau/tau11_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[0])
                np.save(path_name + '/tau/tau12_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[1])
                np.save(path_name + '/tau/tau13_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[2])
                np.save(path_name + '/tau/tau22_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[3])
                np.save(path_name + '/tau/tau23_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[4])
                np.save(path_name + '/tau/tau33_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[5])
            #-------------------------------------------------------------#
            # Storing the strain rates                                    #
            #-------------------------------------------------------------#
            if S_write is True:
                np.save(path_name + '/strain-rates/S11_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[0])
                np.save(path_name + '/strain-rates/S12_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[1])
                np.save(path_name + '/strain-rates/S13_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[2])
                np.save(path_name + '/strain-rates/S22_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[3])
                np.save(path_name + '/strain-rates/S23_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[4])
                np.save(path_name + '/strain-rates/S33_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[5])
            #-------------------------------------------------------------#
            # Storing the vorticity data                                  #
            #-------------------------------------------------------------#
            if vorticity_write is True:
                np.save(path_name + '/vorticity1/Omega1_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, omega[2])
                np.save(path_name + '/vorticity2/Omega2_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, omega[1])
                np.save(path_name + '/vorticity3/Omega3_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, omega[0])

            t_spec += dt_spec
            ispec += 1

        # -- output physical-space solution fields for restarting and analysis
        if t_test >= t_rst:
            # writer.write_scalar('%s-Velocity1_%3.3d.rst' %
            #                     (pp.pid, irst), U[0], np.float64)
            # writer.write_scalar('%s-Velocity2_%3.3d.rst' %
            #                     (pp.pid, irst), U[1], np.float64)
            # writer.write_scalar('%s-Velocity3_%3.3d.rst' %
            #                     (pp.pid, irst), U[2], np.float64)

            t_rst += dt_rst
            irst += 1

        # -- Update the forcing mean scaling
        if t_test >= t_drv:
            # call solver.computeSource_linear_forcing to compute dvScale only
            kwargs['dvScale'] = Sources[0](computeRHS=False)
            t_drv += dt_drv
        # -- integrate the solution forward in time
        solver.RK4_integrate(dt, *Sources, **kwargs)
        # Calculate Pressure term
        solver.compute_pressure()

        t3 = time.time()

        print('step Time: ',t3-t2)

        if comm.rank == 0:
            print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e Omega = %15.8e  P_k_bar = %15.8e"
                  % (tstep, t_sim, dt, KE, Omega, P_K_bar))

        # limiter = limiter + 1
        # if limiter == 5:
        #     sys.exit()

        t_sim   += dt
        solver.get_time(t_sim)
        tstep   += 1

        sys.stdout.flush()  # forces Python 3 to flush print statements

    # -------------------------------------------------------------------------
    # Finalize the simulation
    KE          = 0.5*comm.allreduce(psum(np.square(U)))/solver.Nx
    Omega       = 0.5*comm.allreduce(psum(np.square(omega)))/solver.Nx
    P_K_bar     = comm.allreduce(psum(P_ke))/solver.Nx
    tseries.append([tstep, t_sim, dt, KE, Omega, P_K_bar])

    if comm.rank == 0:
        #fname = '%s/%s-%3.3d_KE_tseries.txt' % (pp.adir, pp.pid, ispec)
        fname   = path_name + '/ke-omega-avg.txt'
        header  = 'Kinetic Energy Time series\n'
        header  += '%s %s %s %s %s %s'\
                        %('time step'.center(30),\
                            'simulation time'.center(30),\
                            'delta time'.center(30),\
                            'kinetic energy'.center(30),\
                            'enstrophy'.center(30),\
                            'P_K_bar'.center(30))
        np.savetxt(fname, tseries, fmt='%30.12e', header=header)

        print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e Omega = %15.8e P_k_bar = %15.8e"
              % (tstep, t_sim, dt, KE, Omega, P_K_bar))
        print("\n----------------------------------------------------------")
        print("MPI-parallel Python spectralLES simulation finished at {}."
              .format(timeofday()))
        print("----------------------------------------------------------")

    # -- output kinetic energy spectrum to file
    spect3d = np.sum(np.real(U_hat*np.conj(U_hat)), axis=0)
    spect3d[..., 0] *= 0.5
    spect1d = shell_average(comm, spect3d, Kmod)

    if comm.rank == 0:
        fh = open(path_name + '/KE-spectra.txt', 'w')
        metadata = Ek_fmt('u_i')
        fh.write('%s\n' % metadata)
        spect1d.tofile(fh, sep='\n', format='% .8e')
        fh.close()

    ## -- output physical-space solution fields for restarting and analysis
    #writer.write_scalar('%s-Velocity1_%3.3d.rst' %
    #                    (pp.pid, irst), U[0], np.float64)
    #writer.write_scalar('%s-Velocity2_%3.3d.rst' %
    #                    (pp.pid, irst), U[1], np.float64)
    #writer.write_scalar('%s-Velocity3_%3.3d.rst' %
    #                    (pp.pid, irst), U[2], np.float64)

    return


###############################################################################
# Add a parser for this problem
###############################################################################
hit_parser = argparse.ArgumentParser(prog='Homogeneous Isotropic Turbulence',
                                     parents=[spectralLES.parser])

hit_parser.description = ("A large eddy simulation model testing and analysis "
                          "script for homogeneous isotropic turbulence")
hit_parser.epilog = ('This program uses spectralLES, %s'
                     % spectralLES.parser.description)

config_group = hit_parser._action_groups[2]

config_group.add_argument('-p', '--pid', type=str, default='test',
                          help='problem prefix for analysis outputs')
config_group.add_argument('--dt_drv', type=float,
                          help='refresh-rate of forcing pattern')

time_group = hit_parser.add_argument_group('time integration arguments')

time_group.add_argument('--cfl', type=float, default=0.45, help='CFL number')
time_group.add_argument('-t', '--tlimit', type=float, default=np.inf,
                        help='solution time limit')
time_group.add_argument('-w', '--twall', type=float,
                        help='run wall-time limit (ignored for now!!!)')

init_group = hit_parser.add_argument_group('initial condition arguments')

init_group.add_argument('-i', '--init', '--initial-condition',
                        metavar='IC', default='GamieOstriker',
                        choices=['GamieOstriker', 'TaylorGreen'],
                        help='use specified initial condition')
init_group.add_argument('--kexp', type=float,
                        help=('Gamie-Ostriker power-law scaling of '
                              'initial velocity condition'))
init_group.add_argument('--kpeak', type=float,
                        help=('Gamie-Ostriker exponential-decay scaling of '
                              'initial velocity condition'))
init_group.add_argument('--Einit', type=float,
                        help='specify KE of initial velocity field')

rst_group = hit_parser.add_argument_group('simulation restart arguments')

rst_group.add_argument('-l', '--last', '--restart-from-last', dest='restart',
                       action='store_const', const=-1,
                       help='restart from last *.rst checkpoint in IDIR')
rst_group.add_argument('-r', '--rst', '--restart-from-num', type=int,
                       dest='restart', metavar='NUM',
                       help=('restart from specified checkpoint in IDIR, '
                             'negative numbers index backwards from last'))
rst_group.add_argument('--idir', type=str, default='./data/',
                       help='input directory for restarts')

io_group = hit_parser.add_argument_group('simulation output arguments')

io_group.add_argument('--odir', type=str, default='./data/',
                      help='output directory for simulation fields')
io_group.add_argument('--dt_rst', type=float,
                      help='time between restart checkpoints')
io_group.add_argument('--dt_bin', type=float,
                      help='time between single-precision outputs')

anlzr_group = hit_parser.add_argument_group('analysis output arguments')

anlzr_group.add_argument('--adir', type=str, default='./analysis/',
                         help='output directory for analysis products')
anlzr_group.add_argument('--dt_stat', type=float,
                         help='time between statistical analysis outputs')
anlzr_group.add_argument('--dt_spec', type=float,
                         help='time between isotropic power spectral density'
                              ' outputs')


###############################################################################
if __name__ == "__main__":
    #---------------------------------------------------------------------#
    # Main preamble                                                       #
    #---------------------------------------------------------------------#
    call(['clear'])
    #---------------------------------------------------------------------#
    # Writing variables                                                   #
    #---------------------------------------------------------------------#
    ke_field_write      = False
    enst_field_write    = False
    enst_terms_write    = False
    ke_terms_write      = False
    vel_write           = True
    tau_write           = False
    vorticity_write     = False
    S_write             = False
    #---------------------------------------------------------------------#
    # Path variables                                                      #
    #---------------------------------------------------------------------#
    path_name   = 'post-test-run'
    paths       = ['time', 'python', 'data', 'media']
    if os.path.exists(path_name) is False:
        os.mkdir(path_name)
    if enst_field_write is True:
        paths.append('enst')
    if enst_terms_write is True:
        paths.append('A-enst')
        paths.append('B-enst')
        paths.append('C-enst')
        paths.append('D-enst')
        paths.append('P-enst')
    if ke_field_write is True:
        paths.append('ke')
    if ke_terms_write is True:
        paths.append('A-ke')
        paths.append('B-ke')
        paths.append('C-ke')
        paths.append('D-ke')
        paths.append('P-ke')
    if vel_write is True:
        paths.append('velocity1')
        paths.append('velocity3')
        paths.append('velocity2')
    if tau_write is True:
        paths.append('tau')
    if S_write is True:
        paths.append('strain-rates')
    if vorticity_write is True:
        paths.append('vorticity1')
        paths.append('vorticity2')
        paths.append('vorticity3')
    print(paths)
    flag  = True
    for path in paths:
        data_path   = path_name + os.sep + path
        print(data_path)
        if os.path.exists(data_path) is False:
            os.mkdir(data_path)
            flag = False
    if flag is False:
        print('**** Building directories ****')
        sys.exit(24)
    #---------------------------------------------------------------------#
    # ALES simulation                                                     #
    #---------------------------------------------------------------------#
    ales244_static_les_test()
