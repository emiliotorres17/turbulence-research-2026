#!/usr/bin/env bash
f2py3 -c -m Smith5Function Smith5_9.f90 -llapack -lblas
f2py3 -c -m Selective_limiting_k selective-limiting-component-2.f90 -llapack -lblas
f2py3 -c -m Selective_amp selective-amplification.f90 -llapack -lblas

