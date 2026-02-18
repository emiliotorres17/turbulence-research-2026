subroutine selective_limiting(CL, Pij_k, P_k, tauij)
    !---------------------------------------------------------------------!
    ! Preamble                                                            !
    !---------------------------------------------------------------------!
    implicit none
    double precision, intent(in)                                        :: CL
    double precision, intent(in),       dimension(1:9,1:64,1:64,1:64)   :: Pij_k
    double precision, intent(in),       dimension(1:64,1:64,1:64)       :: P_k
    double precision, intent(inout),    dimension(1:9,1:64,1:64,1:64)   :: tauij
!f2py intent(in)        :: CL
!f2py intent(in)        :: Pij_k
!f2py intent(in)        :: P_k
!f2py intent(in,out)    :: tauij
    integer  :: i, j, k, ii 
    !=====================================================================!
    ! Selective limiting                                                  !
    !=====================================================================!
    do k = 1, 64
        do j = 1, 64 
            do i = 1, 64
                !=========================================================!
                ! KE checks                                               !
                !=========================================================!
                !---------------------------------------------------------!
                ! Case (g)                                                !
                !---------------------------------------------------------!
                if (P_k(i,j,k) > 0.0d0 ) then
                    do ii = 1, 9
                        if (Pij_k(ii,i,j,k) > 0.0d0 ) then
                            tauij(ii,i,j,k) = (1.0d0-CL)*tauij(ii,i,j,k)
                        end if
                    end do
                end if 
            end do
        end do
    end do 
end subroutine selective_limiting
