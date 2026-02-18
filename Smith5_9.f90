subroutine Smith5(St,Rt,Sh,Rh,Tijgc,dim,shift,tauij,start)
    implicit none

    integer, intent(in) :: dim, shift, start
    double precision, intent(in), dimension(1:6,1-shift:dim+shift,1-shift:dim+shift,1-shift:dim+shift) &
        :: Tijgc
    double precision, intent(in), dimension(1:6,1-shift:dim+shift,1-shift:dim+shift,1-shift:dim+shift) &
        :: St, Sh
    double precision, intent(in), dimension(1:3,1-shift:dim+shift,1-shift:dim+shift,1-shift:dim+shift) &
        :: Rt, Rh
    double precision, intent(inout), dimension(1:6,start:dim,start:dim,start:dim) :: tauij
!f2py intent(in) :: Tijgc
!f2py intent(in) :: St, Sh
!f2py intent(in) :: Rt, Rh
!f2py intent(in,out) :: tauij

    integer :: x, y, z, i, j, k, bbi, bbj, bbk, id, INFO, LWORK = 15616, n, ii, jj, terms
    double precision, dimension(1:6,1:5) :: Vtilde
    double precision :: hscale
    double precision, dimension(1:54,1:5) :: Vhat
    double precision, dimension(1:54) :: Tij
    double precision, dimension(1:5) :: hij = 0.0d0, VTT
    double precision, dimension(1:5,1:5) :: A = 1.0d0, VTV, IM = 0.0d0

    double precision, dimension(1:5) :: IPIV = 0.0d0
    double precision, dimension(1:15616) :: WORK
    double precision, dimension(1:3,1:3) :: M, W

    !print*,'Started on subroutine'

    terms = 5

    hscale = 2.0d0/3.0d0

    do id = 1,terms; IM(id,id) = 0.001d0; end do

    ! Loop over each grid point and calculate tau_ij there
    ! do k = 23,23; !print*,'k = ',k ! Use this one to print z-midplane images
    do k = 1,dim; !print*,'k = ',k ! Use this one to print z-midplane images
        do j = 1,dim
            do i = 1,dim
                ! Calculate V-tilde at the bounding box center
                Vtilde(1, 1) = 1.0d0
                Vtilde(2, 1) = 0.0d0
                Vtilde(3, 1) = 0.0d0
                Vtilde(4, 1) = 1.0d0
                Vtilde(5, 1) = 0.0d0
                Vtilde(6, 1) = 1.0d0

                ! Building M and W
                M(1,1) = St(1,i,j,k);    M(1,2) = St(2,i,j,k);    M(1,3) = St(3,i,j,k);
                M(2,1) = St(2,i,j,k);    M(2,2) = St(4,i,j,k);    M(2,3) = St(5,i,j,k);
                M(3,1) = St(3,i,j,k);    M(3,2) = St(5,i,j,k);    M(3,3) = St(6,i,j,k);
                
                W(1,1) = 0.0d0;          W(1,2) = Rt(1,i,j,k);    W(1,3) = -Rt(3,i,j,k);
                W(2,1) = -Rt(1,i,j,k);   W(2,2) = 0.0d0;          W(2,3) = Rt(2,i,j,k);
                W(3,1) = Rt(3,i,j,k);    W(3,2) = -Rt(2,i,j,k);   W(3,3) = 0.0d0;

                Vtilde(:,2) = tr(M) ! S
                Vtilde(:,3) = tr(tm(M,M)) ! S^2
                Vtilde(:,4) = tr(tm(W,W)) ! R^2
                Vtilde(:,5) = tr(tm(M,W) - tm(W,M)) ! SR-RS

                ! Loop over every grid point in bounding box
                n = 0
                do bbk = -1,1,2
                    do bbj = -1,1,2
                        do bbi = -1,1, 2
                            ! Coordinate of each point in bounding box
                            x = i + bbi*2
                            y = j + bbj*2
                            z = k + bbk*2

                            ! Calculate Tij for each point in bounding box
                            Tij(1+n*6) = Tijgc(1,x,y,z) 
                            Tij(2+n*6) = Tijgc(2,x,y,z) 
                            Tij(3+n*6) = Tijgc(3,x,y,z) 
                            Tij(4+n*6) = Tijgc(4,x,y,z) 
                            Tij(5+n*6) = Tijgc(5,x,y,z) 
                            Tij(6+n*6) = Tijgc(6,x,y,z) 

                            Vhat(1+n*6, 1) = 1.0d0
                            Vhat(2+n*6, 1) = 0.0d0
                            Vhat(3+n*6, 1) = 0.0d0
                            Vhat(4+n*6, 1) = 1.0d0
                            Vhat(5+n*6, 1) = 0.0d0
                            Vhat(6+n*6, 1) = 1.0d0

                            ! Building M and W
                            M(1,1) = Sh(1,x,y,z);    M(1,2) = Sh(2,x,y,z);    M(1,3) = Sh(3,x,y,z);
                            M(2,1) = Sh(2,x,y,z);    M(2,2) = Sh(4,x,y,z);    M(2,3) = Sh(5,x,y,z);
                            M(3,1) = Sh(3,x,y,z);    M(3,2) = Sh(5,x,y,z);    M(3,3) = Sh(6,x,y,z);
                            
                            W(1,1) = 0.0d0;          W(1,2) = Rh(1,x,y,z);    W(1,3) = -Rh(3,x,y,z);
                            W(2,1) = -Rh(1,x,y,z);   W(2,2) = 0.0d0;          W(2,3) = Rh(2,x,y,z);
                            W(3,1) = Rh(3,x,y,z);    W(3,2) = -Rh(2,x,y,z);   W(3,3) = 0.0d0;
                            
                            Vhat((/1,2,3,4,5,6/)+n*6,2) = tr(M(:,:)) ! S
                            Vhat((/1,2,3,4,5,6/)+n*6,3) = tr(tm(M(:,:),M(:,:))) ! S^2
                            Vhat((/1,2,3,4,5,6/)+n*6,4) = tr(tm(W(:,:),W(:,:))) ! R^2
                            Vhat((/1,2,3,4,5,6/)+n*6,5) = tr(tm(M(:,:),W(:,:)) - tm(W(:,:),M(:,:))) ! SR-RS

                            n = n + 1
                        end do
                    end do
                end do

                ! Coordinate of each point in bounding box
                x = i
                y = j 
                z = k 

                ! Calculate Tij for each point in bounding box
                Tij(1+n*6) = Tijgc(1,x,y,z) 
                Tij(2+n*6) = Tijgc(2,x,y,z) 
                Tij(3+n*6) = Tijgc(3,x,y,z) 
                Tij(4+n*6) = Tijgc(4,x,y,z) 
                Tij(5+n*6) = Tijgc(5,x,y,z) 
                Tij(6+n*6) = Tijgc(6,x,y,z) 

                Vhat(1+n*6, 1) = 1.0d0
                Vhat(2+n*6, 1) = 0.0d0
                Vhat(3+n*6, 1) = 0.0d0
                Vhat(4+n*6, 1) = 1.0d0
                Vhat(5+n*6, 1) = 0.0d0
                Vhat(6+n*6, 1) = 1.0d0

                ! Building M and W
                M(1,1) = Sh(1,x,y,z);    M(1,2) = Sh(2,x,y,z);    M(1,3) = Sh(3,x,y,z);
                M(2,1) = Sh(2,x,y,z);    M(2,2) = Sh(4,x,y,z);    M(2,3) = Sh(5,x,y,z);
                M(3,1) = Sh(3,x,y,z);    M(3,2) = Sh(5,x,y,z);    M(3,3) = Sh(6,x,y,z);
                
                W(1,1) = 0.0d0;          W(1,2) = Rh(1,x,y,z);    W(1,3) = -Rh(3,x,y,z);
                W(2,1) = -Rh(1,x,y,z);   W(2,2) = 0.0d0;          W(2,3) = Rh(2,x,y,z);
                W(3,1) = Rh(3,x,y,z);    W(3,2) = -Rh(2,x,y,z);   W(3,3) = 0.0d0;
                
                Vhat((/1,2,3,4,5,6/)+n*6,2) = tr(M(:,:)) ! S
                Vhat((/1,2,3,4,5,6/)+n*6,3) = tr(tm(M(:,:),M(:,:))) ! S^2
                Vhat((/1,2,3,4,5,6/)+n*6,4) = tr(tm(W(:,:),W(:,:))) ! R^2
                Vhat((/1,2,3,4,5,6/)+n*6,5) = tr(tm(M(:,:),W(:,:)) - tm(W(:,:),M(:,:))) ! SR-RS


                call DGEMM('t','n',terms,terms,54,1.0d0,Vhat,54,Vhat,54,0.0d0,VTV,terms)
                call DGEMM('t','n',terms,1,54,1.0d0,Vhat,54,Tij,54,0.0d0,VTT,terms)

                A = VTV+IM
                call DSYSV('L',terms,1,A,terms,IPIV,VTT,terms,WORK,LWORK,INFO)
                hij = VTT

                hij(1)  = (0.5d0)**(hscale*(0.0d0+1.0d0))*hij(1)
                hij(2)  = (0.5d0)**(hscale*(1.0d0+1.0d0))*hij(2)
                hij(3)  = (0.5d0)**(hscale*(2.0d0+1.0d0))*hij(3)
                hij(4)  = (0.5d0)**(hscale*(2.0d0+1.0d0))*hij(4)
                hij(5)  = (0.5d0)**(hscale*(2.0d0+1.0d0))*hij(5)

                do jj = 1,6
                    do ii = 1,terms
                        tauij(jj,i,j,k) = tauij(jj,i,j,k) + Vtilde(jj,ii)*hij(ii)
                    end do
                end do
            
            end do
        end do
    end do

    !print *, 'Finished with autonomic closure function'

    contains

    function tm(A,B)
        implicit none
        double precision, intent(in), dimension(1:3,1:3) :: A, B
        double precision, dimension(1:3,1:3) :: tm
        tm(1,1) = A(1,1)*B(1,1) + A(1,2)*B(2,1) + A(1,3)*B(3,1)
        tm(1,2) = A(1,1)*B(1,2) + A(1,2)*B(2,2) + A(1,3)*B(3,2)
        tm(1,3) = A(1,1)*B(1,3) + A(1,2)*B(2,3) + A(1,3)*B(3,3)
        tm(2,1) = A(2,1)*B(1,1) + A(2,2)*B(2,1) + A(2,3)*B(3,1)
        tm(2,2) = A(2,1)*B(1,2) + A(2,2)*B(2,2) + A(2,3)*B(3,2)
        tm(2,3) = A(2,1)*B(1,3) + A(2,2)*B(2,3) + A(2,3)*B(3,3)
        tm(3,1) = A(3,1)*B(1,1) + A(3,2)*B(2,1) + A(3,3)*B(3,1)
        tm(3,2) = A(3,1)*B(1,2) + A(3,2)*B(2,2) + A(3,3)*B(3,2)
        tm(3,3) = A(3,1)*B(1,3) + A(3,2)*B(2,3) + A(3,3)*B(3,3)
    end function tm

    function tr(A)
        implicit none
        double precision, intent(in), dimension(1:3,1:3) :: A
        double precision, dimension(1:6) :: tr
        tr(1) = A(1,1)
        tr(2) = A(1,2)
        tr(3) = A(1,3)
        tr(4) = A(2,2)
        tr(5) = A(2,3)
        tr(6) = A(3,3)
    end function tr

end subroutine Smith5
