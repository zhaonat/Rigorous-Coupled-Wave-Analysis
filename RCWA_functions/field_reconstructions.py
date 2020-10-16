import numpy as np

'''
    field reconstructions for 2D RCWA code
    generously written by Beicheng Lou of the Shanhui Fan group
'''

def get_field_ref(rx,ry,rz,Kx,Ky,Kz,k0,field_size,field_pts,dz):
    '''
        field on the reflection side of the domain
        rx, ry, rz: reflection side field coefficients
        field_size:  size of the unit cell
        field_pts: number of samples for plotting the field
    '''
    print(rx.shape,Kx.shape)
    NM = Kx.shape[0]
    Ex=np.zeros([field_pts,field_pts],dtype=np.complex)
    Ey=np.zeros([field_pts,field_pts],dtype=np.complex)
    Ez=np.zeros([field_pts,field_pts],dtype=np.complex)
    xxx=np.linspace(0,field_size,field_pts)
    yyy=np.linspace(0,field_size,field_pts)
    xx,yy=np.meshgrid(xxx,yyy)
    for nm in range(NM):
        kxmn=Kx[nm,nm]*k0
        kymn=Ky[nm,nm]*k0
        kzmn=Kz[nm,nm]*k0
        Ex+=rx[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy-kzmn*dz))
        Ey+=ry[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy-kzmn*dz))
        Ez+=rz[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy-kzmn*dz))
    return Ex,Ey,Ez
    
def get_field_trans(tx,ty,tz,Kx,Ky,Kz,k0,field_size,field_pts,dz):
    NM = Kx.shape[0]
    Ex=np.zeros([field_pts,field_pts],dtype=np.complex)
    Ey=np.zeros([field_pts,field_pts],dtype=np.complex)
    Ez=np.zeros([field_pts,field_pts],dtype=np.complex)
    xxx=np.linspace(0,field_size,field_pts)
    yyy=np.linspace(0,field_size,field_pts)
    xx,yy=np.meshgrid(xxx,yyy)
    for nm in range(NM):
        kxmn=Kx[nm,nm]*k0
        kymn=Ky[nm,nm]*k0
        kzmn=Kz[nm,nm]*k0
        Ex+=tx[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy+kzmn*dz))
        Ey+=ty[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy+kzmn*dz))
        Ez+=tz[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy+kzmn*dz))
    return Ex,Ey,Ez
    
def get_field_phc(tx,ty,tz,Kx,Ky,Kz,k0,field_size,field_pts,dz):
    ## dz won't be used, since sx,sy are already z-dependent
    '''
        tx, ty, tz: solved coefficients of the fourier components
    '''
    NM = Kx.shape[0]
    Ex=np.zeros([field_pts,field_pts],dtype=np.complex)
    Ey=np.zeros([field_pts,field_pts],dtype=np.complex)
    Ez=np.zeros([field_pts,field_pts],dtype=np.complex)
    xxx=np.linspace(0,field_size,field_pts)
    yyy=np.linspace(0,field_size,field_pts)
    xx,yy=np.meshgrid(xxx,yyy)
    for nm in range(NM):
        kxmn=Kx[nm,nm]*k0
        kymn=Ky[nm,nm]*k0
        kzmn=Kz[nm,nm]*k0
        Ex+=tx[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy))
        Ey+=ty[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy))
        Ez+=tz[nm,0]*np.exp(-1j*(kxmn*xx+kymn*yy))
    return Ex,Ey,Ez
