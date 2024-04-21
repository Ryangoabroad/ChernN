import numpy as np
import meep as mp
from meep import mpb
import matplotlib.pyplot as plt
import time as time

from numpy.core.fromnumeric import shape
# from libeigen6 import eigs_PCs


def eigs_PCs(Px, Py, kx, ky, N_max, flag, Polarflag):
    kx_ = kx/(2*np.pi)
    ky_ = ky/(2*np.pi)
    res = 32
    r1 = 181 / 385 / 2
    r2 = 81 / 385 / 2
    geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                  basis1=mp.Vector3(1, 0),
                                  basis2=mp.Vector3(0.5, np.sqrt(3) / 2)
                                  )
    k_points = [mp.cartesian_to_reciprocal(mp.Vector3(kx_, ky_), geometry_lattice)]
    geometry = [mp.Cylinder(r1, center=mp.Vector3(2 / 3, 2 / 3), material=mp.Medium(index=1)),
                mp.Cylinder(r2, center=mp.Vector3(1 / 3, 1 / 3), material=mp.Medium(index=1))]
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=geometry_lattice,
        default_material=mp.Medium(index=3.47),
        k_points=k_points,
        resolution=res,
        num_bands=N_max
    )

    fields = []

    def get_efields(ms, band):
        fields.append(ms.get_efield(band, bloch_phase=False))

    def get_hfields(ms, band):
        fields.append(ms.get_hfield(band, bloch_phase=False))

    if Polarflag:
        ms.run_te(mpb.output_at_kpoint(k_points[0], mpb.fix_hfield_phase, get_hfields))
        eps = ms.get_mu()
    else:
        ms.run_tm(mpb.output_at_kpoint(k_points[0], mpb.fix_efield_phase, get_efields))
        eps = ms.get_epsilon()

    freqss = ms.all_freqs

    converted_eps = eps.T
    converted = []
    for f in fields:
        # 三个点这个语法就是前面的若干项都省略不写
        f = f[..., 0, 2]
        # f.shape: [nx, ny, nz, 3]，前面3个是网格，第4个参数从0到2是表示Ex Ey Ez
        converted.append(f)
   

    return freqss.flatten()*2*np.pi/Px, converted, np.reshape(np.array(converted), (N_max, -1)).T, np.array(converted_eps).flatten()
    #返回的3个值分别是频率，场，和eps






    

#plot the K


do_chern = True  # do Chern Number Calc
do_band = False  # do Band Structure Calc
Pxg = 1  # lattice constant along x
Pyg = 1  # lattice constant along y
N_sam = 10  # Brillouin sampling points ***
N_max = 2  # maximum band number ***
np.set_printoptions(precision=3, suppress=True)

Polarflag = 1  # 极化方式，0，TM极化；1，TE极化

shifted = 0.0
Nkxg = 200
Nkyg = 1
Pkxg = (4 * np.pi ) / Pxg / np.sqrt(3)
Pkyg = (4 * np.pi ) / Pyg / np.sqrt(3)
deltakxg = Pkxg / Nkxg
deltakyg = Pkyg / Nkyg
NK = Nkxg * Nkyg
kcx = np.zeros((NK, 4))
kcy = np.zeros((NK, 4))
# initialized
fieldTot = np.zeros(N_max)
ChernNumber = np.zeros(N_max)
fieldtemp = np.zeros(N_max, dtype=complex)
N_Lband = 1
N_Hband = 2
fieldTotComposite = 0
ChernNumberComposite = 0
fieldtempComposite = 0

if do_chern:
    time_s = time.time()
    fields = []
    fields_cur = np.zeros((Nkxg, N_max))
    for m in range(Nkxg+1):
        for n in range(Nkyg+1):
            print('Calculating', m * (Nkyg+1) + n + 1, 'of', (Nkxg+1) * (Nkyg+1), '...')
            omega, f, field_temp, eps_arr = eigs_PCs(Pxg, Pyg,
                                                  #(m+0.5-Nkxg/2) * deltakxg * np.sqrt(3), (n-0.5) * deltakxg,  #扫描顶点边，从-pi到+pi，重现董建文论文
                                                  (m - Nkxg / 2 + 0.5) * deltakxg * np.sqrt(3) / 2, # 扫描平行四边形Kx
                                                  #(n - Nkyg / 2 + 0.5) * deltakyg - (m - Nkxg / 2 + 0.5) * deltakxg / 2, # 扫描Ky，要键入kx的1/2
                                                  (n + 0.5) * deltakyg / 2 - (m - Nkxg / 2 + 0.5) * deltakxg / 2, 
                                                  N_max, 0, Polarflag)
            fields.append(field_temp)

    print(shape(f))

    for m in range(Nkxg):
        for n in range(Nkyg):
            field1 = fields[m*(Nkyg+1)+n] #提取平行四边形边界的场
            field4 = fields[m*(Nkyg+1)+n+1]
            field3 = fields[(m+1)*(Nkyg+1)+n+1]
            field2 = fields[(m+1)*(Nkyg+1)+n]

            for mm in range(N_max): #对每一个能带计算陈数
                temp = field1[:, mm].conj().T * eps_arr @ field2[:, mm] #一行乘以一列变成一个数
                fieldtemp[mm] = temp / np.abs(temp)
                temp = field2[:, mm].conj().T * eps_arr @ field3[:, mm]
                fieldtemp[mm] *= temp / np.abs(temp) #这里是把前面的值再乘新值
                temp = field3[:, mm].conj().T * eps_arr @ field4[:, mm]
                fieldtemp[mm] *= temp / np.abs(temp)
                temp = field4[:, mm].conj().T * eps_arr @ field1[:, mm]
                fieldtemp[mm] *= temp / np.abs(temp)
                fields_cur[m, mm] = -np.imag(np.log(fieldtemp[mm]))/(deltakxg**2*np.sqrt(3))
                fieldTot[mm] = fieldTot[mm] + np.imag(np.log(fieldtemp[mm]))

            #print('total', fieldTot)


# 下面是计算简并点的陈数
            fieldt1 = (field1[:, N_Lband-1:N_Hband].conj().T * np.tile(eps_arr, (N_Hband - N_Lband + 1, 1)) @ field2[:, N_Lband-1:N_Hband]) @ \
                      (field2[:, N_Lband-1:N_Hband].conj().T * np.tile(eps_arr, (N_Hband - N_Lband + 1, 1)) @ field3[:, N_Lband-1:N_Hband]) @ \
                      (field3[:, N_Lband-1:N_Hband].conj().T * np.tile(eps_arr, (N_Hband - N_Lband + 1, 1)) @ field4[:, N_Lband-1:N_Hband]) @ \
                      (field4[:, N_Lband-1:N_Hband].conj().T * np.tile(eps_arr, (N_Hband - N_Lband + 1, 1)) @ field1[:, N_Lband-1:N_Hband])
            fieldtempComposite = np.linalg.det(fieldt1)
            fieldtempComposite = fieldtempComposite / np.abs(fieldtempComposite)
            fieldTotComposite = fieldTotComposite + np.imag(np.log(fieldtempComposite))

    #plot the eps



    fig, axs = plt.subplots(1,2)
    
    im = axs[0].imshow(np.real(f[0]), interpolation='spline36', alpha=0.9, origin='lower', cmap='jet')
    fig.colorbar(im, ax=axs)
    axs[0].set_axis_off()  # 
    

    axs[1].plot(fields_cur[:, 0]*(2*np.pi)**2) #fields_cur是贝里曲率
    np.savetxt('berry_cur.csv', fields_cur, delimiter=',', fmt='%f')
    plt.show()

    print('Elapsed:', time.time() - time_s)
    ChernNumber = fieldTot / 2 / np.pi
    ChernNumberComposite = fieldTotComposite / 2 / np.pi
    print('Chern Number', ChernNumber)


