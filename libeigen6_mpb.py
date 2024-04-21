import meep as mp
from meep import mpb
import numpy as np


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

    freqs111 = ms.all_freqs

    # Create an MPBData instance to transform the efields
    # md = mpb.MPBData(rectify=True, periods=1)
    # converted_eps = md.convert(eps).T
    converted_eps = eps.T
    converted = []
    for f in fields:
        # 三个点这个语法就是前面的若干项都省略不写
        f = f[..., 0, 2]
        # f.shape: [nx, ny, nz, 3]，前面3个是网格，第4个参数从0到2是表示Ex Ey Ez
        converted.append(f)

    if flag:
        import matplotlib.pyplot as plt
        n_row = 1
        n_col = 2
        fig, _axs = plt.subplots(nrows=n_row, ncols=n_col)
        fig.subplots_adjust(hspace=0.3)
        axs = _axs.flatten()

        # for i, f in enumerate(converted):
        for i in range(N_max):
            f = converted[i]
            #axs[i].contour(converted_eps, cmap='binary')
            im = axs[i].imshow(np.real(f), interpolation='spline36', alpha=0.9, origin='lower', cmap='jet')
            fig.colorbar(im, ax=axs[i])
            plt.show()
            #axs[-1].set_axis_off()  # remove last plot
        

    return freqs111.flatten()*2*np.pi/Px, np.reshape(np.array(converted), (N_max, -1)).T, np.array(converted_eps).flatten()
    #返回的3个值分别是频率，场，和eps


omega, field_temp, eps_arr = eigs_PCs(1, 1,  0,  0, 2, 1, 1)