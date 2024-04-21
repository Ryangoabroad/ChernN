import numpy as np
#import scipy.io
# import scipy.sparse as sparse
# import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import time as time
# from libeigen6 import eigs_PCs
from libeigen6_mpb import eigs_PCs


do_chern = True  # do Chern Number Calc
do_band = False  # do Band Structure Calc
Pxg = 1  # lattice constant along x
Pyg = 1  # lattice constant along y
N_sam = 10  # Brillouin sampling points ***
N_max = 2  # maximum band number ***
np.set_printoptions(precision=3, suppress=True)

Polarflag = 1  # 鏋佸寲鏂瑰紡锛?锛孴M鏋佸寲锛?锛孴E鏋佸寲

shifted = 0.0
Nkxg = 200
Nkyg = 1
Pkxg = (4 * np.pi ) / Pxg / np.sqrt(3)
Pkyg = (4 * np.pi ) / Pyg / np.sqrt(3)
deltakxg = Pkxg / Nkxg
# deltakyg = Pkyg / Nkyg
NK = Nkxg * Nkyg
kcx = np.zeros((NK, 4))
kcy = np.zeros((NK, 4))

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
            omega, field_temp, eps_arr = eigs_PCs(Pxg, Pyg,
                                                  (m+0.5-Nkxg/2) * deltakxg * np.sqrt(3), (n-0.5) * deltakxg,
                                                  N_max, 0, Polarflag)
            fields.append(field_temp)

    # data = scipy.io.loadmat('/home/yiloong/Documents/MATLAB/Chern/matlab.mat')
    # fields = data['fields']
    _, _, eps_arr = eigs_PCs(Pxg, Pyg, 0, 0, N_max, 0, Polarflag)

    for m in range(Nkxg):
        for n in range(Nkyg):
            field1 = fields[m*(Nkyg+1)+n]
            field4 = fields[m*(Nkyg+1)+n+1]
            field3 = fields[(m+1)*(Nkyg+1)+n+1]
            field2 = fields[(m+1)*(Nkyg+1)+n]
            # field1 = fields[..., m*(Nky+1)+n]
            # field4 = fields[..., m*(Nky+1)+n+1]
            # field3 = fields[..., (m+1)*(Nky+1)+n+1]
            # field2 = fields[..., (m+1)*(Nky+1)+n]

            for mm in range(N_max):
                temp = field1[:, mm].conj().T * eps_arr @ field2[:, mm]
                fieldtemp[mm] = temp / np.abs(temp)
                temp = field2[:, mm].conj().T * eps_arr @ field3[:, mm]
                fieldtemp[mm] *= temp / np.abs(temp)
                temp = field3[:, mm].conj().T * eps_arr @ field4[:, mm]
                fieldtemp[mm] *= temp / np.abs(temp)
                temp = field4[:, mm].conj().T * eps_arr @ field1[:, mm]
                fieldtemp[mm] *= temp / np.abs(temp)
                fields_cur[m, mm] = -np.imag(np.log(fieldtemp[mm]))/(deltakxg**2*np.sqrt(3))

    print('Elapsed:', time.time() - time_s)
    plt.plot(fields_cur[:, 0]*(2*np.pi)**2)
    #np.savetxt('berry_cur.csv', fields_cur, delimiter=',', fmt='%f')
    plt.show()



print('end')
