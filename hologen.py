import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import jn, kn
import sys
import time
from tqdm import tqdm

def besselj(l, U):
    return jn(l,U)

def besselk(l, U):
    return kn(l,U)

def EigenvalueEquation(U, l, m, V):
    # Eigenvalue equation as given in Snyder & Love.
    # Equivalent to the Eigenvalue equation given in Gloge.
    W = np.sqrt(V**2 - U**2)
    # Original in matlab
    # retval =  U*besselj(l+1,U)/besselj(l,U) - W*besselk(l+1,W)/besselk(l,W)
    return U*scipy.special.jv(l+1,U)/scipy.special.jv(l,U) - W*scipy.special.jv(l,U)

def find_peaks(x):
    return scipy.signal.find_peaks(x)

def overlap_integral(M1, M2):
    RetVal = np.abs(np.sum(np.sum(M1 * np.conj(M2))))**2
    RetVal /= np.sum(np.sum(np.abs(M1 * np.conj(M1))))
    RetVal /= np.sum(np.sum(np.abs(M2 * np.conj(M2))))
    RetVal = np.sqrt(RetVal)
    return RetVal


# Global variables
def LP_Mode(x, l, m):

    # User-Defined Parameters + Derived Parameters
    diameter = 25e-6
    a = diameter / 2
    wavelength = 633e-9
    n_co = 1.457  # Silica
    n_cl = 1.4536
    k = 2 * np.pi / wavelength
    NA = np.sqrt(n_co**2 - n_cl**2)
    V = a * k * NA
    Delta = (1 - n_cl**2 / n_co**2) / 2

    # Return structure
    Mode = {'WaveguideDiameter': diameter,
            'WaveguideRadius': a,
            'Wavelength': wavelength,
            'n_co': n_co,
            'n_cl': n_cl,
            'k': k,
            'WaveguideNA': NA,
            'IndexContrast': Delta,
            'l': l,
            'm': m}

    # Error Checks
    if n_co < n_cl:
        print('WARNING: n_co must be > n_cl')
        return Mode

    if l < 0:
        print('WARNING: l must be >= 0')
        return Mode

    if m < 1:
        print('WARNING: m must be >= 1')
        return Mode

    if Delta > 1:
        print('WARNING: Waveguide is not weakly guiding')
        return Mode

    # Eigenvalue equation
    arrU = np.arange(0, V, 0.001)
    idxs, _ = find_peaks(-np.abs(EigenvalueEquation(arrU, l, m ,V)))

    if m > len(idxs):
        Guided = False
        Mode['bolGuided'] = Guided
    else:
        U = arrU[idxs[m - 1]]
        W = np.sqrt(V**2 - U**2)
        beta = V / a / (2 * Delta)**(1 / 2) * (1 - 2 * Delta * U**2 / V**2)**(1 / 2)

        if beta < k * n_cl or beta > k * n_co:
            Guided = False
        else:
            Guided = True

        Mode.update({'V': V, 'U': U, 'W': W, 'bolGuided': Guided, 'Beta': beta})

    # Generate Fields
    if Guided:
        x_mesh, y_mesh = np.meshgrid(x, x)
        normradius = np.sqrt(x_mesh**2 + y_mesh**2) / a

        # Non-rotated mode
        theta = np.arctan2(y_mesh, x_mesh)

        F = np.zeros_like(normradius)
        F[normradius <= 1] = np.divide(besselj(l, U * normradius[normradius <= 1]),
                                       besselj(l, U))
        F[normradius > 1] = np.divide(besselk(l, W * normradius[normradius > 1]),
                                      besselk(l, W))

        F = F * np.cos(l * theta)
        Mode.update({'F': F, 'x': x})

        if l != 0:
            # Rotated mode
            theta = theta + np.pi / (2 * l)

            F_rotated = np.zeros_like(normradius)
            F_rotated[normradius <= 1] = np.divide(besselj(l, U * normradius[normradius <= 1]),
                                                    besselj(l, U))
            F_rotated[normradius > 1] = np.divide(besselk(l, W * normradius[normradius > 1]),
                                                   besselk(l, W))

            F_rotated = F_rotated * np.cos(l * theta)
            Mode['F_rotated'] = F_rotated

    return F


def direct_search_symmetry_binary(illum, target, replay_mask, trainingloops):
    # Parameters
    bol_output_to_file = False
    Nx = illum.shape[0]
    illum_edge = Nx // 2
    
    # Error Checks
    if illum.shape != target.shape or illum.shape != replay_mask.shape:
        raise ValueError('illumination, target, and replay_mask must be of the same size')

    # Pre-Processing
    x = np.linspace(0, Nx-1, Nx)
    x_mesh, y_mesh = np.meshgrid(x, x)
    
    # Determine UD axis of symmetry
    target_top = target[:Nx//2, :]
    target_bottom = target[Nx//2:, :]
    target_bottom = np.flipud(target_bottom)
    up_down_symmetry = 1 if np.sum(np.abs(target_top - target_bottom)) < 1 else -1
    
    # Determine LR axis of symmetry
    target_left = target[:, :Nx//2]
    target_right = target[:, Nx//2:]
    target_right = np.fliplr(target_right)
    left_right_symmetry = 1 if np.sum(np.abs(target_left - target_right)) < 1 else -1
    
    # fftshifts
    target = np.fft.fftshift(target)
    replay_mask = np.fft.fftshift(replay_mask)
    illum = np.fft.fftshift(illum)
    
    # Coordinate calculations
    x_mesh = x_mesh[:Nx//2, :Nx//2]
    y_mesh = y_mesh[:Nx//2, :Nx//2]
    
    # Initial hologram value.
    holo = illum * (np.random.randint(2, size=illum.shape) * 2 - 3)
    replay = np.fft.fft2(holo)

    # Ensure power in target is half the power in the hologram
    target = target / np.sqrt(np.sum(np.abs(target)**2)) * np.sqrt(np.sum(np.abs(holo)**2)) * np.sqrt(Nx) * np.sqrt(Nx)
   
    # Take NW quadrant in all cases
    target = target[:Nx//2, :Nx//2]
    replay_mask = replay_mask[:Nx//2, :Nx//2]
    holo = holo[:Nx//2, :Nx//2]
    x_mesh = x_mesh[:Nx//2, :Nx//2]
    y_mesh = y_mesh[:Nx//2, :Nx//2]

    replay = replay[:Nx//2, :Nx//2]

    # Apply mask to relevant arrays
    x_mesh = x_mesh[replay_mask]
    y_mesh = y_mesh[replay_mask]
    #replay = replay[:Nx//2, :Nx//2]
    replay = replay[replay_mask]
    masked_target = np.conj(target[replay_mask])

    target_power = np.sum(np.abs(target)**2)
    masked_target_power = np.sum(np.abs(masked_target)**2)
    
    # Direct Search
    max_local_c = 0
    max_global_c = 0
    iter_no = 0
    disp_str = ''
    gamma = 10
    last_displayed = 0

    
    for aaaaa in tqdm(range(trainingloops)): 
        m = np.random.randint(1, illum_edge + 1)
        n = np.random.randint(1, illum_edge + 1)

        old_pixel = holo[m-1, n-1]

        if np.abs(old_pixel) < 0.001 and np.sqrt(m**2 + n**2) < illum_edge:
            illum_edge = int(np.round(np.sqrt(m**2 + n**2)))
            continue
        new_pixel = -old_pixel

        # Update replay
        delta_replay = (new_pixel - old_pixel) * np.exp(-2 * np.pi * 1j / Nx * ((n - 1) * x_mesh + (m - 1) * y_mesh))
        delta_replay += left_right_symmetry * (new_pixel - old_pixel) * np.exp(-2 * np.pi * 1j / Nx * (-(n - 1) * x_mesh + (m - 1) * y_mesh))
        delta_replay += up_down_symmetry * (new_pixel - old_pixel) * np.exp(-2 * np.pi * 1j / Nx * ((n - 1) * x_mesh - (m - 1) * y_mesh))
        delta_replay += left_right_symmetry * up_down_symmetry * (new_pixel - old_pixel) * np.exp(-2 * np.pi * 1j / Nx * (-(n - 1) * x_mesh - (m - 1) * y_mesh))

        new_replay = replay + delta_replay
        
        # Calculate local overlap integral
        overlap_integral = np.abs(np.sum(new_replay * masked_target))**2
        masked_replay_power = np.sum(np.abs(new_replay)**2)
        local_c = overlap_integral / masked_replay_power
        local_c /= masked_target_power
        global_c = masked_replay_power / target_power
        
        # Determine if result is better
        if local_c**gamma * global_c > max_local_c**gamma * max_global_c:
            max_local_c = local_c
            max_global_c = global_c
            holo[m-1, n-1] = new_pixel
            replay = new_replay
        
    
    holo = np.concatenate((holo, left_right_symmetry * np.fliplr(holo)), axis=1)
    holo = np.concatenate((holo, up_down_symmetry * np.flipud(holo)), axis=0)
    holo = np.fft.fftshift(holo)
    illum = np.fft.fftshift(illum)
    holo[holo >= 0] = 1
    holo[holo < 0] = -1
    '''
    ret_val = {
        'Holo': holo,
        'Illum': illum,
        'DiffractionField': holo * illum,
        'Replay': np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo * illum))),
        'globalc': max_global_c,
        'localc': max_local_c
    }
    '''
    replay = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo * illum)))
    
    return holo, replay



if __name__ == '__main__':
    # User-Entered Parameters
    l = 2
    m = 1
    V = 0
    wavelength = 633e-9
    f1 = 15e-3
    f2 = 10e-3
    Nx = 5000
    Magnification = 1
    HoloDiameter = 1250  # in Pixels
    FibreDiameter = 25e-6
    dx = 3.74e-6
    SMF_NA = 0.14
    trainingloops = 100000


    # Illumination Mask
    x = np.linspace(-Nx/2, Nx/2, Nx)
    x_mesh, y_mesh = np.meshgrid(x, x)
    r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
    Illum = np.ones_like(r_mesh)
    Illum[r_mesh > HoloDiameter/2] = 0

    w0 = f1 * SMF_NA
    x = np.linspace(0, Nx-1, Nx)
    x = x - (Nx-1)/2
    x = x * dx
    x_mesh, y_mesh = np.meshgrid(x, x)
    r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
    Illum = Illum * np.exp(-r_mesh**2 / w0**2)

    # Correct dx for magnification
    dx = dx * Magnification

    # Target Replay Field
    du = wavelength * f2 / (Nx * dx)
    u = np.linspace(0, Nx-1, Nx)
    u = u - (Nx-1)/2
    u = u * du

    Mode = LP_Mode(u, l, m)
    Target = Mode

    # Assuming u and Target are defined
    u_mesh, v_mesh = np.meshgrid(u, u)
    r_mesh = np.sqrt(u_mesh**2 + v_mesh**2)
    Mask = np.ones_like(Target)
    Mask[r_mesh > 20e-6] = 0
    Mask = Mask.astype(bool)

    t1 = time.time()
    Holo, Replay = direct_search_symmetry_binary(Illum, Target, Mask, trainingloops)       
    t2 = time.time()

    elapsed_time = t2 - t1
    print('total time to run search:', elapsed_time)

    #sys.exit()



    plt.figure()
    plt.imshow((Holo), interpolation='nearest')
    plt.title('Hologram')
    plt.axis('square')
    #plt.show()


    # Specify the path to save the CSV file
    csv_file_path = 'pytestholo21.csv'
