import struct
import os
import gc
import array_api_compat.cupy as xp
# import array_api_compat.torch as xp
import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
from cupyx.scipy.ndimage import median_filter
import pandas as pd

def lm_em_update(x_cur, op, adjoint_ones):
    '''
    Update the image estimate using the EM algorithm'''
    epsilon = 1e-10  #  If ybar contains zeros, dividing by it can produce NaNs
    ybar = op(x_cur)
    x = x_cur * op.adjoint(1 / (ybar + epsilon)) / adjoint_ones
    return x

def parallelproj_listmode_reconstruction(psf_path, img_shape=(248, 140, 176), voxel_size=(1.9531, 1.9531, 1.5), 
                                         scanner='vision', num_subsets=2, osem_iterations=3, sensitivity_array=None):
    '''
    Reconstruction using parallelproj library'''
    # choose a device (CPU or CUDA GPU)
    if "numpy" in xp.__name__:
        # using numpy, device must be cpu
        dev = "cpu"
    elif "cupy" in xp.__name__:
        # using cupy, only cuda devices are possible
        dev = xp.cuda.Device(0)
    elif "torch" in xp.__name__:
        # using torch valid choices are 'cpu' or 'cuda'
        dev = "cuda"

    if scanner == 'vision':
        # Siemens Vision Scanner
        num_rings = 80
        radius = 410.0  # mm
        max_z = 131.15  # mm
        num_sides = 760
        # TOF
        TOF_resolution = 225
        psf = 3.5  # resolution of the scanner in mm        
        depth = 20  #mm depth of crystal (depth is the max DOI if all photons are incident perpendicular to the crystal)
        mu = 0.082 # mm^-1 attenuation coefficient
        FWHM_detector= 3.39  # mm (given by detector side length, 2*pi*radius/num_sides)
    else:
        raise ValueError("Scanner not supported")
    
    tofbin_FWHM = TOF_resolution * 1e-12 * 3e8 / 2 *1e3 # *1e3 to mm;  *1e-12 to s; *3e8 to m/s;  /2 to get one-way distance;
    sigma_tof = tofbin_FWHM / 2.355 # to get sigma from FWHM
    tofbin_width = sigma_tof * 1.03  # sigma_tof * 1.03, as given in https://parallelproj.readthedocs.io/en/stable/python_api.html#module-parallelproj.tof # ps, it is the minimum time difference between the arrival of two photons that it can detect. it is diveded by 2 because if one of them arrivs TOF_resolution

    num_tofbins = 51
    if num_tofbins % 2 == 0:
        num_tofbins -= 1
    print("num_tofbins", num_tofbins)
    tof_params = parallelproj.TOFParameters(
        sigma_tof=sigma_tof, num_tofbins=num_tofbins, tofbin_width=tofbin_width
    )
    enable_tof = True

    # Blurring due to detector resolution, crystal size, DOI, positron range
    res_model = parallelproj.GaussianFilterOperator(
        img_shape, sigma=psf / (2.35 * xp.asarray(voxel_size))
    )
    
    adjoint_ones = to_device(xp.asarray(sensitivity_array, dtype=xp.float32), dev)
    
    # Define the structure format for one data record
    format_string = 'Q f i f f f f f f h h'
    record_size = struct.calcsize(format_string)

    # Define the dtype for numpy based on the format string
    dtype = np.dtype([
        ('emission_time', 'u8'), # unsigned long long int (Q) (emission_time (ps))
        ('travel_time', 'f4'),   # float (f) (travel_time (ps))
        ('emission_voxel', 'i4'),# int (i) (emission voxel)
        ('energy', 'f4'),        # float (f) (energy)
        ('z', 'f4'),             # float (f) (z (cm))
        ('phi', 'f4'),           # float (f) (phi (rad))
        ('vx', 'f4'),            # float (f) (vx; x component of the incident photon direction)
        ('vy', 'f4'),            # float (f) (vy; y component of the incident photon direction)
        ('vz', 'f4'),            # float (f) (vz; z component of the incident photon direction)
        ('index1', 'i2'),        # short int (h)  Flag for scatter: =0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter)
        ('index2', 'i2')         # short int (h) (index2)
    ])

    # Read the binary file in chunks and convert directly to DataFrame
    chunk_size = 100000
    events = []
    with open(psf_path, 'rb') as file:
        while True:
            data = file.read(record_size * chunk_size)
            if not data:
                break
            chunk = np.frombuffer(data, dtype=dtype)
            event = pd.DataFrame(chunk).loc[:, ['travel_time','z', 'phi', 'vx', 'vy', 'vz']]
            events.append(event)

    events = pd.concat(events, ignore_index=True)
    num_events = events.shape[0]
    travel_time = xp.asarray(events.travel_time)
    vx = xp.asarray(events.vx)
    vy = xp.asarray(events.vy)
    vz = xp.asarray(events.vz)
    phi = xp.asarray(events.phi)
    events_x = radius * xp.cos(phi)
    events_y = radius * xp.sin(phi)
    events_z = xp.asarray(events.z)

    # 1. DOI effect
    # Accounting for the angle of incidence of the photon, larger DOI if incidence angle is larger
    # We are getting the length of the path inside the detectors of each photon, solving:
    # r_salida (posicion donde sale del scanner) = r_vec (posicion donde llega al scanner) + v (direccion incidente) * DOI (escalar); sqrt(r_salida(0)**2 + r_salida(1)**2) = R+20mm (radio del escaner + anchura del cristal, ya que sale en el borde del cristal)

    max_dois = 1 / (vx**2 + vy**2) * (-(vx * events_x + vy * events_y) + xp.sqrt((vx * events_x + vy * events_y)**2 - (vx**2 + vy**2) * (events_x**2 + events_y**2 - (radius + depth)**2)))
    del events, data, chunk, event

    uniform_rands = xp.random.uniform(0, 1, num_events)  # one sample per event

    # Cumulative distribution function capturing that the probability of 
    # a photon being detected is higher when photons enter the crystal 
    # and decreases exponentially as they move into the crystal, considering a maximum depth of d and normalizing
    # F(x) = 1 - exp(-mu*x) / (1 - exp(-mu*max_doi))
    # Then, inverse transform sampling to sample values from the distribution
    # Sampling from this inverse function means picking random probabilities and finding the points on the distribution that correspond to those probabilities
    # (inverse of the CDF: F^-1(u) = -ln(1 - u(1 - exp(-mu*d))) / mu;

    dois = -np.log(1 - uniform_rands * (1 - xp.exp(-mu * max_dois))) / mu
    del uniform_rands

    # 2. Detector resolution effect:
    # The detector resolution is modeled as a Gaussian distribution with a FWHM of 3.39 mm
    # Values are sampled from a normal distribution with FWHM = 3.39 mm and added to the x, y, and z coordinates of the events
    sigma_detector = FWHM_detector / 2.355
    mean = 0.0
    event_displacement = xp.random.normal(mean, sigma_detector, num_events * 3)
    
    events_x = events_x + event_displacement[:num_events] + vx * dois  # x position of the detector
    events_y = events_y + event_displacement[num_events:2*num_events] + vy * dois  # y position of the detector
    events_z = events_z * 10.0 + event_displacement[2*num_events:3*num_events] + vz * dois   # *10.0 for cm to mm

    # 3. Crystal size effects:
    # the angle and z position of the event are rounded to the position of the crystal
    crystal_phi_positions = xp.linspace(0, 2 * xp.pi, num_sides, endpoint=False) - np.pi
    events_phi = xp.arctan2(events_y, events_x)  # angle of the event once DOI and detector resolution are considered
    events_phi[events_phi > xp.pi - 2*xp.pi/num_sides] -= 2*xp.pi  # wrap around, so that if it wont be assigned to the bin below when it should be assigned to phi=-pi
    events_phi = xp.digitize(events_phi, crystal_phi_positions) * 2 * xp.pi / num_sides - xp.pi  # round to the closest crystal phi position
    events_x = radius * xp.cos(events_phi)  # x position of the crystal corresponding to the event
    events_y = radius * xp.sin(events_phi)  # y position of the crystal corresponding to the event
    crystal_z_positions = xp.linspace(-max_z, max_z, num_rings)
    events_z = xp.digitize(xp.asarray(events_z), crystal_z_positions) * 2 * max_z / num_rings - max_z  # round to the closest crystal z position

    # TOF bin
    bin = xp.round((travel_time[0::2] - travel_time[1::2]) / (TOF_resolution / 2.355 * 1.03)).astype(int)  # / 2.355 * 1.03 to match the spatial tof_bin width
    bin = xp.repeat(bin, 2)
    del crystal_phi_positions, crystal_z_positions, events_phi, radius, event_displacement, dois, travel_time, vx, vy, vz, max_dois, phi

    event_start_coordinates = xp.asarray(xp.stack((events_x[0::2], events_y[0::2], events_z[0::2]), axis=1))
    event_end_coordinates = xp.asarray(xp.stack((events_x[1::2], events_y[1::2], events_z[1::2]), axis=1))
    event_tof_bins = bin[0::2]
    del events_x, events_y, events_z, bin

    lm_proj = parallelproj.ListmodePETProjector(
        event_start_coordinates, event_end_coordinates, img_shape, voxel_size
    )

    if enable_tof:
        lm_proj.tof_parameters = tof_params
        lm_proj.event_tofbins = event_tof_bins
        lm_proj.tof = enable_tof
        
    subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

    lm_pet_subset_linop_seq = []

    for i, sl in enumerate(subset_slices):
        subset_lm_proj = parallelproj.ListmodePETProjector(
            event_start_coordinates[sl, :], event_end_coordinates[sl, :], img_shape, voxel_size
        )

        # enable TOF in the LM projector
        subset_lm_proj.tof_parameters = lm_proj.tof_parameters
        if lm_proj.tof:
            subset_lm_proj.event_tofbins = 1 * event_tof_bins[sl]
            subset_lm_proj.tof = lm_proj.tof

        lm_pet_subset_linop_seq.append(
            parallelproj.CompositeLinearOperator(
                (subset_lm_proj, res_model)
            )
        )

    
    del event_start_coordinates, event_end_coordinates, event_tof_bins

    lm_pet_subset_linop_seq = parallelproj.LinearOperatorSequence(lm_pet_subset_linop_seq)

    # number of OSEM iterations
    num_iter = osem_iterations
    beta = 0.05
    adjoint_ones = adjoint_ones + adjoint_ones.max()  ###
    x = xp.ones(img_shape, dtype=xp.float32, device=dev)

    for i in range(num_iter):
        for k, sl in enumerate(subset_slices):
            print(f"OSEM iteration {(k+1):03} / {(i + 1):03} / {num_iter:03}", end="\r")
            x = lm_em_update(
                x,
                lm_pet_subset_linop_seq[k],
                adjoint_ones / num_subsets,
            )
            x = (1.0-beta)*x + beta*median_filter(x, size=3)
    x = x.get()
    for subset in lm_pet_subset_linop_seq:
        del subset
    del lm_pet_subset_linop_seq, lm_proj, adjoint_ones, subset_lm_proj
    xp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    os.remove(psf_path)
    
    return x