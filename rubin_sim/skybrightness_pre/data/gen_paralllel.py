import ipyparallel as ipp
import numpy as np

# Generate all the sky save files in parallel
# start engines first with, e.g., ipcluster start -n 4
# Very random note, the engines may need to be started in the same directory as the code? I
# never understand python name-space issues, especially in parallel.

if __name__ == "__main__":

    # Connect to parallel clients
    rc = ipp.Client()
    dview = rc[:]

    # import the function on all the engines
    dview.execute("import generate_sky")

    # Make a quick small one for speed loading
    #generate_sky(mjd0=59579, mjd_max=59579+10., outpath='healpix_6mo', outfile='small_example.npz_small')

    nyears = 20  # 13
    day_pad = 5
    #day_pad = 30
    # Full year
    # mjds = np.arange(59560, 59560+365.25*nyears+day_pad+366, 366)
    # 6-months
    mjds = np.arange(59560, 59560+366*nyears+366/2., 366/2.)

    #result = dview.map_sync(lambda mjd1, mjd2:
    #                        generate_sky.generate_sky(mjd0 = mjd1, mjd_max=mjd2+30, outpath='healpix_6mo', verbose=False),
    #                        mjds[:-1], mjds[1:])

    result = dview.map_sync(lambda mjd1, mjd2:
                            generate_sky.generate_sky(mjd0 = mjd1, mjd_max=mjd2, outpath='opsimFields_20', fieldID=True, verbose=False),
                            mjds[:-1], mjds[1:]+day_pad)

