from generate_sky import generate_sky


if __name__ == "__main__":
    """
    Let's generate a small test file that can be used for unit tests etc
    """
    mjd0 = 59853  # 2022-10-01
    test_length = 32  # days
    generate_sky(mjd0=mjd0, mjd_max=mjd0+test_length, outpath='healpix')

    # Also do 3 days
    test_length = 3.1  # days
    generate_sky(mjd0=mjd0, mjd_max=mjd0+test_length, outpath='healpix')
