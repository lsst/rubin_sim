from generate_sky import generate_sky

if __name__ == "__main__":

    mjd0 = 60218.0
    length = 8
    generate_sky(mjd0=mjd0, mjd_max=mjd0 + length)
