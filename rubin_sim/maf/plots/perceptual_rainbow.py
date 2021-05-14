from matplotlib.colors import LinearSegmentedColormap

def makePRCmap():
    colors = [[135, 59, 97],
              [143, 64, 127],
              [143, 72, 157],
              [135, 85, 185],
              [121, 102, 207],
              [103, 123, 220],
              [84, 146, 223],
              [69, 170, 215],
              [59, 192, 197],
              [60, 210, 172],
              [71, 223, 145],
              [93, 229, 120],
              [124, 231, 103],
              [161, 227, 95],
              [198, 220, 100],
              [233, 213, 117]]
    mpl_colors = []
    for color in colors:
        mpl_colors.append(tuple([x / 255. for x in color]))
    cmap = LinearSegmentedColormap.from_list('perceptual_rainbow', mpl_colors)
    return cmap
