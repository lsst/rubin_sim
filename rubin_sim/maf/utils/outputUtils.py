from __future__ import print_function
from builtins import map
from builtins import str
from builtins import range
import sys
import numpy as np

__all__ = ['nameSanitize', 'printDict', 'printSimpleDict']


def nameSanitize(inString):
    """
    Convert a string to a more file name (and web) friendly format.

    Parameters
    ----------
    inString : str
        The input string to be sanitized. Typically these are combinations of metric names and metadata.

    Returns
    -------
    str
        The string after removal/replacement of non-filename friendly characters.
    """
    # Replace <, > and = signs.
    outString = inString.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')
    # Remove single-spaces, strip '.'s and ','s
    outString = outString.replace(' ', '_').replace('.', '_').replace(',', '')
    # and remove / and \
    outString = outString.replace('/', '_').replace('\\', '_')
    # and remove parentheses
    outString = outString.replace('(', '').replace(')', '')
    # Remove ':' and ';"
    outString = outString.replace(':', '_').replace(';', '_')
    # Replace '%' and #
    outString = outString.replace('%', '_').replace('#', '_')
    # Remove '__'
    while '__' in outString:
        outString = outString.replace('__', '_')
    return outString


def _myformat(args, delimiter=' '):
    # Generic line formatter to let you specify delimiter between text fields.
    writestring = ''
    # Wrap in a list if something like an int gets passed in
    if not hasattr(args, '__iter__'):
        args = [args]
    for a in args:
        if isinstance(a, list):
            if len(a) > 1:
                ap = ','.join(map(str, a))
            else:
                ap = ''.join(map(str, a))
            writestring += '%s%s' % (ap, delimiter)
        else:
            writestring += '%s%s' % (a, delimiter)
    return writestring


def _myformatdict(adict, delimiter=' '):
    # Generic line formatter used for dictionaries.
    writestring = ''
    for k, v in adict.items():
        if isinstance(v, list):
            if len(v) > 1:
                vp = ','.join(map(str, v))
            else:
                vp = ''.join(map(str, v))
            writestring += '%s:%s%s' % (k, vp, delimiter)
        else:
            writestring += '%s:%s%s' % (k, v, delimiter)
    return writestring


def printDict(content, label, filehandle=None, delimiter=' ', _level=0):
    """
    Print dictionaries (and/or nested dictionaries) nicely.
    Can also print other simpler items (such as numpy ndarray) nicely too.
    This is used to print the config files.

    Parameters
    ----------
    content : dict
        The content to pretty print.
    label : str
        A header for this level of the dictionary.
    filename : file
        Output destination. If None, prints to stdout.
    delimiter : str
        User specified delimiter between fields.
    _level : int
        Internal use (controls level of indent).
    """
    # Get set up with basic file output information.
    if filehandle is None:
        filehandle = sys.stdout
    # And set character to use to indent sets of parameters related to a single dictionary.
    baseindent = '%s' % (delimiter)
    indent = ''
    for i in range(_level-1):
        indent += '%s' % (baseindent)
    # Print data (this is also the termination of the recursion if given nested dictionaries).
    if not isinstance(content, dict):
        if isinstance(content, str) or isinstance(content, float) or isinstance(content, int):
            print('%s%s%s%s' % (indent, label, delimiter, str(content)), file=filehandle)
        else:
            if isinstance(content, np.ndarray):
                if content.dtype.names is not None:
                    print('%s%s%s' % (indent, delimiter, label), file=filehandle)
                    for element in content:
                        print('%s%s%s%s%s' % (indent, delimiter, indent, delimiter, _myformat(element)), file=filehandle)
                else:
                    print('%s%s%s%s' % (indent, label, delimiter, _myformat(content)), file=filehandle)
            else:
                print('%s%s%s%s' % (indent, label, delimiter, _myformat(content)), file=filehandle)
        return
    # Allow user to specify print order of (some or all) items in order via 'keyorder'.
    #  'keyorder' is list stored in the dictionary.
    if 'keyorder' in content:
        orderkeys = content['keyorder']
        # Check keys in 'keyorder' are actually present in dictionary : remove those which aren't.
        missingkeys = set(orderkeys).difference(set(content.keys()))
        for m in missingkeys:
            orderkeys.remove(m)
        otherkeys = sorted(list(set(content.keys()).difference(set(orderkeys))))
        keys = orderkeys + otherkeys
        keys.remove('keyorder')
    else:
        keys = sorted(content.keys())
    # Print data from dictionary.
    print('%s%s%s:' % (indent, delimiter, label), file=filehandle)
    _level += 2
    for k in keys:
        printDict(content[k], k, filehandle, delimiter, _level)
    _level -= 2


def printSimpleDict(topdict, subkeyorder, filehandle=None, delimiter=' '):
    """
    Print a simple one-level nested dictionary nicely across the screen,
    with one line per top-level key and all sub-level keys aligned.

    Parameters
    ----------
    topdict : dict
        The dictionary to pretty print
    subkeyorder : list of strings
        The order to print the values of the dictionary.
    filehandle : file
        File output object, if None then uses stdout.
    delimiter : str
        User specified delimiter between fields.
    """
    # Get set up with basic file output information.
    if filehandle is None:
        filehandle = sys.stdout
    # Get all sub-level keys.
    subkeys = []
    for key in topdict:
        subkeys += list(topdict[key].keys())
    subkeys = list(set(subkeys))
    # Align subkeys with 'subkeyorder' and then alphabetize any remaining.
    missingkeys = set(subkeyorder).difference(set(subkeys))
    for m in missingkeys:
        subkeyorder.remove(m)
    otherkeys = sorted(list(set(subkeys).difference(set(subkeyorder))))
    subkeys = subkeyorder + otherkeys
    # Print header.
    writestring = '#'
    for s in subkeys:
        writestring += '%s%s' % (s, delimiter)
    print(writestring, file=filehandle)
    # Now go through and print.
    for k in topdict:
        writestring = ''
        for s in subkeys:
            if s in topdict[k]:
                if isinstance(topdict[k][s], str) or isinstance(topdict[k][s], float) or isinstance(topdict[k][s], int):
                    writestring += '%s%s' % (topdict[k][s], delimiter)
                elif isinstance(topdict[k][s], dict):
                    writestring += '%s%s' % (_myformatdict(topdict[k][s], delimiter=delimiter), delimiter)
                else:
                    writestring += '%s%s' % (_myformat(topdict[k][s]), delimiter)
            else:
                writestring += '%s' % (delimiter)
        print(writestring, file=filehandle)
