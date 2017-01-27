#--------------------------------
# Name:         python_common.py
# Purpose:      Common Python support functions
# Created       2017-01-26
# Python:       2.7
#--------------------------------

import argparse
import glob
import logging
import os


def get_ini_path(workspace):
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    ini_path = tkFileDialog.askopenfilename(
        initialdir=workspace, parent=root, filetypes=[('INI files', '.ini')],
        title='Select the target INI file')
    root.destroy()
    return ini_path


def parse_int_set(nputstr=""):
    """Return list of numbers given a string of ranges

    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
    """
    selection = set()
    invalid = set()

    # Tokens are comma seperated values
    # AttributeError will get raised when nputstr is empty
    try:
        tokens = [x.strip() for x in nputstr.split(',')]
    except AttributeError:
        return set()

    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    # print "Invalid set: " + str(invalid)
    return selection


def remove_file(file_path):
    """Remove a feature/raster and all of its anciallary files"""
    file_ws = os.path.dirname(file_path)
    for file_name in glob.glob(os.path.splitext(file_path)[0] + ".*"):
        os.remove(os.path.join(file_ws, file_name))


def valid_file(arg):
    if os.path.isfile(arg):
        return arg
    else:
        raise argparse.ArgumentTypeError('{} does not exist'.format(arg))


def month_range(start, end):
    """Generate month numbers between start and end, wrapping if necessary

    Equivalent to wrapped_range(start, end, x_min=1, x_max=12)

    Args:
        start (int): Start month
        end (int): End month

    Yields:
        int: The next month number

    Examples:
        >>> month_range(1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> month_range(10, 9))
        [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> month_range(3, 5))
        [3, 4, 5]
        >>> month_range(10, 1))
        [10, 11, 12, 1]

    """
    m = int(start)
    while True:
        yield m
        if m == end:
            break
        m += 1
        if m > 12:
            m = 1


def wrapped_range(start, end, x_min=1, x_max=12):
    """Return the values between a range b for a given start/end

    Args:
        start (int): Start value
        end (int): End value
        x_min (int): Minimum value
        x_max (int): Maximum value

    Yields:
        int: The next number in the wrapped range

    Examples:
        >>> wrapped_range(1, 12, 1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> wrapped_range(None, None, 1, 12))
        []
        >>> wrapped_range(None, 12, 1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> wrapped_range(1, None, 1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> wrapped_range(10, 9, 1, 12))
        [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> wrapped_range(3, 5, 1, 12))
        [3, 4, 5]
        >>> wrapped_range(10, 1, 1, 12))
        [10, 11, 12, 1]

    """
    if start is None and end is None:
        return
    if start is None:
        start = x_min
    if end is None:
        end = x_max

    x = int(start)
    while True:
        yield x
        if x == end:
            break
        x += 1
        if x > x_max:
            x = x_min
