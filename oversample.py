"""
Functions for oversampling correction
"""

import numpy as np


def distance(a, b):
    """
    Calculate distance between coordinates a and b.

    Params
    ------
    a : tuple
    b : tuple

    Returns
    -------
    out : float
        Squared distance between coordinates a and b.
    """

    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def oversamplecorr(centroids, radius):
    """
    Correction for oversampling given list of centroids.

    Params
    ------
    centroids : dictionary
        Dictionary of centroids where key is slice position and items are lists
        of coordinate positions of detected cells.
    radius : int
        Radius with which to claim cells are overlapping.

    Returns
    -------
    out : dictionary
        Output of dictionary of oversampled corrected cell positions.
    """

    keepcentroids = {}
    overlapcentroids = {}
    i = 0

    # First check if there are more than two layers
    if len(list(centroids.keys())) > 1:
        # Loop through successive layers and identify overlapping cells
        for layer1, layer2 in zip(list(centroids.keys())[:-1], list(centroids.keys())[1:]):
            # First check if layers are successive otherwise you cannot correct
            if layer2-layer1 == 1:
                # Store cell centroids for each layer
                layer1centroids = centroids[layer1]
                layer2centroids = centroids[layer2]

                # Loop through each cell in layer 1 and check if overlapping
                for cell in layer1centroids:
                    # Get a boolean list with True in position of cell in layer 2 if cell in layer 1 overlaps and is the minumum distance
                    distances = np.array([distance(cell, cell2) for cell2 in layer2centroids])
                    mindistance = distances == np.min(distances)
                    withindistance = np.array([distance(cell, cell2)<=radius**2 for cell2 in layer2centroids])
                    overlapping = mindistance&withindistance

                    # First check if cell is already within the overlap dictionary, overlapcentroids
                    overlapkey = [key for key, value in overlapcentroids.items() if cell in value]

                    # If there is a True in the overlapping list, then there is a minimum distance oversampled cell detected
                    if True in overlapping:
                        # If so, only add the paired cell
                        if overlapkey:
                            overlapcentroids.setdefault(overlapkey[0],[]).append(layer2centroids[np.argmax(overlapping)])

                        # Else, add both the new cell and pair to it's own unique dictionary key
                        else:
                            overlapcentroids.setdefault(i,[]).append(cell)
                            overlapcentroids.setdefault(i,[]).append(layer2centroids[np.argmax(overlapping)])
                            # Update counter to keep track of number of overlapped cells in total
                            # Uses this as key
                            i += 1

                    # Only if all overlapping is False and the cell is not detected in overlapcentroids already, then add cell to keep
                    if (not True in overlapping) and (not overlapkey):
                        # If no overlap is detected, then stick cell into keep dictionary
                        keepcentroids.setdefault(cell[2], []).append(cell)
            else:
                layer1centroids = centroids[layer1]
                for cell in layer1centroids:
                    keepcentroids.setdefault(cell[2], []).append(cell)

        # Account for the last layer
        layer2centroids = centroids[layer2]
        for cell in layer2centroids:
            overlapkey = [key for key, value in overlapcentroids.items() if cell in value]
            if overlapkey:
                break
            else:
                keepcentroids.setdefault(cell[2], []).append(cell)

        # Go through each overlapping cell and take the middle cell
        # Stick middle cell into the keep dictionary at the relevant slice position
        for key, overlapcells in overlapcentroids.items():
            midcell = overlapcells[int(len(overlapcells)/2)]
            keepcentroids.setdefault(midcell[2], []).append(midcell)

    else:
        keepcentroids = centroids

    return keepcentroids
