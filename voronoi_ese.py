#

# -*- coding: utf-8 -*-
#
# code for adaptive spatial binning of 2D fits files.
# requieres:    -   asciidata
#               -   numpy 1.3.0 (essential for the numpy.histogram routine!!!)
#               -   pyfits
#               -   pylab (just for plotting, alternatively start program with quiet=True)
#
################################################################################
#                                                                              #
#   !!!     RECODED and modified Version of the IDL-programm by     !!!        #
#   !!!        Cappellari & Copin    and    Diehl & Statler         !!!        #
#                                                                              #
################################################################################
#
# Perform adaptive spatial binning of Input-Data to reach a chosen constant
# signal-to-noise ratio per bin.
#
# Further information on VORONOI_2D_BINNING algorithm can be found in:
#
#   1.  Cappellari M., Copin Y., 2003, MNRAS, 342, 345
#       http://www-astro.physics.ox.ac.uk/~mxc/idl
#
#   2.  S. Diehl and T. S. Statler, MNRAS, 368:497-510, May 2006
#       http://public.lanl.gov/diehl/software/wvtbinning/
#
################################################################################
#                                                                              #
# Code-Structure of this document:                                             #
#                                                                              #
#   1.  routines for adding, assigning and binning signal & noise              #
#       -   add_signal                                                         #
#       -   add_noise                                                          #
#       -   bin_signal                                                         #
#       -   bin_noise                                                          #
#       -   wvt_assign_to_bin                                                  #
#        -  wvt_assign_to_bin_scale                                            #
#                                                                              #
#   2.  computing of different centroids and roundness values                  #
#       -   wvt_unweighted_centroid                                            #
#       -   wvt_weighted_centroid                                              #
#       -   wvt_addto_unweighted_centroid                                      #
#       -   wvt_addto_weighted_centroid                                        #
#       -   wvt_bin_roundness                                                  #
#                                                                              #
#   3.  routines about pixel- and bin-neighborlists                            #
#       -   match_int                                                          #
#       -   wvt_make_neighborlist                                              #
#       -   wvt_find_neighbors                                                 #
#       -   wvt_assign_neighbors                                               #
#       -   wvt_find_binneighbors       (not used)                             #
#       -   wvt_recursive_neighbors     (not used)                             #
#                                                                              #
#   4.  routines for checking and error prevention                             #
#       -   wvt_renumber_binclass                                              #
#       -   wvt_check_all_bin                                                  #
#       -   wvt_check_binislands                                               #
#                                                                              #
#   5.  routines for the calculation of the binning properties                 #
#       -   wvt_calc_bin_sn                                                    #
#       -   compute_bin_quantities                                             #
#       -   show_binclass                                                      #
#                                                                              #
#   6.  basic routines for accretion, reassigning and equalizing of the bins   #
#       -   wvt_bin_accretion                                                  #
#       -   wvt_reassign_bad_bins                                              #
#       -   wvt_equal_mass                                                     #
#                                                                              #
#   7.  routines for reading input and writing outputs                         #
#       -   read_fits                                                          #
#       -   write_fits                                                         #
#       -   write_binning_results                                              #
#                                                                              #
#   8.  main program routines                                                  #
#       -   wvt_binning                                                        #
#       -   binning_fits                                                       #
#                                                                              #
################################################################################


import asciidata, numpy, os, pyfits, pylab, sys


################################################################################
#                                                                              #
#   1.  some simple routines for adding and bining signal & noise              #
#                                                                              #
################################################################################
def add_signal(signal):
# add the signal for given values in 'signal'
    return numpy.sum(signal)

def add_noise(noise):
# add the noise for given values in 'noise'
    return numpy.sqrt(numpy.sum(noise**2))

def bin_signal(signal):
# bin the signal given values in 'signal'
    return numpy.mean(signal)

def bin_noise(noise):
# bin the noise given values in 'noise'
    return numpy.sqrt(numpy.sum(noise**2))/float(len(noise))

def wvt_assign_to_bin(x, y, xnode, ynode, nodeSN):
# Assigns each pixel to the S/N weighted closest pixel
# i.e. this constructs the weighted voronoi tesselation
    return numpy.argmin(((x-xnode)**2 + (y-ynode)**2)*nodeSN)

def wvt_assign_to_bin_scale(x, y, xnode, ynode, scale):
# Assigns each pixel to the S/N weighted closest pixel
# i.e. this constructs the weighted voronoi tesselation
    return numpy.argmin(((x-xnode)/scale)**2 + ((y-ynode)/scale)**2)

################################################################################
#                                                                              #
#   2.  computing of different centroids and roundness values                  #
#                                                                              #
################################################################################
def wvt_unweighted_centroid(x, y):
# Computes the geometric center of one bin

    mass = float(len(x))
    xbar = numpy.sum(x)/mass
    ybar = numpy.sum(y)/mass

    return xbar, ybar

def wvt_weighted_centroid(x, y, density):
# Computes the weighted centroid of one bi, e.g. for weighted bins and if
# Gersho's conjecture is used. CAREFUL! Gersho's conjecture is invalid for
# negative data. If the total signal is negative, the weighted centroid
# will be set to the geometric center instead.

    mass = numpy.sum(density)
    # Check if the bin has negative signal
    if mass <= 0:
        # For negative overall signal take this approximation:
        ind = numpy.where(density > 0)[0]
        if  len(ind) > 0:
            # Only use positive signal and get the weighted center from there:
            mass = numpy.sum(density[ind])
            xbar = numpy.sum((x*density)[ind])/mass
            ybar = numpy.sum((y*density)[ind])/mass
        else:
            # Take the geometrical mean of the bin IF there is NO positive pixel
            mass = float(len(x))
            xbar = numpy.sum(x)/mass
            ybar = numpy.sum(y)/mass
    else:
        xbar = numpy.sum(x*density)/mass
        ybar = numpy.sum(y*density)/mass

    return xbar, ybar

def wvt_addto_unweighted_centroid(x, y, xbar_old, ybar_old, mass_old):
# For speed, this procedure computes the geometric center of a bin by adding
# a new list of xy values for an existing bin. Useful in bin accretion step.

    mass_old = float(mass_old)
    mass = float(float(len(x)) + mass_old)
    xbar = (numpy.sum(x)+(mass_old*xbar_old))/mass
    ybar = (numpy.sum(y)+(mass_old*ybar_old))/mass

    return xbar, ybar, mass


def wvt_addto_weighted_centroid(x, y, xy_mass, xbar_old, ybar_old, mass_old):
# For speed, this procedure computes the geometric center of a bin by adding
# a new list of xy values for an existing bin. Useful in bin accretion step.

    if type(xy_mass)!='numpy.ndarray': xy_mass = numpy.array([xy_mass]) 

    ind = numpy.where(xy_mass > 0)
    mass = float(numpy.sum(xy_mass[ind]) + mass_old)
    if mass > 0:
        xbar = (numpy.sum(x*xy_mass[ind])+(mass_old*xbar_old))/mass
        ybar = (numpy.sum(y*xy_mass[ind])+(mass_old*ybar_old))/mass
    else:
        xbar = xbar_old
        ybar = ybar_old

    return xbar, ybar, mass

def wvt_bin_roundness(x, y, pixelSize):
# Returns the "roundness" of a bin
# Implements equation (5) of Cappellari & Copin (2003)

    n = len(x)
    equivalentRadius = numpy.sqrt(float(n)/numpy.pi)*pixelSize
    xbar = numpy.mean(x)        # unweighted centroid here!
    ybar = numpy.mean(y)
    maxDistance = numpy.sqrt(numpy.max((x-xbar)**2 + (y-ybar)**2))
    roundness = (maxDistance/equivalentRadius) - 1.0

    return roundness


################################################################################
#                                                                              #
#   3.  routines about pixel- and bin-neighborlists                            #
#                                                                              #
################################################################################
def match_int(a, b):
# routine to match integer values in two vectors (can have different lengths)
# by using histogram algorithm

    minab = max(min(a), min(b))      # Only need intersection ranges
    maxab = min(max(a), max(b))

    # If either set is empty, or their ranges don't intersect
    suba  = []
    subb  = []
    count =  0
    if  (minab > maxab) or (maxab < 0): return suba, subb, count

    comrange = numpy.arange(minab, maxab+1)           # common values
    binrange = numpy.arange(minab-0.5, maxab+1.5,1)   # bin-borders of common values
    ha = numpy.histogram(a, binrange)[0]
    hb = numpy.histogram(b, binrange)[0]

    common = numpy.logical_and(ha, hb)   # True for common elements
    count  = numpy.sum(common)
    if count > 0:
        suba = numpy.searchsorted(a, comrange[common])   # indices in a for common elements
        subb = numpy.searchsorted(b, comrange[common])   # indices in b for common elements

    return suba, subb, count

def wvt_make_neighborlist(x, y):
# Create a list of neighbors for each pixel

    npix = len(x)
    neighborlist = numpy.zeros((npix,4), dtype=numpy.int)

    # Find sorted unique values of x and y
    uniquey  = numpy.unique(y)
    uniquex  = numpy.unique(x)

    # make a square which contains the pixel numbers [[0, 1, 2,... ,npix]]
    # and an outer edge of the width of one pixel
    # non-defined pixels and outer edge pixel are set to -1
    mask = numpy.zeros((len(uniquex)+2,len(uniquey)+2), dtype=numpy.int)-1
    for i in xrange(npix):
        ix = numpy.where(uniquex == x[i])[0]
        iy = numpy.where(uniquey == y[i])[0]
        mask[ix+1,iy+1] = i

    # each pixel gets assigned to the reversed sorted neighbourlist
    for i in xrange(len(uniquex)):
        for j in range(len(uniquey)):
            pix = mask[i,j]
            if pix >= 0:
                neighborlist[pix,:] = numpy.sort([mask[i-1,j], mask[i,j+1],
                                                  mask[i+1,j], mask[i,j-1]])  

    return neighborlist

def wvt_find_neighbors(group, neighborlist, neighbors, exneighbors, newmember):
# Given the neighborlist for each pixel in the image, and a group of test
# pixels, return a list of neighbors of the group's members that aren't
# already group members themselves.
# exneighbor has length of pixel, 0-was never exneighbor, 1-was exneighbor

    if (len(group) < 3) or (len(exneighbors) == 0):
        # reject pixels from neighbors that are already in the group
        neighbors = numpy.unique(neighborlist[group])
        subneighbor, subgroup, count = match_int(neighbors, group)
        if count > 0: neighbors[subneighbor] = -1

        # reject all neighbours with value = -1
        good = numpy.where(neighbors > -1)[0]
        if len(good) > 0: neighbors = neighbors[good]
        else:             neighbors = []

        # create exneighbor list with the same length as npixels
        # and set all actual neighbors to exneighbors
        exneighbors = numpy.zeros(neighborlist.shape[0], dtype=numpy.int)
        if len(neighbors) > 0: exneighbors[neighbors] = 1

        # set all group pixels NOT to ex-neighbors
        exneighbors[group] = -1

    else:
        # add newmember-neighbour to neighborlist
        tmp = neighborlist[newmember]
        ind  = numpy.where(tmp > -1)[0]
        if len(ind) > 0:
            tmp = tmp[ind]
            subtmp = numpy.where(exneighbors[tmp] == 0)[0]
            if len(subtmp) > 0:
                tmp = tmp[subtmp]
                exneighbors[tmp] = 1
                neighbors = numpy.append(neighbors, tmp)

        # We already know that the given list of neighbors is unique, since
        # we use the output from the last run. So there is no need to match them
        # again with everything.
        # First check which neighbors of the new members are already part of
        # the group. All group members have an index of -1.
        exneighbors[newmember] = -1
        subneighbors = numpy.where(neighbors != newmember)[0]
        if len(subneighbors) > 0: neighbors = neighbors[subneighbors]
        else: neighbors = []

    return neighbors, exneighbors

def wvt_assign_neighbors(neighborlist, binclass):
# Finds the binnumber of all neighbors of a bin and returns a list of adjacent
# bins. You have to take into account that there might be no neighbors (-1)
    return numpy.append(binclass,[-1])[neighborlist]

def wvt_find_binneighbors(neighborlist, neighborbinclass, binclass, area):
## Produces the final list of *unique* bin neighbors.

    #if len(area) == 0:
        #area = numpy.histogram(binclass, range(0, nbins+1), new=True)[0]
    good = numpy.where(area > 0)[0]    # Check for zero-size Voronoi bins

    binneighbors = {}
    for i in good:
        ind = numpy.where(binclass == i)[0]
        tmplist = numpy.unique(neighborbinclass[ind])
        binneighbors[i] = tmplist[numpy.where(tmplist > -1)]

    return binneighbors

def wvt_recursive_neighbors(binneighbors, current, nlevels):
# Function to increase the speed of constructing the WVT in the next
# iteration. Takes the list of bin neighbors for each single bin and
# recursively finds the next closest neighbors by stepping down "nlevels"
# in the hierarchy. "Current" is the list of bins you would find neighbors for.

    for i in xrange(nlevels+1):
        tmplist = numpy.copy(current)
        for j in current:
            tmplist = numpy.append(tmplist, binneighbors[j])
        current = numpy.unique(tmplist)

    return current

################################################################################
#                                                                              #
#   4.  routines for checking and error prevention                             #
#                                                                              #
################################################################################
def wvt_renumber_binclass(binclass, start_binclass=0, area=[]):
# Kicks out zero pixel bins and renumbers the rest continuously starting at
# the value start_binclass

    if len(area) == 0:
        area = numpy.histogram(binclass, range(start_binclass, (max(binclass)+2)))[0]
    good = numpy.where(area > 0)[0]

    # Only loop over nonzero bins
    for i in xrange(len(good)):
        ind = numpy.where(binclass == good[i] + start_binclass)
        binclass[ind] = i + start_binclass

    return binclass

def wvt_check_all_bins(binclass, x, y, xnode, ynode):
# Sanity checks: make sure you don't have coinciding bin centers (possible
# in WVTs, IF one bin is completely enclosed by another bin of larger scale
# length. In cases with holes in the data, it is generally possible to have
# the bin center outside the valid data and the bin containing zero pixels.

    area = numpy.histogram(binclass, range(0, max(binclass)+2))[0]
    bad = numpy.where(area == 0)[0]    # Check for zero-size Voronoi bins

    while len(bad) > 0:
        # You have to make sure that you don't assign the center of another 1 pixel bin
        good2 = numpy.where(area > 1)[0]
        for n in bad:
            tmp = numpy.argsort((x-xnode[n])**2+(y-ynode[n])**2)
            loop = 0
            while True:
                if numpy.sum(numpy.where(good2 == binclass[tmp[loop]])[0]) > 0:
                    ind = tmp[loop]
                    break
                loop += 1
            # Set the centroid to the center of the pixel
            binclass[ind] = n
            xnode[n] = x[ind]
            ynode[n] = y[ind]
            print 'Bin with zero pixels found: ', n
        if len(bad) != 0:
            area = numpy.histogram(binclass, range(0, max(binclass)+2))[0]
        bad = numpy.where(area == 0)[0]

    return binclass, area

def wvt_check_binislands(x, y, binclass, neighborbinclass, xnode, ynode):
# Checks that every bin is not enclosed by another bin, i.e. has more than
# one neighboring bin. Note that this check should be avoided if you have
# lots of gaps in your data since then you could have an isolated bin due
# to the fact that the gaps are "cornering" the bin.

    nbins = len(xnode)
    area = numpy.histogram(binclass, range(0, max(binclass)+2))[0]
    bad = numpy.where(area == 0)[0]    # Check for zero-size Voronoi bins

    recalc_area = False
    for k in xrange(nbins):
        ind = numpy.where(binclass == k)[0]    # Find subscripts of pixels in bin k.
        neighbornodes = numpy.unique(neighborbinclass[ind])
        w1 = numpy.logical_not(neighbornodes == -1)
        w2 = numpy.logical_not(neighbornodes == k)
        w = numpy.logical_and(w1, w2)
        if numpy.sum(w) >  0: neighbornodes = neighbornodes[w]
        if numpy.sum(w) == 1:
            k2 = neighbornodes[0]
            print 'Bin', k, 'is enclosed by bin', k2, '| Redistributing bins.'
            ind = numpy.append(ind, numpy.where(binclass == k2)[0]) # add the other bin pixels
            tmp = numpy.array([k, k2], dtype=numpy.int)   # only search among 2 nodes
            for i in ind:
                j = wvt_assign_to_bin(x[i], y[i], xnode[tmp], ynode[tmp], 1.)
                binclass[i] = tmp[j]
            recalc_area = True

    if recalc_area == True:
        area = numpy.histogram(binclass, range(0, max(binclass)+2))[0]
    return binclass, area


################################################################################
#                                                                              #
#   5.  routines for the calculation of the binning properties                 #
#                                                                              #
################################################################################
def wvt_calc_binSN(binclass, signal, noise):
# Calculates the S/N values for all bins
# New version, should be considerably faster for large nbin
# Note: binnumber has to start at 0 and end at nbins-1!

    nbins = max(binclass)+1
    binSN = numpy.zeros(nbins, dtype=numpy.float)

    #if len(area) == 0:
    area = numpy.histogram(binclass, range(0, nbins+1))[0]
    good = numpy.where(area > 0)[0]    # Check for zero-size Voronoi bins

    # Only loop over nonzero bins
    for i in good:
        ind = numpy.where(binclass == i)[0]      # Find subscripts of pixels in bin i
        binSN[i] = add_signal(signal[ind])/add_noise(noise[ind])

    return binSN, area

def compute_bin_quantities(x, y, signal, noise, binclass):
# At the end of the computation evaluate the bin luminosity-weighted
# centroids (xbar,ybar) and the corresponding final S/N of each bin.

    nbins = max(binclass)+1

    xbar      = numpy.zeros(nbins, dtype=numpy.float)
    ybar      = numpy.zeros(nbins, dtype=numpy.float)
    binsignal = numpy.zeros(nbins, dtype=numpy.float)
    binnoise  = numpy.zeros(nbins, dtype=numpy.float)
    binSN     = numpy.zeros(nbins, dtype=numpy.float)
    for i in xrange(nbins):
        ind = numpy.where(binclass == i)[0]
        xbar[i], ybar[i] = wvt_weighted_centroid(x[ind], y[ind], signal[ind])
        binsignal[i]     = bin_signal(signal[ind])
        binnoise[i]      = bin_noise(noise[ind])
        binSN[i]         = binsignal[i]/binnoise[i]

    return xbar, ybar, binsignal, binnoise, binSN

def show_binclass(x, y, binclass):
# plots a map of all bins

    size = (max(y)+1, max(x)+1)
    binclass_arr = numpy.zeros(size, dtype=numpy.float)

    for i in xrange(len(x)): binclass_arr[y[i], x[i]] = binclass[i]
    pylab.imshow(binclass_arr, vmin=0, interpolation='nearest', origin='lower',
                 cmap=pylab.get_cmap('gist_ncar_r'))


################################################################################
#                                                                              #
#   6.  basic routines for accretion, reassigning and equalizing of the bins   #
#                                                                              #
################################################################################
def wvt_bin_accretion(x, y, signal, noise, targetSN, dens,
                      pixelSize, center, max_area, quiet, plot):
# Implementation of the bin accretion algorithm. Optimized for speed when
# working with large images. Uses a list of neighboring pixels to search
# only pixels adjacent to the already binned pixels when trying to add
# the next pixel, instead of searching the whole list.
# Based on steps (i)-(v) in section 5.1 of Cappellari & Copin (2003),
# but severely modified

    npix = len(x)
    print 'PixelSize = ', pixelSize

    # Create neighbor list for each pixel
    print '...making neighbor list...'
    neighborlist = wvt_make_neighborlist(x, y)

    binclass = numpy.zeros(npix, dtype=numpy.int)      # will contain bin numberof pixel
    good     = numpy.zeros(npix, dtype=numpy.int)      # will contain 1 if bin has been accepted as goos
    binSN    = numpy.zeros(npix, dtype=numpy.float)    # will contain S/N of the bin

    if (len(center) == 2):
        # start at the specified center, if this keyword is set
        currentBin = [numpy.argmin((y-center[0])**2 + (x-center[1])**2)]
        #print currentBin
        print 'Bin accretion starts at the specified center:'
    else: 
        # start from the pixel with highest S/N
        currentBin = [numpy.argmax(dens)]
        print 'Bin accretion starts at highest S/N value:'

    currentBin = numpy.asarray(currentBin, dtype=numpy.int)
    SN = dens[currentBin]
    print 'First bin starts at X = %i, Y = %i' %(x[currentBin], y[currentBin])

    # start parameters
    xbar_start  = x[currentBin]
    ybar_start  = y[currentBin]
    neighbors   = numpy.zeros((0),dtype=numpy.int)
    exneighbors = numpy.zeros((0),dtype=numpy.int)
    newmember   = 0     # number of the new test 
    goodold     = 0     # 1 if old bin was marked as good
    totarea     = 0     # total area of all binned pixel
    totgoodbins = 0     # total number of good bins

    #if (plot == True): pylab.ion()

    for ind in xrange(1,npix+1):     # The first bin will be assigned BIN = 1

        totgoodbins += goodold

        if (quiet == False) and goodold == 1:
            print r'Bin: %3i | S/N: %5.2f | n_pixels: %3i | %5.2f%% done' \
                   %(totgoodbins, SNold, binareaold, 100.*(1.-float(len(unBinned))/float(npix)))

        # .some starting parameters
        binarea    = 1     # actual area of the bin
        mass       = 0.    # acutal mass of the bin (density * area)

        binclass[currentBin] = ind  # Here currentBin is still made of one pixel
        xbar = x[currentBin]
        ybar = y[currentBin]        # Centroid of one pixel

        while True:

            # Test if the bin is good enough already
            # Better way to decrease the scatter around the target S/N later on:
            # Use the average bin members' S/N to estimate the S/N when another
            # pixel is added to see if adding another pixels would actually
            # increase the scatter around the target S/N due to "overshooting".
            # Also stop accreting if the bin size reached "max_area"
            modtargetSN = targetSN - SN*(numpy.sqrt(1.+1./binarea)-1.)/2.
            if (SN >= modtargetSN) or (binarea >= max_area):
                good[currentBin] = 1
                SNold = numpy.copy(SN)
                binSN[currentBin] = SN
                break

            # Find nearest neighbors of pixels in the current bin, i.e.
            # pixels contiguous with the bin, that have a chance of being accreted
            # For speed, remember the neighbors from the last accretion step and
            # simply add the new ones.
            neighbors, exneighbors = wvt_find_neighbors(currentBin, neighborlist, neighbors, exneighbors, newmember)

            # Accretable neighbors are those that aren't already binned
            #neighbors_ind = numpy.where(neighbors > -1)[0]
            wh = numpy.where(binclass[neighbors] == 0)[0]
            # Stop if there aren't any accretable neighbors
            if len(wh) == 0:
                if SN > 0.8*targetSN:
                    good[ currentBin] = 1
                    binSN[currentBin] = SN
                break
            # Otherwise keep only the accretable ones
            neighbors = neighbors[wh]

            # Search only the neighbors to get the next pixel
            k = numpy.argmin((x[neighbors]-xbar)**2 + (y[neighbors]-ybar)**2)

            # Remember the old verified neighbors and the new members of the bin
            # exneighbors=neighbors
            newmember = neighbors[k]
            nextBin   = numpy.append(currentBin, neighbors[k])
            roundness = wvt_bin_roundness(x[nextBin], y[nextBin], pixelSize)

            # Compute the S/N one would obtain by adding
            # the CANDIDATE pixel to the current bin
            SNOld = numpy.copy(SN)
            SN = add_signal(signal[nextBin])/add_noise(noise[nextBin])

            # Test whether the CANDIDATE pixel is connected to the
            # current bin, whether the POSSIBLE new bin is round enough
            # and whether the resulting S/N would get closer to targetSN
            if roundness > 1.:
                if SN > 0.8*targetSN:
                    good[ currentBin] = 1
                    binSN[currentBin] = SN
                break

            # If all the above tests are negative then accept the CANDIDATE pixel,
            # add it to the current bin, and continue accreting pixels
            binclass[neighbors[k]] = ind
            currentBin = numpy.copy(nextBin)

            # Update the centroid of the current bin
            xbar, ybar, mass = wvt_addto_weighted_centroid(x[newmember], y[newmember], dens[newmember]**2, xbar, ybar, mass)

            binarea += 1    # Update the binarea of the bin

        goodold    = good[currentBin[0]]
        binareaold = numpy.copy(binarea)

        unBinned = numpy.where(binclass == 0)[0]
        if len(unBinned) == 0: break      # Stop if all pixels are binned
        binned = numpy.where(binclass != 0)[0]
        totarea += binareaold

        # Find the closest unbinned pixel to the centroid of all
        # the binned pixels, and start a new bin from that pixel.
        k = numpy.argmin((x[unBinned]-xbar_start)**2 + (y[unBinned]-ybar_start)**2)
        currentBin = numpy.asarray([unBinned[k]],dtype=numpy.int)        # The bin is initially made of one pixel
        SN = add_signal(signal[currentBin])/add_noise(noise[currentBin])

        if plot == True:
            pylab.cla()
            show_binclass(x, y, binclass*good)
            pylab.draw()

    if plot == True:
        #pylab.ioff()
        pylab.gca()

    # Set to zero all bins that did not reach the target S/N
    binclass *= good

    return binclass, neighborlist

def wvt_reassign_bad_bins(x, y, binclass, quiet):
# Find pixels that the bin accretion step wasn't able to assign to a bin and
# reassign them to the next closest bin
# Implements steps (vi)-(vii) in section 5.1 of Cappellari & Copin (2003)

    # Find the centroid of all succesful bins.
    # CLASS = 0 are unbinned pixels which are excluded.
    binclass = wvt_renumber_binclass(binclass, 1)

    area = numpy.histogram(binclass, range(1, max(binclass)+2))[0]
    good = numpy.where(area > 0)[0]     # Obtain the index of the good bins
    xnode = numpy.zeros(len(good), dtype=numpy.float)
    ynode = numpy.zeros(len(good), dtype=numpy.float)
    for i in xrange(len(good)):
        ind = numpy.where(binclass == good[i] + 1)[0]       # Find subscripts of pixels of each bin
        xnode[i], ynode[i] = wvt_unweighted_centroid(x[ind], y[ind])

    if (quiet == False):
        f = pylab.figure()
        pylab.subplot(121)
        show_binclass(x, y, binclass)

    # Reassign pixels to the closest centroid of a good bin
    bad  = numpy.where(binclass == 0)[0]
    for i in bad:
        ind = wvt_assign_to_bin_scale(x[i], y[i], xnode, ynode, 1.)
        binclass[i] = good[ind] + 1

    if (quiet == False):
        pylab.subplot(122)
        show_binclass(x, y, binclass)
        pylab.draw()
        #raw_input()

    # Recompute all centroids of the reassigned bins.
    # These will be used as starting point for the WVT.
    area = numpy.histogram(binclass, range(0,(max(binclass)+2)))[0]
    good = numpy.where(area > 0)[0] # Re-obtain the index of the good bin
    for i in xrange(len(good)):
        ind = numpy.where(binclass == good[i])[0]       # Find subscripts of pixels in each bin
        xnode[i], ynode[i] = wvt_unweighted_centroid(x[ind], y[ind])

    # No more "bad" bins with binnumber=0, i.e. start counting at 0!
    binclass = wvt_renumber_binclass(binclass, 0, area)

    return binclass, xnode, ynode

def wvt_equal_mass(x, y, signal, noise, targetSN, dens, binclass, neighborlist,
                   xnode, ynode, max_area, max_iter, gersho ,quiet, plot):
# Iteration with the new modified Lloyd algorithm that takes advantage of the
# know average S/N per pixel to generate a WVT with equal S/N per bin.
# Procedure described in Diehl & Statler (2005)

    npix  = len(x)
    nbins = len(xnode)

    if max_iter == 0: max_iter = 100   # number of maximal iterations

    # In case of negative data values (e.g bg subtraction), the bin properties
    # can get negative. In this case it is advisable to restrict the size of the
    # bins by max_area
    min_areaSN = float(targetSN)/float(max_area)

    # when gersho conjecture is chosen, compute th VT as described by CC03
    # the density is then defined as the square of the S/N ratio
    if gersho == True: dens2 = dens**2

    #if (plot == True): pylab.ion()

    # Start the iteration!
    iter = 0   # number of iterations
    diff = 1.

    while diff != 0. and iter <= max_iter:

        xnodeold = numpy.copy(xnode)
        ynodeold = numpy.copy(ynode)

        # compute new binSN, area and number of bins
        binSN, area = wvt_calc_binSN(binclass, signal, noise)

        if gersho == False:
            binareaSN = binSN/area
            binoldSN  = numpy.copy(binareaSN)
            binareaSN[numpy.where(binareaSN < min_areaSN)] = min_areaSN
        else:
            # If you use Gersho's conjecture, the scale lengths are all equal,
            # i.e. the WVT reduces to a VT
            binareaSN = numpy.ones(nbins, dtype=numpy.float)

        # Computes (Weighted) Voronoi Tessellation of the pixels grid
        for i in xrange(npix):
            binclass[i]  = wvt_assign_to_bin(x[i], y[i], xnode, ynode, binareaSN)
        neighborbinclass = wvt_assign_neighbors(neighborlist, binclass)

        # Now make sure that still each bin has at least one pixel
        binclass, area = wvt_check_all_bins(binclass, x, y, xnode, ynode)

        # Recompute the new node centers
        good = numpy.where(area > 0)[0] # Check for zero-size Voronoi bins
        for i in good:
            ind = numpy.where(binclass == i)[0]
            if gersho == False:     # get actual positions of the nodes
                xnode[i], ynode[i] = wvt_unweighted_centroid(x[ind], y[ind])
            else:
                xnode[i], ynode[i] = wvt_weighted_centroid(x[ind], y[ind], dens2[ind])

        # Redistribute enclosed bins ("bin islands")
        binclass, area = wvt_check_binislands(x, y, binclass, neighborbinclass, xnode, ynode)

        if plot == True:
            pylab.cla()
            show_binclass(x, y, binclass)
            pylab.draw()

        oldbinclass = numpy.copy(binclass)
        area = numpy.histogram(binclass, range(0, max(binclass)+2))[0]
        diff = numpy.sum((xnode-xnodeold)**2 + (ynode-ynodeold)**2)

        if quiet == False:
            if iter > 0: print  r'Iteration: %3i | Difference: %5.4f %%' %(iter, diff)
            else:        print 'Initial WVT done.'

        iter += 1

    if plot == True:
        pylab.cla()

    if diff == 0.:
        print "Iteration converged to a stable WVT solution"

    if iter > max_iter:
        print 'Iteration reached maximum number of iterations. This should'
        print 'not happen in general except for data with an extremely large'
        print 'dynamical range. Check your input files and/or restart the'
        print 'iteration from this point with the save_all and resume keywords.'

    scale = numpy.sqrt(1./binareaSN)

    # return the generators and scales of the weighted Voronoi bins
    return binclass, xnode, ynode, scale, area


################################################################################
#                                                                              #
#   8.  routines for writing the outputs                                       #
#                                                                              #
################################################################################
def read_fits(filename):
# reads a fits_file and returns header and data

    hdu    = pyfits.open(filename)
    data   = hdu[0].data
    header = hdu[0].header
    hdu.close()

    return header, data

def write_fits(array, header, filename):
# write a numpy-array to a fits file with a header, creates automatic header when header = '0'

    hdu_out = pyfits.PrimaryHDU(array)
    if (header != 0):
        hdu_out.header = header
    os.system('rm -f ' +filename)
    hdu_out.writeto(filename)

def write_binning_results(x, y, signal, noise, outfile_prefix, size, binclass,
                          xnode, ynode, xbar, ybar, binsignal, binnoise, binSN, area, scale,
                          full_output, valid_out, valid_SN, valid_area):
# stores binning results in two asciifiles and four fits-files
# ..._bin.dat:         information of each computed bin: position of generators
#                      and flux-weighted centroids, signal, noise, number of pixels,
#                      scale parameter and distance to the bin with the highest S/N
# ..._pixel.dat:       binNr of each original pixels as well as signal,noise and
#                      distance to the pixel with the highest S/N
# ..._bin_nr.fits:     bin_map: each bin has his binNr as value
# ..._bin_signal.fits: bin_map: each bin has his signal as value
# ..._bin_noise.fits:  bin_map: each bin has his noise as value
# ..._bin_sn.fits:     bin_map: each bin has his S/N as value

    # ascii-output
    if (valid_out == 0) or (valid_out ==2):
        out_ascii = asciidata.create(7, len(x))
        print 'storing binnumbers in ' +outfile_prefix+'.voronoi.pixel.dat'
        out_ascii.header = '#    X      Y   binNr      signal           noise               sn              distance\n'
        max_sn_bin = numpy.argmax(signal/noise)
        for i in xrange(len(x)):
            out_ascii[0][i] = x[i]
            out_ascii[1][i] = y[i]
            out_ascii[2][i] = binclass[i] + 1
            out_ascii[3][i] = signal[i]
            out_ascii[4][i] = noise[i]
            out_ascii[5][i] = signal[i]/noise[i]
            out_ascii[6][i] = numpy.sqrt((x[max_sn_bin]-x[i])**2 + (y[max_sn_bin]-y[i])**2)
        for i in xrange(0,3): out_ascii[i].reformat('%6i')
        for i in xrange(3,5): out_ascii[i].reformat('%16.6e')
        for i in xrange(5,7): out_ascii[i].reformat('%16.6f')
        out_ascii.writeto(outfile_prefix+'.voronoi.pixel.dat')

        out_ascii = asciidata.create(11, len(xnode))
        out_ascii.header = '# binNr         xnode            ynode            xbar             ybar\
            signal           noise               sn        pixNr        scale           distance\n'
        print 'storing bin properties in '+outfile_prefix+'.voronoi.bin.dat'
        max_sn_bin = numpy.argmax(binSN)
        for i in xrange(len(xnode)):
            out_ascii[0][i] = i + 1
            out_ascii[1][i] = xnode[i]
            out_ascii[2][i] = ynode[i]
            out_ascii[3][i] = xbar[i]
            out_ascii[4][i] = ybar[i]
            out_ascii[5][i] = binsignal[i]
            out_ascii[6][i] = binnoise[i]
            out_ascii[7][i] = binSN[i]
            out_ascii[8][i] = area[i]
            out_ascii[9][i] = scale[i]
            out_ascii[10][i] = numpy.sqrt((xbar[max_sn_bin]-xbar[i])**2 + (ybar[max_sn_bin]-ybar[i])**2)
        out_ascii[0].reformat('%6i')
        for i in xrange(1,5):   out_ascii[i].reformat('%16.6f')
        for i in xrange(5,7):   out_ascii[i].reformat('%16.6e')
        out_ascii[7].reformat('%16.6f')
        out_ascii[8].reformat('%5i')
        for i in xrange(9,11): out_ascii[i].reformat('%16.6f')
        out_ascii.writeto(outfile_prefix+'.voronoi.bin.dat')

        if full_output == True:
            # re-ararranging results for output
            signal_arr = numpy.zeros(size, dtype=numpy.float)
            noise_arr  = numpy.zeros(size, dtype=numpy.float)
            bin_arr    = numpy.zeros(size, dtype=numpy.float)-1
            sn_arr     = numpy.zeros(size, dtype=numpy.float)
            for i in xrange(len(x)):
                bin_arr[   y[i], x[i]] = binclass[i] + 1
                signal_arr[y[i], x[i]] = binsignal[binclass[i]]
                noise_arr[ y[i], x[i]] = binnoise[binclass[i]]
                sn_arr[    y[i], x[i]] = binSN[binclass[i]]

            # fits-output
            write_fits(signal_arr, 0, outfile_prefix+'.bin.fits')
            write_fits(noise_arr,  0, outfile_prefix+'.bin.sig.fits')
            write_fits(bin_arr,    0, outfile_prefix+'.bin.nr.fits')
            write_fits(sn_arr,     0, outfile_prefix+'.bin.sn.fits')

    if (valid_out == 1) or (valid_out == 2):
        select_bin = numpy.where(numpy.logical_and(binSN >= valid_SN, area < valid_area) == True)[0]
        select_pix = numpy.zeros(0,dtype=numpy.int)
        for i in select_bin:
            select_pix = numpy.append(select_pix, numpy.where(binclass == i)[0])

        out_ascii = asciidata.create(7, len(select_pix))
        print 'storing valid binnumbers in ' +outfile_prefix+'.voronoi.pixel.valid.dat'
        out_ascii.header = '#    X      Y   binNr      signal           noise               sn              distance\n'
        max_sn_bin = numpy.argmax(signal/noise)
        for i in xrange(len(select_pix)):
            out_ascii[0][i] = x[select_pix[i]]
            out_ascii[1][i] = y[select_pix[i]]
            out_ascii[2][i] = binclass[select_pix[i]] + 1
            out_ascii[3][i] = signal[select_pix[i]]
            out_ascii[4][i] = noise[select_pix[i]]
            out_ascii[5][i] = signal[select_pix[i]]/noise[select_pix[i]]
            out_ascii[6][i] = numpy.sqrt((x[max_sn_bin]-x[select_pix[i]])**2 + (y[max_sn_bin]-y[select_pix[i]])**2)
        for i in xrange(0,3): out_ascii[i].reformat('%6i')
        for i in xrange(3,5): out_ascii[i].reformat('%16.6e')
        for i in xrange(5,7): out_ascii[i].reformat('%16.6f')
        out_ascii.writeto(outfile_prefix+'.voronoi.pixel.valid.dat')

        out_ascii = asciidata.create(11, len(select_bin))
        out_ascii.header = '# binNr         xnode            ynode            xbar             ybar\
            signal           noise               sn        pixNr        scale           distance\n'
        print 'storing valid bin properties in '+outfile_prefix+'.voronoi.bin.valid.dat'
        max_sn_bin = numpy.argmax(binSN)
        for i in xrange(len(select_bin)):
            out_ascii[0][i] = select_bin[i] + 1
            out_ascii[1][i] = xnode[select_bin[i]]
            out_ascii[2][i] = ynode[select_bin[i]]
            out_ascii[3][i] = xbar[select_bin[i]]
            out_ascii[4][i] = ybar[select_bin[i]]
            out_ascii[5][i] = binsignal[select_bin[i]]
            out_ascii[6][i] = binnoise[select_bin[i]]
            out_ascii[7][i] = binSN[select_bin[i]]
            out_ascii[8][i] = area[select_bin[i]]
            out_ascii[9][i] = scale[select_bin[i]]
            out_ascii[10][i] = numpy.sqrt((xbar[max_sn_bin]-xbar[select_bin[i]])**2 + (ybar[max_sn_bin]-ybar[select_bin[i]])**2)
        out_ascii[0].reformat('%6i')
        for i in xrange(1,5):   out_ascii[i].reformat('%16.6f')
        for i in xrange(5,7):   out_ascii[i].reformat('%16.6e')
        out_ascii[7].reformat('%16.6f')
        out_ascii[8].reformat('%5i')
        for i in xrange(9,11): out_ascii[i].reformat('%16.6f')
        out_ascii.writeto(outfile_prefix+'.voronoi.bin.valid.dat')


        if full_output == True:
            # re-ararranging results for output
            signal_arr = numpy.zeros(size, dtype=numpy.float)
            noise_arr  = numpy.zeros(size, dtype=numpy.float)
            bin_arr    = numpy.zeros(size, dtype=numpy.float)-1
            sn_arr     = numpy.zeros(size, dtype=numpy.float)
            for i in xrange(len(select_pix)):
                bin_arr[   y[select_pix[i]], x[select_pix[i]]] = binclass[select_pix[i]] + 1
                signal_arr[y[select_pix[i]], x[select_pix[i]]] = binsignal[binclass[select_pix[i]]]
                noise_arr[ y[select_pix[i]], x[select_pix[i]]] = binnoise[binclass[select_pix[i]]]
                sn_arr[    y[select_pix[i]], x[select_pix[i]]] = binSN[binclass[select_pix[i]]]

            # fits-output
            write_fits(signal_arr, 0, outfile_prefix+'.bin.valid.fits')
            write_fits(noise_arr,  0, outfile_prefix+'.bin.sig.valid.fits')
            write_fits(bin_arr,    0, outfile_prefix+'.bin.nr.valid.fits')
            write_fits(sn_arr,     0, outfile_prefix+'.bin.sn.valid.fits')


################################################################################
#                                                                              #
#   7.  main programs routines                                                 #
#                                                                              #
################################################################################
def wvt_binning(x, y, signal, noise, targetSN , dens,
                pixelSize, center, gersho, max_area, max_iter, quiet, plot,
                valid_SN, valid_area):

    npix = len(x)

    if max_area == 0: max_area = npix   # maximal bin area

    # perform basic tests to catch common input errors
    if (len(y) != npix) or (len(signal) != npix) or (len(noise) != npix):
        print 'Input vectors (x, y, signal, noise) must have the same size'
    if (type(targetSN) != int) and (type(targetSN) != float):
        print 'targetSN must be a scalar'
    if numpy.min(noise) < 0:
        print 'Noise cannot be negative'
    if (add_signal(signal)/add_noise(noise)) < targetSN:
        print ('Not enough S/N in the whole set of pixels.\n'+
               'Many pixels may have noise but virtually no signal.\n'+
               'They should not be included in the set to bin,\n'+
               'or the pixels should be optimally weighted.\n'+
               'See Cappellari & Copin (2003, Sec.2.1) and README file.')
    if numpy.min(signal/noise) > targetSN:
        print 'All pixels have enough S/N and binning is not needed'

    # prevent division by zero for pixels with noise = 0
    noise[numpy.where(noise == 0)] = numpy.min(noise[numpy.where(noise > 0)])*1e-9

    if (len(dens) != npix):
        dens = numpy.zeros(npix, numpy.float)
        for i in range(npix): dens[i] = float(signal[i])/float(noise[i])

    # This is the main routine.
    # It simply calls in sequence the different steps of the algorithms and
    # optionally plots the results at the end of the calculation.
    print 'bin-accretion...'
    binclass, neighborlist = wvt_bin_accretion(x, y, signal, noise, targetSN, dens,
                                               pixelSize, center, max_area, quiet, plot)
    print numpy.max(binclass), ' initial bins.'

    print 'Reassign bad bins...'
    binclass, xnode, ynode = wvt_reassign_bad_bins(x, y, binclass, quiet)
    print len(xnode), ' good bins.'

    print 'Extremely modified Lloyd algorithm...'
    binclass, xnode, ynode, scale, area = \
        wvt_equal_mass(x, y, signal, noise, targetSN, dens, binclass, neighborlist,
                       xnode, ynode, max_area, max_iter, gersho, quiet, plot)

    xbar, ybar, binsignal, binnoise, binSN = compute_bin_quantities(x, y, signal, noise, binclass)

    w1 = area == 1
    w2 = area > 1
    w3 = numpy.logical_and(binSN >= valid_SN, area < valid_area)
    w4 = numpy.logical_and(w2, w3)
    averageSN = numpy.mean(binSN[w4])
    print '----------------------------------------------------'
    print 'Validation thresholds:'
    print '     S/N      = %.2f' %(valid_SN)
    print '     pixelNr  = %i'   %(valid_area)
    print '----------------------------------------------------'
    print 'Binning results:'
    print '     Binned pixels:  %i/%i = %.2f%%' %(npix-len(w1), npix, float(npix-len(w1))/float(npix)*100.)
    print '     Valid bins:     %i/%i = %.2f%%' %(numpy.sum(w4)+numpy.sum(w1), len(xnode), float(numpy.sum(w4)+numpy.sum(w1))/float(len(xnode))*100.)
    print '----------------------------------------------------'
    print 'S/N statistics for valid binned pixel:'
    print '     target S/N:     %.2f'    %(targetSN)
    print '     average S/N:    %.2f'    %(averageSN)
    print '     S/N scatter around target S/N:  %.2f %%' %(numpy.std((binSN[w4]-targetSN)/targetSN)*100.)
    print '     S/N scatter around average S/N: %.2f %%' %(numpy.std((binSN[w4]-averageSN)/averageSN)*100.)
    print '----------------------------------------------------'

    return binclass, x, y, xnode, ynode, xbar, ybar, binsignal, binnoise, binSN, area, scale

def binning_fits(signalfile, noisefile, targetSN, outfile_prefix, full_output=True,
                 pixelSize=1., center=[], dens=[], max_area=0, max_iter=0,
                 noise_mode='var', gersho=False, quiet=False, plot=False,
                 valid_out=0, valid_SN=0, valid_area=0):
# read fits files and calls adaptive binning algorithmus
# ESEnrique: modified to read from function call. 20110420

#    datasignal = read_fits(signalfile)[1]  # ESE commented out
#    datanoise  = read_fits(noisefile)[1]   # ESE commented out
    datasignal = signalfile
    datanoise  = noisefile
    if (noise_mode == 'var'): datanoise = numpy.sqrt(datasignal)

    size   = datasignal.shape
    npix   = numpy.sum(datasignal != 0)
    y      = numpy.zeros(npix, dtype=numpy.int)
    x      = numpy.zeros(npix, dtype=numpy.int)
    signal = numpy.zeros(npix, dtype=numpy.float)
    noise  = numpy.zeros(npix, dtype=numpy.float)

    print npix,'pixel. ', 'Noise mode:',noise_mode

    n = 0
    for i in xrange(size[0]):
        for k in xrange(size[1]):
            if datasignal[i,k]!=0.:
                y[n] = i
                x[n] = k
                signal[n] = datasignal[i,k]
                noise[n]  = datanoise[i,k]
                n += 1

    if valid_SN   == 0: valid_SN   = 0.8*targetSN
    if valid_area == 0: valid_area = 1.2*max_area

#    binclass, xnode, ynode, xbar, ybar, binsignal, binnoise, binSN, area, scale = \
    return wvt_binning(x, y, signal, noise, targetSN, dens,
                    pixelSize, center, gersho, max_area, max_iter, quiet, plot,
                    valid_SN, valid_area)

#    write_binning_results(x, y, signal, noise, outfile_prefix, size, binclass,
#                          xnode, ynode, xbar, ybar, binsignal, binnoise, binSN, area, scale,
#                          full_output, valid_out, valid_SN, valid_area)

