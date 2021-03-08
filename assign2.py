################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from scipy.ndimage import convolve1d
from scipy.ndimage import maximum_filter
# import cv

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image
    rgb_weights=[0.299, 0.587, 0.114]
    img_gray = np.dot(img_color[..., :3], rgb_weights)
    # plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
    # plt.show()
    print(img_gray.shape)
    print(img_gray)
    # TODO: using the Y channel of the YIQ model to perform the conversion

    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result


    # TODO: form a 1D horizontal Guassian filter of an appropriate size

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    # sigma = 1
    n = int(sigma * (2 * np.log(1000)) ** 0.5)
    x = np.arange(-n, n + 1)
    filter = np.exp((x ** 2) / -2 / (sigma ** 2))
    # print(filter)
    # filter /= filter.sum()
    # print(filter)
    img_smoothed = convolve1d(img, filter, 1, np.float64, 'constant', 0, 0)
    one_matrix = np.ones(img.shape)

    one_matrix_weighted= convolve1d(one_matrix,filter, 1, np.float64, 'constant', 0, 0)
    img_smoothed = np.divide(img_smoothed, one_matrix_weighted)
    return img_smoothed


def quadratic(y, x, matrix):
    east = matrix[y, x + 1] if x < matrix.shape[1] - 1 else 0
    south = matrix[y + 1, x] if y < matrix.shape[0] - 1 else 0
    north = matrix[y-1, x] if y > 0 else 0 # matrix[vertical, horizontal]
    west = matrix[y, x-1] if x > 0 else 0
    c = (east - west)/2
    d = (south - north)/2
    e = matrix[y, x]
    dx = -c/2/(((west + east - 2 * matrix[y, x]) / 2)+1e-8)
    dy = -d/2/(((north + south - 2 * matrix[y, x]) / 2)+1e-8)
    R = ((west + east - 2 * matrix[y, x]) / 2)*dx*dx + ((north + south - 2 * matrix[y, x]) / 2)*dy*dy + ((east - west)/2)*dx + ((south - north)/2)*dy + e
    return (dx, dy, R)



################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    img_smoothed=smooth1D(img, sigma)
    img_smoothed=smooth1D(img_smoothed.T, sigma)
    img_smoothed=img_smoothed.T


    # TODO: smooth the image along the horizontal direction

    return img_smoothed

def max_filter(x, y, img):
    for each in range(-1, 2): # consider 8-neighbour only
        for every in range(-1, 2):
            if img[y+every, x+each] > img[y, x]:
                return False
    return True

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy
    # Below code convert image gradient in x direction
    # Ix = np.diff(img, axis=1)
    # Iy = np.diff(img, axis=0)
    Iy, Ix = np.gradient(img)

    # TODO: compute Ix2, Iy2 and IxIy
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = np.multiply(Ix, Iy)

    # TODO: smooth the squared derivatives
    Ix2 = smooth2D(Ix2, sigma)
    Iy2 = smooth2D(Iy2, sigma)
    IxIy = smooth2D(IxIy, sigma)

    # TODO: compute cornesness functoin R
    det = np.multiply(Ix2,Iy2) - np.multiply(IxIy, IxIy)
    trace = np.add(Ix2, Iy2)
    r = det - 0.04*np.square(trace)


    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy
    # allCorners = maximum_filter(r,size=1)
    # plt.imshow(allCorners, cmap=plt.get_cmap("gray"))
    # plt.show()




    # TODO: perform thresholding and discard weak corners
    # print(threshold)
    # print(allCorners)
    # corners = np.where(allCorners > 40.0)
    count=0
    subPixelAccuracy=[]
    h, w = r.shape
    corners = []
    # print(r.size)
    h,w=r.shape
    for y in range(h-1):
        for x in range(w-1):
            if x == 0 or y == 0:
                continue
            if max_filter(x, y, r):
                dx, dy, R = quadratic(y, x, r)
                subPixelAccuracy.append((x + dx, y + dy, R))
    for each in subPixelAccuracy:
        if each[2] >= threshold:
            corners.append(each)
    # print(count)



    # corners=allCorners[allCorners > threshold]

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = np.sqrt(1), help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()
    # img_smoothed = smooth2D(img_gray, args.sigma)
    # plt.imshow(img_smoothed, cmap=plt.get_cmap("gray"))
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
