# harris-corner-detection
features implemented:
- Use the formula for the Y-channel of the YIQ model in performing the color-to-grayscale image conversion.
- Compute Ix and Iy correctly by finite differences.
- Construct images of Ix2, Iy2, and Ix Iy correctly.
- Compute a proper filter size for a Gaussian filter based on its sigma value.
- Construct a proper 1D Gaussian filter.
- Smooth a 2D image by convolving it with two 1D Gaussian filters.
- Handle the image border using partial filters in smoothing.
- Construct an image of the cornerness function R correctly.
- Identify potential corners at local maxima in the image of the cornerness function R.
- Compute the cornerness value and coordinates of the potential corners up to sub-pixel accuracy by quadratic approximation.
- Use the threshold value to identify strong corners for output.
