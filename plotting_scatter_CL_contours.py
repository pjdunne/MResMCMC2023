import numpy as np
import matplotlib.pyplot as plt

def plot_scatter_CL_contours(x_data, y_data, bins, labels=['x','y']):
    '''
    Plotting scatter plot and confidence level contours with specified data and number of bins

    x_data: 1d np.array
    y_data: 2d np.array
    bins: int
    '''
    # Create 2D histogram with defined bins
    H, xedges1, yedges1 = np.histogram2d(x_data, y_data, bins=20)

    # Find the value corresponding to the 68% highest density region
    sorted_H = np.sort(H.ravel())
    cumulative_H = np.cumsum(sorted_H)
    level1 = sorted_H[np.searchsorted(cumulative_H, 0.32 * cumulative_H[-1])] #68% CL
    level2 = sorted_H[np.searchsorted(cumulative_H, 0.05 * cumulative_H[-1])] #95% CL
    level3 = sorted_H[np.searchsorted(cumulative_H, 0.003 * cumulative_H[-1])] #99.7 CL

    # Plot the contour of the 1-sigma highest density region
    X11, Y11 = np.meshgrid(xedges1[:-1], yedges1[:-1])
    contour = plt.contour(X11, Y11, H.T, colors=['orange','blue','red'],levels=[level3, level2, level1])
    h,_ = contour.legend_elements()

    # Plot the scatter plot of the data
    plt.scatter(x_data, y_data, s=5, color='green', alpha=0.3, label='accepted points')

    plt.legend(h, ["99.7% C.L.", "95% C.L.","68% C.L."], loc="upper left")

    # plt.xlim(4.5,9)#xlimits/ylimits
    # plt.ylim(4.5,9)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('Scatter plot with credible region contours with {} bins'.format(bins))
    plt.show()