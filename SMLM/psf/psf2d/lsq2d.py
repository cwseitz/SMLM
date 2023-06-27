import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
from SMLM.utils.localize import *

def fit_psf(frame,
            spots,
            delta=3,
            diagnostic=True,
            pltshow=True,
            diag_max_dist_err=1,
            diag_max_sig_to_sigraw=3,
            diag_min_slope=0,
            diag_min_mass=0,
            truth_df=None,
            segm_df=None):

    df = pd.DataFrame([], columns=['frame', 'x_raw', 'y_raw', 'r', 'sig_raw',
            'peak', 'mass', 'mean', 'std',
            'A', 'x', 'y', 'sig_x', 'sig_y', 'phi',
            'area', 'dist_err', 'sigx_to_sigraw', 'sigy_to_sigraw', 'slope'])

    df['frame'] = spots['frame'].to_numpy()
    df['x_raw'] = spots['x'].to_numpy()
    df['y_raw'] = spots['y'].to_numpy()
    df['r'] = spots['r'].to_numpy()
    df['sig_raw'] = spots['sigma'].to_numpy()

    good_fitting_num = 0
    for i in df.index:
        x0 = int(df.at[i, 'x_raw'])
        y0 = int(df.at[i, 'y_raw'])
        patch = frame[x0-delta:x0+delta+1, y0-delta:y0+delta+1]

        df.at[i, 'peak'] = patch.max()
        df.at[i, 'mass'] = patch.sum()
        df.at[i, 'mean'] = patch.mean()
        df.at[i, 'std'] = patch.std()

        try:
            p, p_err = fit_gaussian_2d(patch)
            A = p[0]
            x0_refined = x0 - delta + p[1]
            y0_refined = y0 - delta + p[2]
            sig_x = p[3]
            sig_y = p[4]
            phi = p[5]
            sig_raw = df.at[i, 'sig_raw']
            mean = df.at[i, 'mean']
            mass = df.at[i, 'mass']
            slope = (A - mean) / (9 * np.pi * sig_x * sig_y)
            df.at[i, 'A'] = A
            df.at[i, 'x'] = x0_refined
            df.at[i, 'y'] = y0_refined
            df.at[i, 'sig_x'] = sig_x
            df.at[i, 'sig_y'] = sig_y
            df.at[i, 'phi'] = phi
            df.at[i, 'area'] = np.pi * sig_x * sig_y
            # df.at[i, 'mass'] = patch.sum()
            df.at[i, 'dist_err'] = ((x0_refined - x0)**2 + \
                            (y0_refined - y0)**2) ** 0.5
            df.at[i, 'sigx_to_sigraw'] = sig_x / sig_raw
            df.at[i, 'sigy_to_sigraw'] = sig_y / sig_raw
            df.at[i, 'slope'] = slope

            if (x0_refined - x0)**2 + (y0_refined - y0)**2 \
                    < (diag_max_dist_err)**2 \
            and sig_x < sig_raw * diag_max_sig_to_sigraw \
            and sig_y < sig_raw * diag_max_sig_to_sigraw \
            and slope > diag_min_slope \
            and mass > diag_min_mass:
                good_fitting_num = good_fitting_num + 1
            print(f'Sucessfully fit spot {i}')
        except:
            pass

    try:
        print("Predict good fitting number and ratio in frame %d: [%d, %.2f]" %
            (frame.frame_no, good_fitting_num,
            good_fitting_num/len(spots)))
    except:
        pass

    psf_df = df

    plt_array = []
    if diagnostic:
        f1 = df.copy()
        df_filt = pd.DataFrame([], columns=['tot_foci_num'],
                index=['detected', 'fit_success', 'dist_err', 'sigx_to_sigraw',
                        'sigy_to_sigraw'])
        df_filt.loc['detected'] = len(f1)
        f1 = f1.dropna(how='any', subset=['x', 'y'])
        df_filt.loc['fit_success'] = len(f1)
        f1 = f1[ f1['dist_err']<diag_max_dist_err ]
        df_filt.loc['dist_err'] = len(f1)
        f1 = f1[ f1['sigx_to_sigraw']<diag_max_sig_to_sigraw ]
        df_filt.loc['sigx_to_sigraw'] = len(f1)
        f1 = f1[ f1['sigy_to_sigraw']<diag_max_sig_to_sigraw ]
        df_filt.loc['sigy_to_sigraw'] = len(f1)
        f1 = f1[ f1['slope']>diag_min_slope ]

        image = frame
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(image, cmap="gray")

        for i in f1.index:
            Fitting_X = np.indices(image.shape)
            p0,p1,p2,p3,p4,p5 = (f1.at[i,'A'], f1.at[i,'x'], f1.at[i,'y'],
                    f1.at[i,'sig_x'], f1.at[i,'sig_y'], f1.at[i,'phi'])
            Fitting_img = gaussian_2d(Fitting_X,p0,p1,p2,p3,p4,p5)
            contour_img = np.zeros(image.shape)
            x1,y1,r1 = f1.at[i,'x'], f1.at[i,'y'], f1.at[i,'r']
            x1 = int(round(x1))
            y1 = int(round(y1))
            r1 = int(round(r1))
            contour_img[x1-r1:x1+r1+1,
                        y1-r1:y1+r1+1] = Fitting_img[x1-r1:x1+r1+1,
                                                     y1-r1:y1+r1+1]
            ax.contour(contour_img, cmap='cool')

        anno_blob(ax, f1, marker='x', plot_r=1, color=(1,0,0,0.8))

        if isinstance(segm_df, pd.DataFrame):
            anno_scatter(ax, segm_df, marker='^', color=(0,0,1,0.8))

        if isinstance(truth_df, pd.DataFrame):
            anno_scatter(ax, truth_df, marker='o', color=(0,1,0,0.8))

        ax.text(0.95,
                0.00,
                """
                Predict good fitting foci num and ratio: %d, %.2f
                """ %(good_fitting_num, good_fitting_num/len(spots)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 12,
                color = (1, 1, 1, 0.8),
                transform=ax.transAxes)
        plt_array = plot_end(fig, pltshow)

    return psf_df, plt_array

    
def gaussian_2d(X, A, x0, y0, sig_x, sig_y, phi):

    x = X[0]
    y = X[1]
    a = (np.cos(phi)**2)/(2*sig_x**2) + (np.sin(phi)**2)/(2*sig_y**2)
    b = -(np.sin(2*phi))/(4*sig_x**2) + (np.sin(2*phi))/(4*sig_y**2)
    c = (np.sin(phi)**2)/(2*sig_x**2) + (np.cos(phi)**2)/(2*sig_y**2)
    result_array_2d = A*np.exp(-(a*(x-x0)**2+2*b*(x-x0)*(y-y0)+c*(y-y0)**2))

    return result_array_2d

def get_moments(img):

    total = img.sum()
    X, Y = np.indices(img.shape)
    x0 = (X*img).sum()/total
    y0 = (Y*img).sum()/total
    col = img[:, int(y0)]
    sig_x = np.sqrt(np.abs((np.arange(col.size)-y0)**2*col).sum()/col.sum())
    row = img[int(x0), :]
    sig_y = np.sqrt(np.abs((np.arange(row.size)-x0)**2*row).sum()/row.sum())
    A = img.max()
    phi = 0
    params_tuple_1d = A, x0, y0, sig_x, sig_y, phi
    return params_tuple_1d

def fit_gaussian_2d(img, diagnostic=False):

    X = np.indices(img.shape)
    x = np.ravel(X[0])
    y = np.ravel(X[1])
    xdata = np.array([x,y])
    ydata = np.ravel(img)
    p0 = get_moments(img)

    popt, pcov = curve_fit(gaussian_2d, xdata, ydata, p0=p0)
    p_sigma = np.sqrt(np.diag(pcov))
    p_err = p_sigma

    if diagnostic:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        (A, x0, y0, sig_x, sig_y, phi) = popt
        (A_err, x0_err, y0_err, sigma_x_err, sigma_y_err, phi_err) = p_err
        Fitting_data = gaussian_2d(X,A,x0,y0,sig_x,sig_y,phi)
        ax.contour(Fitting_data, cmap='cool')
        ax.text(0.95,
                0.00,
                """
                x0: %.3f (\u00B1%.3f)
                y0: %.3f (\u00B1%.3f)
                sig_x: %.3f (\u00B1%.3f)
                sig_y: %.3f (\u00B1%.3f)
                phi: %.1f (\u00B1%.2f)
                """ %(x0, x0_err,
                      y0, y0_err,
                      sig_x, sigma_x_err,
                      sig_y, sigma_y_err,
                      np.rad2deg(phi), np.rad2deg(phi_err)),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize = 12,
                color = (1, 1, 1, 0.8),
                transform=ax.transAxes)
        plt.show()

    return popt, p_err
