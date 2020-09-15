"""
Medical Image Analysis (8DC00)
Project Registration
Project Group 20
Rebecca Küpper (1008070)
Milan Pit (1025441)
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output
from time import time
import sys

## new ##
sys.path.append("../project")
## --- ##

def chooseImage(filenumber, t1t2):
    """
    Chooses the right images according to the input
    
    The inputs are:
    filenumber - indicates number of the image, one of ['1_1', '1_2', '1_3', '2_1', '2_2', '2_3', '3_1', '3_2', '3_3']
    t1t2  -  indicates if you want T1-to-T1 registration or T2-to-T1 registration
    
    The outputs are:
    path_I: the path to the fixed image
    path_Im: the path to the moving image
    """
    
    filelist = ['1_1', '1_2', '1_3', '2_1', '2_2', '2_3', '3_1', '3_2', '3_3']
    
    #Assures that the input is right
    assert (filenumber in filelist), "Invalid input, filenumber has to be in {}".format(filelist) 
    assert (t1t2=='t1' or t1t2=='t2'), "Invalid input, t1t2 has to be 't1' or 't2'"

    
    #Picks images depending on input
    path_I  = '../data/image_data/{}_t1.tif'.format(filenumber)

    if t1t2 == 't1':
        path_Im = '../data/image_data/{}_t1_d.tif'.format(filenumber)
    else:
        path_Im  = '../data/image_data/{}_t2.tif'.format(filenumber)
    
    return path_I, path_Im

def pointBasedRegistration(filenumber='1_1',t1t2='t1'):
    """
    Performes point-based registration on two images
    This can be between T1-to-T1 registration or T2-to-T1 registration
    
    The inputs are:
    filenumber - indicates number of the image, one of ['1_1', '1_2', '1_3', '2_1', '2_2', '2_3', '3_1', '3_2', '3_3']
    t1t2  -  indicates if you want T1-to-T1 registration or T2-to-T1 registration
    
    The outputs are:
    Im_t  - transformed moving image T(Im)
    E_reg - registration error
    """
    
    #Chooses images from given input
    path_I, path_Im = chooseImage(t1t2, filenumber)
    
    
    I = plt.imread(path_I)
    Im = plt.imread(path_Im)

    #Selects points for registration
    X, Xm = util.my_cpselect(path_I, path_Im)

    #Makes transformation matrix for registration depending on selected points and applies to the image
    T = reg.ls_affine(X,Xm)
    Im_t, Xt = reg.image_transform(Im, T)
    
    #Selects points for registration error
    X_ev, X_ev_m = util.my_cpselect(path_I, path_Im)
    
    #Transforms evaluation points of moved image by inverse transformation matrix
    T_inv = np.linalg.inv(T)
    X_ev_h = util.c2h(np.array(X_ev_m))    
    
    #Computes registration error using average distance using Pythagoras
    n = len(X_ev[1])
    dist = 0
        
    X_ev_t = T_inv.dot(X_ev_h)
    for idx in range(0, n-1):
        dist = dist + np.sqrt((X_ev[0][idx] - X_ev_t[0][idx])**2 + (X_ev[1][idx] - X_ev_t[1][idx])**2)
    
    E_reg = dist / n
    
    print(E_reg)
    
    return Im_t, E_reg

def intensityBasedRegistration(affine=True, corr=True, iterations=250, mu=1e-3, t1t2='t1', filenumber='1_1'):
    """
    This function is an application of intensity based image registration.
    It uses three available methods of intensity based registration:
    rigid correlation, affine correlation and affine mutual information.
    These functions calculate similarity between the two input images, which is used to register the images.

    The inputs are:
    affine (default=True): A boolean that determines whether the affine or rigid method is used.
        True means the affine method is used, False means the rigid method is used.
    corr (default=True): A boolean that determines whether the similarity is calculated using correlation or mutual information.
        True means that correlation is used, False mean that mutual information is used. If affine=False, correlation will automatically be used.
    iterations (default=250): An integer that determines the amount of times the gradient ascent is updated.
    mu (default=1e-3): A float that determines the learning rate of the gradient ascent.

    The output is:
    A single image containing:
        The final registration; The parameters of the registration; The similarity curve of the two images.

    An example of a correct function call:
    intensityBasedRegistration(True, True, 50, 1e-2)
    """

    #Sanitizes input
    iterations = int(iterations)
    
    # Choose images from given input
    path_I, path_Im = chooseImage(t1t2, filenumber)
    
    
    I = plt.imread(path_I)
    Im = plt.imread(path_Im)

    #Sets initial parameters and function based on input
    if(affine):
        x = np.array([0., 1., 1., 0., 0., 0., 0.])
        if(corr):
            fun = lambda x: reg.affine_corr(I, Im, x)
        else:
            fun = lambda x: reg.affine_mi(I, Im, x)
    else:
        x = np.array([0., 0., 0.])
        fun = lambda x: reg.rigid_corr(I, Im, x)

    similarity = np.full((iterations, 1), np.nan)

    fig = plt.figure(figsize=(20,10))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    
    #Shows parameters in image
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    #Sets up similarity curve
    ax2 = fig.add_subplot(122, xlim=(0, iterations), ylim=(0, 1))

    learning_curve, = ax2.plot(range(1,iterations+1), similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity (%s)' %("Correlation"*(corr) + "Mutual Information"*(1-corr)))
    ax2.grid()

    #Logging steps are calculated. Cannot be done easier,
    #as it is not guaranteed that the amount of iterations is evenly divisible by 4
    step1 = int(iterations/4-1)
    step2 = int(iterations/2-1)
    step3 = int(iterations*3/4-1)

    #Stores start time of gradient ascent
    start_time = time()
    
    #Applies gradient descent [iterations] times
    for k in np.arange(iterations):
            
        #Gradient is calculated and applied to the parameters
        g = reg.ngradient(fun, x)
        x += g*mu

        #Calls similarity function to calculate the similarity and transformed image
        S, Im_t, _ = fun(x)

        #Logs time elapsed and estimated total time of the gradient ascent
        print("Iteration {:d}/{:d}, {:.2f}% done".format(k+1, iterations, (k+1)/iterations * 100))
        
        if(k == 0 or k == step1 or k == step2 or k == step3):
            print("Elapsed time: {:.1f} s\nEstimated time: {:.1f} s".format(
                time()-start_time, (time()-start_time) * (iterations/(k+1))))
            
        elif(k+1==iterations):
            print("Duration: {:.2f} s".format(time()-start_time))
            
        #Updates moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        #Updates similarity curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

    #Logs end result of similarity
    print("Final similarity: %f" %(S))

    #Shows final image and plot (required for non-jupyter python)
    plt.show()        

if(__name__ == "__main__"):
    #Test example of function

    intensityBasedRegistration(True, False, 250, 9e-5, '3_3', 't2')
    #pointBasedRegistration('t1','1_2')
