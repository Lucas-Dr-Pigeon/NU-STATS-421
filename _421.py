# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from math import factorial
from math import exp, log
from math import sqrt
from scipy.stats import gamma
import scipy
import math
def hw3_pr2_b(i0 = 240, i1 = 260):
    Is = np.arange(i0, i1+ 1)
    Cs = np.zeros(i1 - i0 + 1)
    for i, I in enumerate(Is):
        Cs[i] = factorial(500) / ( factorial(500 - I) * factorial(I) )
    return 1 - (0.5) ** 500 * np.sum(Cs)

def hw3_pr3():
    cumps = 0
    for r in [0,1,2,3,4,5,6]:
        cumps += 4**r * exp(-4) / factorial(r)
        
    return 1-cumps

def hw3_pr4(alpha = 4, beta = 6):
    X = np.arange(0, 3, 0.01)
    pdfs = np.zeros(X.shape[0])
    def Gamma(alpha):
        XX = np.arange(0, 30000, 1)
        gs = np.zeros(XX.shape[0])
        for j, xx in enumerate(XX):
            gs[j] = xx**(alpha - 1) * exp(-xx)
        return np.sum(gs)
    G = Gamma(alpha)
    def Gamma_distribution(x, alpha, beta):
        return ((beta ** alpha)) * (x ** (alpha - 1)) * exp(-beta * x) / G
    for i, x in enumerate(X):
        pdfs[i] = Gamma_distribution(x, alpha, beta)
    del i, x
    plt.plot(X, pdfs, color = "black", label = "alpha = 4, beta = 6")
    plt.legend()
    plt.show()
    def get_Mean():
        xfs = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            xfs[i] = x * pdfs[i]
        del i, x
        return np.sum(xfs) * (3 / xfs.shape[0])
    def get_Mode():
        return X[np.argmax(pdfs)]
    def get_Variance():
        mu = get_Mean()
        varis = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            varis[i] = (x - mu)**2 * pdfs[i]
        del i, x
        return np.sum(varis) * (3 / varis.shape[0])
    def get_Skewness():
        mu = get_Mean()
        stdv = sqrt(get_Variance())
        skews = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            skews[i] = ((x - mu)/stdv)**3  * pdfs[i]
        del i, x
        return np.sum(skews) * (3 / skews.shape[0])
    mean = get_Mean()
    mode = get_Mode()
    variance = get_Variance()
    skewness = get_Skewness()
    print ("Mean: " + str(mean) + " Mode: " + str(mode) + " Variance: " + str(variance) + " Skewness: " + str(skewness) )
    return mean, mode, variance, skewness

def hw3_pr5( p_range = tuple([0.05, 0.95]), a_choice = [0.1, 0.5, 0.99] ):
    ps = np.arange(p_range[0], p_range[1], 0.01)
    for a in a_choice:
        Ns = np.zeros(ps.shape[0])
        for i, p in enumerate(ps):
            Ns[i] = log(1 - a) / log(1 - p)
        plt.plot(ps, Ns, label = "a = " + str(a))
    plt.xlabel("p")
    plt.ylabel("Number of Trials")
    plt.legend()
    plt.show()
        
def hw3_pr6_bcd(N = 10000, step = 0.1, M = 25):
    Xs = np.arange(0 - N/2 * step, 0 + N/2 * step, step)
    PDFs = np.zeros(Xs.shape[0])
    XFs = np.zeros(Xs.shape[0])
    Varis = np.zeros(Xs.shape[0])
    for i,x in enumerate(Xs):
        PDFs[i] = exp(-(x + 1)) if x >= -1 else 0
        XFs[i] = x * PDFs[i]
    mu = np.sum(XFs) * step
    print ("The mean is " + str(mu))
    
    for i, x in enumerate(Xs):
        Varis[i] = (x - mu)**2 * PDFs[i]
    var = np.sum(Varis) * step
    print ("The variation is " + str(var))
    plt.plot(Xs, PDFs, color = "blue")
    plt.show()
    
    SMs = np.zeros(1000)
    for trial in range(1000):
        Xi = np.asarray([random.choices(Xs, PDFs) for i in range(25)])
        SMs[trial] = np.mean(Xi)
    print ("The generated 1000 values has mean: " + str(np.mean(SMs)) +" variance: " + str(np.var(SMs)) )
    plt.hist(SMs, color = "grey" , bins = 20, density = True)
    plt.show()
    
def hw4_a():
    print ("Solving problem hw4_a ... ")
    gaussians = np.random.normal(10,1,1000)
    plt.hist(gaussians, color = "grey" , bins = 50, density = True)
    plt.show()
    print ("The estimate mean of the generated gaussian is: " + str(np.mean(gaussians)))
    print ("The standard deviation of the generated gaussian: " + str(np.std(gaussians)))
    
def hw4_b(iterations = 10000):
    print ("Solving problem hw4_b ... ")
    mus = np.zeros(iterations)
    for i in range(iterations):
        mus[i] = np.mean(np.random.normal(1,1,1000))
    plt.hist(mus, color = "grey" , bins = 50, density = True)
    plt.show()
    print ("The mean of the estimate means of gaussians is: " + str(np.mean(mus)))
    print ("The standard deviation of the estimate means of gaussians is: " + str(np.std(mus)))
    
def hw4_c(Ne = 1000, Nv = 1000):
    print ("Solving problem hw4_c ... ")
    mus = np.zeros(Ne)
    varis = np.zeros(Ne)
    for i in range(Ne):
        gaussian = np.random.normal(10,1,Nv)
        mus[i] = np.mean(gaussian)
        varis[i] = np.var(gaussian)
    plt.hist(mus, color = "grey" , bins = 50, density = True)
    plt.title("Distribution of means")
    plt.show()
    plt.hist(varis, color = "grey" , bins = 50, density = True)
    plt.title("Distribution of variances")
    plt.show()
    
    print ("The mean of the estimate means of gaussians is: " + str(np.mean(mus)))
    print ("The standard deviation of the estimate means of gaussians is: " + str(np.std(mus)))
    print ("The mean of the estimate variances of gaussians is: " + str(np.mean(varis)))
    print ("The standard deviation of the estimate variances of gaussians is: " + str(np.std(varis)))
        
def hw4_d():
    print ("Solving problem hw4_d ... ")
    gaussians1 = np.random.normal(10, 1, 860)
    gaussians2 = np.random.normal(10, 7, 140)
    gaussians = np.append(gaussians1, gaussians2)
    plt.hist(gaussians, color = "grey" , bins = 50, density = True)
    plt.xlim(-10, 30)
    plt.show()
    print ("The estimate mean of the generated gaussian is: " + str(np.mean(gaussians)))
    print ("The standard deviation of the generated gaussian: " + str(np.std(gaussians)))

def hw4_e(Ne = 1000):
    print ("Solving problem hw4_e ... ")
    results = np.zeros((12, 4))
    for F in range(12):
        mus = np.zeros(Ne)
        varis = np.zeros(Ne)
        for i in range(Ne):
            gaussians = np.append(np.random.normal(10, 1, 860), np.random.normal(10, 7, 140))
            gaussians = np.sort(gaussians, axis = 0)
            trimmedgaussians = gaussians[ int (F*2/100 * gaussians.shape[0]) : int (gaussians.shape[0] - F*2/100*gaussians.shape[0]) ]
            mus[i] = np.mean(trimmedgaussians)
            varis[i] = np.std(trimmedgaussians)
        print ("Rejection factor " + str(F*2) + "%: " )
        print ("   mean of sample means: " + str(np.mean(mus)) + " st.dev of sample means: " + str(np.std(mus)))
        print ("   mean of sample variances: " + str(np.mean(varis)) + " st.dev of sample variances: " + str(np.std(varis)))
        
        results[F] = np.asarray( [ np.mean(mus), np.std(mus), np.mean(varis), np.std(varis) ] )
    return results
     
class hw5_1e():
    def __init__(self, s = 1*10**4):
        arr_x = np.arange(1/s, 1, 1/s)
        pds = { }
        for x in arr_x:
            pds[x] = 4 * (1 - x) ** 3
        self.pdf = pds
    
    def __Generate_Samples__(self, samplesize = 1000):
        xs = np.asarray(list(self.pdf.keys()))
        pdfs = np.asarray(list(self.pdf.values()))
        samples = np.zeros(0)
        for s in range(samplesize):
            samples = np.append(samples, random.choices(xs, pdfs))
            
        plt.hist(samples, color = "grey" , bins = 50, density = True)
        plt.xlim(0, 1)
        plt.show()
        return samples
    
    def __Estimate__(self, samples):
        import sympy as sy
        from sympy import Symbol
        from sympy.solvers import solve
        N = samples.shape[0]
        return ( - N / np.sum(np.log(1-samples)) )

    
    def __Generate_N_Estimates__(self, iterations = 1000):
        a_list = []
        for it in range(iterations):
            print ("Iteration #" + str(it))
            samples = self.__Generate_Samples__()
            a_list.append(self.__Estimate__(samples))
        return np.asarray(a_list)
    
    def __Plot_N_Estimates__(self, estimates):
        plt.hist(estimates, color = "grey", bins = 50, density = True)
        plt.show()
        print ("The mean of estimates is " + str( np.mean(estimates)))
        print ("The std.deviation of the estimates is " + str(np.std(estimates)))
  
class hw5_pr2():
    def __init__(self):
        pass
    
    '''
    For problem (a)
    '''
    def __Calculate_C__(self, x_range = [0,10], K = 0.4):
        cumu = 0 
        for x in np.arange(x_range[0], x_range[1], 0.01):
            cumu += x* exp(-K * x ** 2)
        print ("C = " + str(1 * (1 / 0.01) / cumu ) )
    '''
    For problem (b)
    '''
    def __Plot_PDF__(self, x_range = [0,4], K = 0.4):
        x_list = np.arange(x_range[0], x_range[1], 0.01)
        pdf_list = 2 * K * x_list * np.exp(-K * x_list **2)
        plt.plot(x_list, pdf_list, color = "black")
        plt.ylabel("Probability Density")
        plt.xlabel("x")
        plt.show()
        
    def __Load_Dat__(self, path = 'MLvalues.dat'):
        data = []
        with open(path, 'r') as f:
            strs = f.readlines()
        for dat in strs:
            data.append(float(dat))
        return np.asarray(data)
    '''
    For problem (d) (e) (g)
    '''
    def __K_vs_NLL__(self, K_range = [0.30, 0.50]):
        Xj = self.__Load_Dat__()
        NLL_list = []
        K_list = np.arange(K_range[0], K_range[1], 0.001)
        N = Xj.shape[0]
        for K in K_list:
            NLL_list.append(-N * log(2 * K) - np.sum(np.log(Xj)) + K * np.sum( Xj ** 2 ))
        plt.plot(K_list, NLL_list, color = "black")
        plt.ylabel("Negative Log Likelihood")
        plt.xlabel("K")
        plt.show()
        
        K_est = np.argmin(NLL_list)
        NLL_min = np.min(NLL_list)
        print ( "The optimal K is " + str (round(K_list[K_est], 3)))
        Ku = np.argmin( np.absolute(NLL_list[K_est:] - NLL_min - 0.5 ))
        print ( "Ku = " + str(round (K_list[K_est:][Ku], 3)))
        Kl = np.argmin( np.absolute(NLL_list[:K_est] - NLL_min - 0.5 ))
        print ( "Kl = " + str(round (K_list[:K_est][Kl], 3)))
        print (" (Ku + Kl)/2 = " + str ( round (K_list[K_est:][Ku] * 0.5 + K_list[:K_est][Kl] * 0.5, 3))) 
        
    '''
    For problem (i)
    '''
    class distribution():
        def __init__(self, xs = np.arange(0, 10, 0.01), K = 0.4):
            pds = { }
            for x in xs:
                pds[x] = 2 * K * x * exp(-K * x ** 2)
            self.pdf = pds
            
        def __Generate_Samples__(self, samplesize = 100):
            xs = np.asarray(list(self.pdf.keys()))
            pdfs = np.asarray(list(self.pdf.values()))
            samples = np.zeros(0)
            for s in range(samplesize):
                samples = np.append(samples, random.choices(xs, pdfs))
            return samples
    
        def __Estimate__(self,samples):
            
            return samples.shape[0] / np.sum(samples ** 2)
        
        
    def __Generate_Dataset_and_Estimate__(self, N = 100, samplesize = 100):
        datasets = np.zeros((N, samplesize))
        estimates = np.zeros(N)
        for it in range(N):
            distri = self.distribution()
            datasets[it] = distri.__Generate_Samples__()
            estimates[it] = distri.__Estimate__(datasets[it])
        print ("The mean of estimates is " + str(np.mean(estimates)))
        print ("The std.deviation of estimates is " + str(np.std(estimates)))
        plt.hist(estimates, color = "grey", bins = 50, density = True)
        plt.show()
    
class hw6:
    def __init__(self):
        pass
    
    def prob1_a(self):
        X = np.arange(3,11,0.1)
        GX = np.zeros(shape = X.shape)
        for i, x in enumerate(X):
            GX[i] = scipy.stats.norm.pdf(x, 7, 1.7)
        plt.plot(X, GX, color = "black")
        plt.plot(np.full(int(0.5/0.001),7),np.arange(0, 0.5, 0.001), linestyle = "dashed", color = "red")
        plt.ylim(0,0.25)
        plt.show()
        
    def prob1_b(self):
        '''
        find the boundary of 68% confidence interval
        '''
        xu = 7
        xl = 7
        c = 0
        step = 0.01
        while c < 0.68:
            c += scipy.stats.norm.pdf(xu, 7, 1.7) * step + scipy.stats.norm.pdf(xl, 7, 1.7) * step
            xu += step
            xl -= step
        print ("68% confidence interval: [" + str(round(xl,2)) + ", " + str(round(xu,2)) + "]")
        
        '''
        find the boundary of 68% confidence interval
        '''
        xu = 7
        xl = 7
        c = 0
        step = 0.01
        while c < 0.95:
            c += scipy.stats.norm.pdf(xu, 7, 1.7) * step + scipy.stats.norm.pdf(xl, 7, 1.7) * step
            xu += step
            xl -= step
        print ("95% confidence interval: [" + str(round(xl,2)) + ", " + str(round(xu,2)) + "]")
        
        
    def prob1_c(self):
        print ("The probability of observing an estimate > 0.97 is:" + str( round ( 1 - scipy.stats.norm.cdf(9.7,7,1.7), 2)))
        
    def prob1_e(self):
        obs = np.random.normal(7, 1.7, 1000)
        res = np.sum(np.logical_and(obs + 1.7 > 7, obs - 1.7 < 7)) / obs.shape[0]
        print ("The fraction of the 1000 intervals containing the true value is: " + str(round(res * 100, 2)) + "%")
        
            
    def prob2(self):
        U = np.random.uniform(0,1,100000)
        X = 1.5 * np.tan((U - 0.5)*math.pi) + 10
        plt.hist(X, color = "green" , edgecolor = 'black', bins = np.arange(-10, 30, 0.5), density = True)
        plt.xlim(-10,30)
        plt.show()
        
    def prob3(self):
        U = np.random.uniform(0,1,100000)
        X = np.random.uniform(0,5,100000)
        PDFx = np.exp(-X) * (np.cos(X)) ** 2
        Comp = np.greater(PDFx, U)
        Accept_X = X[np.where(Comp == True)[0]]
        plt.hist(Accept_X, color = "orange" , edgecolor = 'black', bins = np.arange(0, 5, 0.05), density = True)
        plt.xlim(0,5)
        plt.show()

    def prob4(self):
        Si = np.arange(0, 360, 0.1)
        R =  ( np.cos(Si) ) ** 2
        X = R * np.cos(Si)
        Y = R * np.sin(Si)
        plt.plot(X, Y,  color = "olive", linewidth = 0.6)
        plt.plot(np.arange(-10, 10, 20/1000), np.zeros(1000), linestyle = "dashed", linewidth = 0.5, color = "black")
        plt.plot(np.zeros(1000), np.arange(-10, 10, 20/1000), linestyle = "dashed", linewidth = 0.5, color = "black")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()
        
        RandomX = np.random.uniform(-1, 1, 100000)
        RandomY = np.random.uniform(-1, 1, 100000)
        RandomR = np.sqrt(RandomX ** 2 + RandomY ** 2)
        ActualR = RandomX ** 2 / (RandomX ** 2 + RandomY ** 2)
        
        AcceptX = RandomX[np.where(RandomR < ActualR)[0]]
        AcceptY = RandomY[np.where(RandomR < ActualR)[0]]
        RejectX = RandomX[np.where(RandomR >= ActualR)[0]]
        RejectY = RandomY[np.where(RandomR >= ActualR)[0]]
        area = np.sum(RandomR < ActualR) / RandomX.shape[0] * 4
        plt.plot(X, Y,  color = "olive", linewidth = 0.6)
        plt.plot(np.arange(-10, 10, 20/1000), np.zeros(1000), linestyle = "dashed", linewidth = 0.5, color = "black")
        plt.plot(np.zeros(1000), np.arange(-10, 10, 20/1000), linestyle = "dashed", linewidth = 0.5, color = "black")
        
        plt.scatter(AcceptX, AcceptY, s = 0.1, color = "blue")
        plt.scatter(RejectX, RejectY, s = 0.1, color = "red")
        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()
        
        print ("The area is: " + str (area) )
                
    def prob5(self):
        Measures = np.array([1.34, 1.41, 1.29])
        Sigmas = np.array([0.11, 0.15, 0.04])
        X2 = []
        for d in np.arange(0, 2, 0.001):
            X2.append( np.sum ((Measures - d) ** 2 / Sigmas ** 2  ) )
        X2 = np.asarray(X2)
        plt.plot(np.arange(0, 2, 0.001), X2, color = "black")
        plt.ylim(0, 1300)
        plt.plot(np.full(200,np.arange(0, 2, 0.001)[np.argmin(X2)]), np.arange(0, 1200, 6), linestyle = "dotted", color = "red")
        plt.show()
        print ("The best value for d is: " + str( np.arange(0,2,0.001)[np.argmin(X2)] ))
        print ("The straight value for d is: " + str( np.mean(Measures)) )
        print ("The weighted value for d is: " + str( np.average(Measures, weights = 1/Sigmas ** 2)) )
        
        du_Array = np.zeros(1000)
        dw_Array = np.zeros(1000)
        for it in range(1000):
            di_Array = np.zeros(Measures.shape[0])
            for i in range(Measures.shape[0]):
                di_Array[i] = np.random.normal( Measures[i], Sigmas[i], 1 )[0]
            du_Array[it] = np.mean(di_Array)
            dw_Array[it] = np.average(di_Array, weights = 1/Sigmas ** 2)
        print ("The standard deviation of du is: " + str (np.std(du_Array)))
        print ("The standard deviation of dw is: " + str (np.std(dw_Array)))
        
        
    class samples():
        def __init__(self):
            self.Y = np.array([110.093, 103.484, 96.732, 94.276, 62.247, 57.499, 45.512, 33.824, 25.254, 18.267, 13.776])
            self.Sigmas = np.array([0.228, 0.241, 0.259, 0.265, 0.403, 0.435, 0.543, 0.738, 0.990, 1.329, 2.393])
            self.T = np.array([1.5, 2.5, 3.7, 4.1, 11.2, 12.5, 16.3, 21.5, 26.5, 31.5, 41.5])
    def prob6(self):
        data = self.samples()
        X2 = np.zeros(int(2/0.001))
        for i, r in enumerate(np.arange(16, 18, 0.001)):
            X2[i] = np.sum(
                ( data.Y - 120 *np.exp(-data.T/r) ) ** 2 / data.Sigmas ** 2
                )
        r_opt = np.arange(16,18,0.001)[np.argmin(X2)]
        plt.plot(np.arange(16, 18, 0.001), X2, color = "black")
        plt.plot(np.full(200,r_opt), np.arange(0, 1200, 6), linestyle = "dotted", color = "red")
        plt.xlim(16.5, 17.5)
        plt.ylim(0, 300)
        plt.show()
        print ("The minimal X2 value is: " + str(np.min(X2)))
        print ("The optimal r value is: " + str(r_opt) )
        print ("The 68% confidence interval is: [" + str( 
            round(np.arange(16, 18, 0.001)[:np.argmin(X2)][np.argmin(np.abs(X2[:np.argmin(X2)] - np.min(X2) - 1 ))] ,2)
            ) 
            + "," + str(
            round(np.arange(16, 18, 0.001)[np.argmin(X2):][np.argmin(np.abs(X2[np.argmin(X2):] - np.min(X2) - 1 ))], 2)
                )
            +"]"
            )
        
        plt.errorbar(data.T, data.Y, yerr = data.Sigmas,  fmt='.', capsize=3, color = "black")
        ys = np.zeros( int (45 / 0.01))
        xs = np.arange(0, 45, 0.01)
        plt.plot(xs, 120*np.exp(-xs/r), color = "red", linewidth = 1 )
        plt.show()
        
        Residuals =  data.Y - 120*np.exp(-data.T/r)
        plt.scatter(data.T, Residuals, color = "blue")
        plt.ylabel("Residual")
        plt.ylim(-3.5, 3.5)
        plt.xlim(0, 45)
        plt.xlabel("t")
        plt.plot(data.T, Residuals, color = "black", linestyle = "dashed")
        plt.plot(np.arange(0,45,0.1), np.zeros(450), linestyle = "dotted", color = "red")
        

class hw7():
    def __init__(self):
        pass
    
    def __Fit__(self, X, Y, Sigma, Coeff):
        Ye = np.zeros(Y.shape)
        for i,coef in enumerate(Coeff): 
            Ye += X ** i * coef
        chi_sq = np.sum(( (Y - Ye)/Sigma ) ** 2 )
        df = Y.shape[0] - Coeff.shape[0]
        chi2p = scipy.stats.chi2.pdf(chi_sq, df)
        
        
        plt.plot(np.arange(0, 30, 0.01), scipy.stats.chi2.pdf(np.arange(0, 30, 0.01), df), 
                 color ="red", label = "v = " +str(df))
        plt.plot(np.full(100, chi_sq), np.arange(0, chi2p, chi2p/100), 
                 linestyle = "dotted", color = "black")
        plt.ylim(0, 0.15)
        plt.xlabel("Chi-square")
        plt.ylabel("Probability Density")
        plt.legend()
        # plt.show()
        
        return {"Expected Y": Ye, "Chi2": chi_sq, "Chi2_P":chi2p}
        
    
    def prob1_linear(self):
        import scipy.optimize as opt
        X = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        Y = np.asarray([3.91, 4.61, 4.75, 6.10, 6.91, 6.86, 6.22, 9.55, 9.97, 10.22, 12.23])
        Sigma = np.full(11, 0.6)
        def Linear(x, y0, a):
            return y0 + a*x
        Coeff, Covar = opt.curve_fit(Linear, X, Y, sigma = Sigma)
        print ("***********************************************")
        print ("Outputs of Linear Regression ... ")
        print ("The best-fit values for 'y0' and 'a' are " + str(round(Coeff[0],2)) 
               + ", " + str(round(Coeff[1],2)))
        
        fit = self.__Fit__(X, Y, Sigma, Coeff)
        plt.errorbar(X, Y, Sigma, fmt='.', capsize=3, color = "black")
        plt.plot(np.arange(0, 10.1, 0.01), Linear(np.arange(0, 10.1, 0.01), 
                                                  Coeff[0], Coeff[1]), color = "red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        print ("The Chi-Square value is " + str(round(fit["Chi2"], 2)))
        print ("The degrees of freedom are " + str(Y.shape[0] - Coeff.shape[0] ))
        print ("The Chi-Square probability is " + str(round(fit["Chi2_P"], 2)) )
        Pstd = np.sqrt(np.diag(Covar))
        print ("The probability to get a higher Chi-square is " + str(    
                1 - scipy.stats.chi2.cdf(fit["Chi2"], Y.shape[0] - Coeff.shape[0])
            ) )
        print ("Z-score of differene between 'y0' and 4 is " + str (
               (Coeff[0] - 4)/Pstd[0]
              ) )
        
        
        
    def prob1_quadratic(self):
        import scipy.optimize as opt
        X = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        Y = np.asarray([3.91, 4.61, 4.75, 6.10, 6.91, 6.86, 6.22, 9.55, 9.97, 10.22, 12.23])
        Sigma = np.full(11, 0.6)
        def Quadratic(x, y0, a, b):
            return y0 + a*x + b*x**2
        Coeff, Covar = opt.curve_fit(Quadratic, X, Y, sigma = Sigma)
        print ("***********************************************")
        print ("Outputs of Quadratic Regression ... ")
        print ("The best-fit values for 'y0' 'a' 'b' are " 
               + str(round(Coeff[0],2)) + ", " + str(round(Coeff[1],2)) 
               + ", " + str(round(Coeff[2],2))  )
        fit = self.__Fit__(X, Y, Sigma, Coeff)
        plt.errorbar(X, Y, Sigma, fmt='.', capsize=3, color = "black")
        plt.plot(np.arange(0, 10.1, 0.01), Quadratic(np.arange(0, 10.1, 0.01), Coeff[0]
                                                     , Coeff[1], Coeff[2]), color = "red")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        print ("The Chi-Square value is " + str(round(fit["Chi2"], 2)))
        print ("The degrees of freedom are " + str(Y.shape[0] - Coeff.shape[0] ))
        print ("The Chi-Square probability is " + str(round(fit["Chi2_P"], 2)) )
        Pstd = np.sqrt(np.diag(Covar))
        print ("The uncertainty on b is " + str(Pstd[-1]))
        print ( str(Coeff[-1]/Pstd[-1]) )
        print ("Z-score of differene between 'y0' and 4 is " + str (
               (Coeff[0] - 4)/Pstd[0]
              ) )
        
        
    class pr2():
        def __init__(self, datpath="KNN_data.dat"):
            self.data = np.loadtxt(datpath)
            self.X = self.data[:,1].astype("int")
            self.Y = self.data[:,2].astype("float")
            self.Sigma = self.data[:,3].astype("float")
            
        def TrueFunction(self, x):
            return 100 * scipy.stats.norm.pdf(x, 12, 5) + 25 * scipy.stats.norm.pdf(x, 40, 12)
            
        def _a_(self):
            plt.errorbar(self.X, self.Y, self.Sigma, fmt='.', capsize=3, color = "black")
            plt.plot( np.arange(self.X[0],self.X[-1],0.01), 
                     self.TrueFunction(np.arange(self.X[0],self.X[-1],0.01)), color ="red" )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        
        def KNNRegression(self, X, k=3):
            Ye = np.zeros(X.shape[0])
            for i, x in enumerate(X):
                Ye[i] = np.mean(self.Y[np.argsort(np.absolute(x - self.X))[:k]])
            return Ye
            
        def _b_(self):
            plt.errorbar(self.X, self.Y, self.Sigma, fmt='.', capsize=3, color = "black")
            plt.plot( np.arange(self.X[0],self.X[-1],0.01), 
                     self.KNNRegression(np.arange(self.X[0],self.X[-1],0.01)), color ="red" )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        
        def _c_(self):
            plt.errorbar(self.X, self.Y, self.Sigma, fmt='.', capsize=3, color = "black")
            plt.plot( np.arange(self.X[0]+4,self.X[-1]-4,0.01), 
                     self.KNNRegression(np.arange(self.X[0]+4,self.X[-1]-4,0.01), k=9), color ="red" )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
            
        def _d_(self, K = [3, 9]):
            print ("Solving Problem (d) ... ")
            Index = np.arange(7, 65, 1)
            Xp = self.X[Index]
            Yp = self.Y[Index]
            Yt = self.TrueFunction(Xp)
            Chi2 = np.sum( ( (Yp - Yt ) / 0.2 ) ** 2 )
            print ( "Chi-Square = " + str(Chi2) )
            for k in K:
                Yk = self.KNNRegression(Xp, k)
                MISE = np.sum(( Yk - Yt ) ** 2)
                print ("When K = " + str(k) + ": MISE" + str(k) + " = " + str(MISE) )
                
        def _e_(self, K = [3, 9], _2Indexs = np.asarray([np.arange(7, 25, 1), np.arange(24, 65, 1)] ) ):
            print ("Solving Problem (e) ... ")
            for Index in _2Indexs:
                Xp = self.X[Index]
                Yt = self.TrueFunction(Xp)
                for k in K:
                    Yk = self.KNNRegression(Xp, k)
                    MISE = np.sum(( Yk - Yt ) ** 2)
                    print ("When X is between [" + str(Xp[0]) + "," + str(Xp[-1]) + "] ", end='', flush=True)
                    print ("and K = " + str(k) + ": MISE"+ str(k) + " = " + str(MISE) )
        def WeightedKNNRegression(self, X, k = 3, e = 1):
            Ye = np.zeros(X.shape[0])
            for i, x in enumerate(X):
                KNNX = np.argsort(np.absolute(x - self.X))[:k]
                KNNY = self.Y[KNNX]
                W = 1 / ( (x - KNNX) ** 2 + 1 )
                Ye[i] = np.average(KNNY, weights = W / np.sum(W))
            return Ye
           
        def _f_(self, _2Indexs = np.asarray([np.arange(7, 25, 1), np.arange(24, 65, 1)] )):
            print ("Solving Problem (f) ... ")
            Index = np.arange(7, 65, 1)
            Xp = self.X[Index]
            Yp = self.Y[Index]
            Yt = self.TrueFunction(Xp)
            
            Yg9 = self.WeightedKNNRegression(Xp, 9)
            Yf9 = self.KNNRegression(Xp, 9)
            Yf3 = self.KNNRegression(Xp, 3)
            fig, ax = plt.subplots()
            ax.plot(Xp, Yg9, color = "red", label = "g(x)(k=9)" )
            ax.plot(Xp, Yf9, color = "blue", linestyle = "dashed", label = "f(x)(k=9)")
            ax.plot(Xp, Yf3, color = "green", linestyle = "dashed", label = "f(x)(k=3)")
            ax.errorbar(self.X, self.Y, self.Sigma, fmt='.', capsize=3, color = "black")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            fig.show()
            for Index in _2Indexs:
                Xp = self.X[Index]
                Yp = self.Y[Index]
                Yt = self.TrueFunction(Xp)
                
                Yg9 = self.WeightedKNNRegression(Xp, 9)
                Yf9 = self.KNNRegression(Xp, 9)
                Yf3 = self.KNNRegression(Xp, 3)
                print ("When X is between [" + str(Xp[0]) + "," + str(Xp[-1]) + "] ... ")
                print ("When weighted, MISE9 = " + str( np.sum(( Yg9 - Yt ) ** 2)) )
                print ("When not weighted, MISE9 = " + str( np.sum(( Yf9 - Yt ) ** 2)) )  
                print ("When not weighted, MISE3 = " + str( np.sum(( Yf3 - Yt ) ** 2)) )  
                
    class pr3():
        def __init__(self):
            self.Short = np.asarray([0.33,0.25,0.26,0.26,0.22,0.30,0.20,0.25,0.21,0.27])
            self.Tall = np.asarray([0.32,0.34,0.30,0.40,0.39,0.27,0.39,0.27,0.31,0.36])
            
        def _a_(self):
            print ("The mean of the shorter group is " + str( round(np.mean(self.Short), 2) ) )
            print ("The variance of the shorter group is " + str( round(np.var(self.Short), 6) ) )
            print ("The mean of the taller group is " + str( round(np.mean(self.Tall), 2) ) )
            print ("The variance of the taller group is " + str( round(np.var(self.Tall), 6) ) )
            
        def BinomialPDF(self, n, r, p):
            import operator as op
            from functools import reduce
            r = int (min(r, n-r))
            c = reduce(op.mul, range(n,n-r,-1), 1) / reduce(op.mul, range(1, r+1), 1)
            return c * p**r * (1-p)**(n-r)
            
        def _bc_(self):
            P_list = np.arange(0.01, 1, 0.01)
            NLL_list = np.zeros(P_list.shape)
            for i, p in enumerate(P_list):
                for r in self.Short * 100:
                    NLL_list[i] -= math.log(self.BinomialPDF(100, r, p))
            print ("For the shorter group ... ")
            op1 = P_list[np.argmin(NLL_list)]
            os1 = math.sqrt(op1 * (1 - op1) /1000 )
            print ("The optimal p value is: " + str(op1))
            print ("The sigma value is: " + str(os1))
            
            P_list = np.arange(0.01, 1, 0.01)
            NLL_list = np.zeros(P_list.shape)
            for i, p in enumerate(P_list):
                for r in self.Tall * 100:
                    NLL_list[i] -= math.log(self.BinomialPDF(100, r, p))
            print ("For the taller group ... ")
            op2 = P_list[np.argmin(NLL_list)]
            os2 = math.sqrt(op2 * (1 - op2) /1000)
            print ("The optimal p value is: " + str(op2))
            print ("The sigma value is: " + str(os2))
            self.p1 = op1
            self.p2 = op2
            self.v1 = np.var(self.Short)
            self.v2 = np.var(self.Tall)
            self.z = (self.p2 - self.p1) / math.sqrt( self.v1 + self.v2 )
            print ("The z value of this sample is around " + str(self.z))
            print ("The probability of observing a more extreme z value is " 
                   + str(1 - scipy.stats.norm.cdf(self.z, 0, 1)) )
            
class hw8():
    
    def __init__(self):
        pass
    
    class pr1(): 
        def __init__(self):
            pass    
        
        def _a_(self):
            plt.plot( np.arange(0,250), scipy.stats.norm.pdf(np.arange(0,250),100,math.sqrt(100)),
                     color = "blue", linestyle = "dashed", label = "Team A Background")
            plt.plot( np.arange(0,250), scipy.stats.norm.pdf(np.arange(0,250),100 + 26,math.sqrt(26 + 100)), 
                     color = "blue", label = "Team A Background + Signal")
            plt.xlim(0, 200)
            plt.ylim(0, 0.10)
            plt.legend()
            plt.show()
            plt.plot( np.arange(0,250), scipy.stats.norm.pdf(np.arange(0,250),50,math.sqrt(50)), 
                     color = "red", linestyle = "dashed", label = "Team R Background")
            plt.plot( np.arange(0,250), scipy.stats.norm.pdf(np.arange(0,250),50 + 21,math.sqrt(50 + 21)), 
                     color = "red", label = "Team R Background + Signal")
            plt.xlim(0, 200)
            plt.ylim(0, 0.10)
            plt.legend()
            plt.show()
            
        def _b_(self):
            print ("The Tail Probability in Team-A's Case is: " +
                   str ( 1 - scipy.stats.norm.cdf(  100 + 26, 100, math.sqrt(100) ) )
                       )
            print ("The Tail Probability in Team-R's Case is: " +
                   str ( 1 - scipy.stats.norm.cdf(  50 + 21, 50, math.sqrt(50) ) )
                       )
            
        def _c_(self):
            ''' Team A'''
            Series_n = np.arange(0, 250)
            Series_a = np.asarray(
                [ 1 - scipy.stats.norm.cdf(n, 100, math.sqrt(100) ) for n in Series_n ]
                )
            Series_b = np.asarray(
                [ 1 - scipy.stats.norm.cdf(n, 100 + 26, math.sqrt(100 + 26) ) for n in Series_n ]
                )
            plt.plot(Series_a, Series_b, color = "blue", label = "ROC Team A")
            ''' Team R'''
            Series_n = np.arange(0, 250)
            Series_a = np.asarray(
                [ 1 - scipy.stats.norm.cdf(n, 50, math.sqrt(50) ) for n in Series_n ]
                )
            Series_b = np.asarray(
                [ 1 - scipy.stats.norm.cdf(n, 50 + 21, math.sqrt(50 + 21) ) for n in Series_n ]
                )
            plt.plot(Series_a, Series_b, color = "red", label = "ROC Team R")
            
            plt.plot(np.full(1000, 1.0), np.arange(0, 1, 1/1000), color = "black", linestyle = ":")
            plt.plot(np.arange(0, 1, 1/1000), np.full(1000, 1.0), color = "black", linestyle = ":")
            plt.legend()
            plt.title("ROC Curves")
            plt.xlabel("False positives")
            plt.ylabel("True positives")
            plt.show()

    class pr2():
        
        def __init__(self):
            self.all_dist = np.arange(0, 1000, 0.1)
            self.EcoEdison_dist = self.Bayes_Rule( self.all_dist, Prior = np.full(10000, 1/1000) )   
        
        def _a_(self):
            self.EcoEdison_dist = self.Bayes_Rule( self.all_dist, Prior = np.full(10000, 1/1000) )   
            self.EcoEdison_dist.Get_Posterior( Evidence = scipy.stats.norm.pdf(self.EcoEdison_dist.Events, 540.1, 22.5), count = 0 )
            
        def _b_(self):
            self.EcoEdison_dist = self.Bayes_Rule( self.all_dist, Prior = scipy.stats.norm.pdf(self.EcoEdison_dist.Events, 530, 30) )   
            self.EcoEdison_dist.Get_Posterior( Evidence = scipy.stats.norm.pdf(self.EcoEdison_dist.Events, 540.1, 22.5) , count = 1)
            
        def _c_(self):
            self.EcoEdison_dist = self.Bayes_Rule( self.all_dist, Prior = scipy.stats.norm.pdf(self.EcoEdison_dist.Events, 570, 10) )   
            self.EcoEdison_dist.Get_Posterior( Evidence = scipy.stats.norm.pdf(self.EcoEdison_dist.Events, 540.1, 22.5) , count = 2)
            
        class Bayes_Rule():
            def __init__(self, Events, Prior):
                self.Events = Events
                self.Prior = Prior
                
            def Get_Credible_Interval(self, Events, PD, rate = 0.68, dtype = "Gaussian"):
                if dtype == "Gaussian":
                    mu = np.sum(Events * PD)
                    sigma = math.sqrt(np.sum( (Events - mu)**2 * PD  ))
                    return mu, sigma
            
            def Get_Posterior(self, Evidence, count):
                self.Prior = self.Prior * Evidence / np.sum(self.Prior * Evidence)
                colors = ["blue", "red", "green"]
                labels = ["flat", "friend", "CEO"]
                plt.plot(self.Events, self.Prior, color = colors[count], label = labels[count])
                plt.xlim(0, 1000)
                plt.xlabel("Distance / m")
                plt.ylabel("Probability")
                plt.legend()
                # plt.show()
                mu, sigma = self.Get_Credible_Interval(self.Events, self.Prior)
                print ("The 68% credible interval for D is [" + str(mu - sigma) + "," + str(mu + sigma) + "]")
   
    class pr3():
        def __init__(self):
            self.all_p = np.arange(0, 1, 1/1000)
        
        def _ef_(self, r = 14, n = 75):
            PDF = self.all_p ** r * (1 - self.all_p) ** (n - r) 
            K = 1 / (np.sum(PDF) * 1/1000)
            PDF *= K
            mu = round(np.sum(self.all_p * PDF) / 1000, 3)
            plt.plot(self.all_p, PDF, color = "red", label = "curve e")
            print ("The expected value of E[p] is: " + str(mu)) 
            cl_low, cl_up = self.Find_68_CL(mu, PDF)
            print ("The 68% CL is: [" + str(cl_low) + "," + str(cl_up)+"]")
            pr = self.Compare(0.25, PDF)
            print ("The probability of p > 0.25 is " + str(pr))
            
        def _g_(self, r = 14, n = 75):
            PDF = self.all_p ** (r - 1) * (1 - self.all_p) ** (n - r)
            K = 1 / (np.sum(PDF) * 1/1000)
            PDF *= K
            mu = round(np.sum(self.all_p * PDF) / 1000, 3)
            plt.plot(self.all_p, PDF, color = "blue", label = "curve g")
            print ("The expected value of E[p] is: " + str(mu)) 
            cl_low, cl_up = self.Find_68_CL(mu, PDF)
            print ("The 68% CL is: [" + str(cl_low) + "," + str(cl_up)+"]")
            pr = self.Compare(0.25, PDF)
            print ("The probability of p > 0.25 is " + str(pr))
            
        def _i_(self, r = 24+14, n = 110+75):
            PDF = self.all_p ** r * (1 - self.all_p) ** (n - r) 
            K = 1 / (np.sum(PDF) * 1/1000)
            PDF *= K
            mu = round(np.sum(self.all_p * PDF) / 1000, 3)
            plt.plot(self.all_p, PDF, color = "green", label = "curve i")
            print ("The expected value of E[p] is: " + str(mu)) 
            cl_low, cl_up = self.Find_68_CL(mu, PDF)
            print ("The 68% CL is: [" + str(round(cl_low,3)) + "," + str(round(cl_up,3))+"]")
            pr = self.Compare(0.25, PDF)
            print ("The probability of p > 0.25 is " + str(pr)) 
            plt.legend()
            plt.show()
            
        def Find_68_CL(self, mu, PDF):
            idx_mu = np.where(self.all_p == mu)[0][0]
            cdf = PDF[idx_mu] * 1/1000
            idx_u = idx_mu
            idx_l = idx_mu
            while cdf < 0.68:
                idx_u += 1
                idx_l -= 1
                cdf += PDF[idx_u] * 1/1000
                cdf += PDF[idx_l] * 1/1000
            return self.all_p[idx_l], self.all_p[idx_u]
        
        def Compare(self, v, PDF, mode = ">", power = 0.10):
            idx_v = np.where(self.all_p == v)[0][0]
            if mode == ">":
                return 1 - np.sum(PDF[:idx_v]) * 1/1000
            elif mode == "<":
                return np.sum(PDF[:idx_v]) * 1/1000
            
class hw9:
    def __init__(self):
        
        pass
     
    
    class pr1():
        def __init__(self):
            
            self.B = np.asarray([100, 200, 300, 400, 500])
            self.V = np.asarray([1.00, 1.89, 2.72, 3.81, 5.12])
        
        def _a_(self):
            model = sm.OLS(self.V, self.B)
            result = model.fit()
            print (result.summary())
            _V = result.predict(self.B)
            chi2 = np.sum(((_V - self.V ) / 0.1 ) ** 2 )
            df = 5 - 1
            print ("The degrees of freedom are: " + str(df))
            print ("The Chi-square is: " + str (chi2))
            print ("The Chi-square p-value is: " + str (
                1 - scipy.stats.chi2.cdf(chi2, df)
                ))
            self.a_ = result.params[0]
            self.V_l = _V
            plt.errorbar(self.B, self.V, yerr = np.full(self.V.shape[0], 0.1),  fmt='.', capsize=3, color = "black")
            plt.plot(self.B, _V, color = "red")
            plt.xlabel("B-field")
            plt.ylabel("Voltage")
            plt.show()
            
            
        def _b_(self):
            data = {"B": self.B, "V":self.V}
            model = smf.ols(formula = 'V ~ I(B**2) + B - 1', data = data)
            result = model.fit()
            print (result.summary())
            _V = result.predict(exog=dict(B = self.B))
            chi2 = np.sum(((_V - self.V ) / 0.1 ) ** 2 )
            df = 5 - 2
            print ("The degrees of freedom are: " + str(df))
            print ("The Chi-square is: " + str (chi2))
            print ("The Chi-square p-value is: " + str (
                1 - scipy.stats.chi2.cdf(chi2, df)
                ))
            self.a = result.params[1]
            self.c = result.params[0]
            self.V_q = _V
            plt.errorbar(self.B, self.V, yerr = np.full(self.V.shape[0], 0.1),  fmt='.', capsize=3, color = "black")
            plt.plot(self.B, _V, color = "blue")
            plt.xlabel("B-field")
            plt.ylabel("Voltage")
            plt.show()

        def _c_(self, sigma = 0.1, pi_q = 0.1, pi_l = 0.9 ):
            ln_F = - 1 / ( 2 * sigma ** 2) * np.sum( (self.V - self.V_q) ** 2 - (self.V - self.V_l) ** 2 ) + math.log(pi_q / pi_l)
            print (ln_F)
                   
        def _d_(self, sigma = 0.1, pi_q = 0.5, pi_l = 0.5 ):
            ln_F = - 1 / ( 2 * sigma ** 2) * np.sum( (self.V - self.V_q) ** 2 - (self.V - self.V_l) ** 2 ) + math.log(pi_q / pi_l)
            print (ln_F)
            
            
    class pr2():

        def __init__(self):
            self.data = np.loadtxt("bacteria.dat")
            self.clas = self.data[:,0]
            self.x = self.data[:,1]
            self.y = self.data[:,2]
            pass
        
        def _solve_(self):
            plt.scatter(self.x[np.where(self.clas == 1)], self.y[np.where(self.clas == 1)], color = "blue", label = "type A", s = 0.2)
            plt.scatter(self.x[np.where(self.clas == 2)], self.y[np.where(self.clas == 2)], color = "red", label = "type B", s = 0.2)
            plt.scatter(
                [ np.mean( self.x[np.where(self.clas == 1)] ) ], [np.mean( self.y[np.where(self.clas == 1)] )], color = "black", marker = 'o'
                )
            plt.scatter(
                [ np.mean( self.x[np.where(self.clas == 2)] ) ], [np.mean( self.y[np.where(self.clas == 2)] )], color = "black", marker = 'o'
                )
            plt.annotate('Type B Mean', (np.mean( self.x[np.where(self.clas == 2)] ),np.mean( self.y[np.where(self.clas == 2)] )),fontsize=14)
            plt.annotate('Type A Mean', (np.mean( self.x[np.where(self.clas == 1)] ),np.mean( self.y[np.where(self.clas == 1)] )),fontsize=14)
            plt.arrow(np.mean( self.x[np.where(self.clas == 2)] ), np.mean( self.y[np.where(self.clas == 2)] ), 
                      (np.mean( self.x[np.where(self.clas == 1)] ) - np.mean( self.x[np.where(self.clas == 2)] )) ,
                      (np.mean( self.y[np.where(self.clas == 1)] ) - np.mean( self.y[np.where(self.clas == 2)] )) ,
                      head_width=0, head_length=0, fc='lightblue', ec='black')
            
            plt.xlabel("size")
            plt.ylabel("aspect ratio")
            plt.legend()
            plt.show()

            print ("The sample means are: " + 
                   str([np.mean( self.x[np.where(self.clas == 1)] ), np.mean( self.y[np.where(self.clas == 1)])] )
                   + " and " +
                   str([np.mean( self.x[np.where(self.clas == 2)] ), np.mean( self.y[np.where(self.clas == 2)])] )
                   )
            print ("The variable sigmas for each group are: "+
                    str([np.std( self.x[np.where(self.clas == 1)] ), np.std( self.y[np.where(self.clas == 1)])] )
                   + " and " +
                   str([np.std( self.x[np.where(self.clas == 2)] ), np.std( self.y[np.where(self.clas == 2)])] )
                   )
            print ("The variable covariance for each group are: " +
                   str(
                       np.cov(  self.x[np.where(self.clas == 1)], self.y[np.where(self.clas == 1)]     ) [1][0]
                       ) + " and " + 
                   str(
                       np.cov(  self.x[np.where(self.clas == 2)], self.y[np.where(self.clas == 2)]     ) [1][0]
                       )
                   )
            V_A = np.cov(  self.x[np.where(self.clas == 1)], self.y[np.where(self.clas == 1)]    )
            V_B = np.cov(  self.x[np.where(self.clas == 2)], self.y[np.where(self.clas == 2)]    )
            W = V_A + V_B
            print (str(W))
        
            mA = np.asarray( [np.mean( self.x[np.where(self.clas == 1)] ), np.mean( self.y[np.where(self.clas == 1)])] )
            mB = np.asarray( [np.mean( self.x[np.where(self.clas == 2)] ), np.mean( self.y[np.where(self.clas == 2)])] )
            
            _a = np.dot(np.linalg.inv(W), mA - mB)

            print (_a)
            
            ''' e '''
            xA = self.x[np.where(self.clas == 1)]
            yA = self.y[np.where(self.clas == 1)]
            xB = self.x[np.where(self.clas == 2)]
            yB = self.y[np.where(self.clas == 2)]
            
            tA = _a[0] * xA + _a[1] * yA
            tB = _a[0] * xB + _a[1] * yB   
            plt.hist(tA, density = True, color = "blue", edgecolor = 'black', bins = 40, label = "type A")
            plt.hist(tB, density = True, color = "red", edgecolor = 'black', bins = 50, label = "type B")
            plt.legend()
            plt.show()
            '''f'''
            ''' fisher linear discrminant '''
            FA = np.zeros( int ((18 - 13) / 0.1 ) )
            FB = np.zeros( int ((18 - 13) / 0.1 ) )
            
            for i,_tc in enumerate(np.arange(13, 18, 0.1)):
                try:
                    FA[i] = math.log(np.sum(tA < _tc) / np.shape(tA)[0] )
                    FB[i] = math.log(np.sum(tB > _tc) / np.shape(tB)[0] )
                except ValueError:
                    continue

            
            sort = sorted(zip(FA, FB))
            tuples = zip(*sort)
            FA, FB = [list(tu) for tu in tuples]
            plt.plot (FA, FB, color = "orange", label = "Fisher Linear Discriminant")
            
            ''' simple cut '''
            FA = np.zeros( int ((50 - 20) / 0.1 ) )
            FB = np.zeros( int ((50 - 20) / 0.1 ) )
            for i,_yc in enumerate(np.arange(20, 50, 0.1)):
                try:
                    if math.log(np.sum(yA < _yc) / np.shape(tA)[0] ) > -2.2:
                        continue
                    FA[i] = math.log(np.sum(yA < _yc) / np.shape(tA)[0] )
                    FB[i] = math.log(np.sum(yB > _yc) / np.shape(tB)[0] )
                except ValueError:
                    continue
            FA = np.delete(FA, np.where(FA == 0)[0])
            FB = np.delete(FB, np.where(FA == 0)[0])
            sort = sorted(zip(FA, FB))
            tuples = zip(*sort)
            FA, FB = [list(tu) for tu in tuples]
            plt.plot (FA, FB, color = "green", label = "simple cut")
            plt.legend()
            plt.xlabel("ln(fA)")
            plt.ylabel("ln(fB)")
            plt.show()

    
                   
    class pr3():
        def __init__(self):
            pass
        
        class ANN():
            def __init__(self, A, B, W):
                self.A = np.asarray(A)
                self.B = np.asarray(B)
                self.W = np.asarray(W)
            
            def __forward__(self, x):
                T = self.A + self.B * x 
                Y = 1 / (1 + np.exp(-T))
                t_z = np.sum(self.W[1:] * Y) + self.W[0]
                z = 1 / (1 + math.exp(- t_z))
                return z
            
        def _a_(self):
            X = np.arange(-2, 2, 0.01)
            ann = self.ANN(A = [1.0, -0.5, 0.1], B = [-0.1, 1.0, -10.0], W = [-4, 0.5, 12, 3])
            Z = np.zeros(shape = X.shape)
            for i, x in enumerate(X):
                z = ann.__forward__(x)
                Z[i] = z
            plt.plot(X, Z, color = "red")
            plt.show()
            
        def _b_(self, n_trial = 10):
            for n in range(n_trial):
                A = np.random.uniform(low=-1.0, high=1.0, size=(3,))
                B = np.random.uniform(low=-1.0, high=1.0, size=(3,))
                W = np.random.uniform(low=-3.0, high=3.0, size=(4,))
                X = np.arange(-3, 3, 0.01)
                ann = self.ANN(A, B, W)
                Z = np.zeros(shape = X.shape[0])
                for i, x in enumerate(X):
                    z = ann.__forward__(x)
                    Z[i] = z
                plt.plot(X, Z, label = str(A) + str(B) + str(W))
            plt.show()
            
        def _c_(self, n_trial = int(1e6)):
            L2 = 10e9
            X = np.asarray(
                 [-0.9, 0.7, -0.5, -0.3, 0.0, 0.3, 0.6, 0.8]
                 )
            Y = np.asarray(
                 [0.85, 0.8, 0.7, 0.5, 0.3, 0.35, 0.62, 0.8]
                 )
            for n in range(n_trial):
                # print ("trial #" + str(n))
                A = np.random.uniform(low=-2.0, high=2.0, size=(3,))
                B = np.random.uniform(low=-5.0, high=5.0, size=(3,))
                W = np.random.uniform(low=-20.0,high=20.0,size=(4,))
               
                Z = np.zeros(shape = X.shape[0])
                for i, x in enumerate(X):
                    ann = self.ANN(A, B, W)
                    z = ann.__forward__(x)
                    Z[i] = z 
                new_L2 = np.sum( (Y - Z) **2 )
                if new_L2 < L2:
                    print (L2)
                    L2 = new_L2
                    coef = {"A":A, "B":B, "W":W}
            X_p = np.arange(-1, 1, 0.01)
            Y_p = np.zeros(shape = X_p.shape)
            for i, x in enumerate(X_p):
                z = ann.__forward__(x)
                Y_p[i] = z
            plt.scatter(X, Y, color = "blue")
            plt.plot(X_p, Y_p, color = "red")
            print (coef)
            plt.show()
                
            
class hw10():
    def __init__(self):
        pass
    
    class MCMC():
        def __init__(self, k=20, ux=50, sx=5, uy=30, sy=3, c = 2):
            self.k = k
            self.ux = ux
            self.sx = sx
            self.uy = uy
            self.sy = sy
            self.c = c
            
        def __simulate__(self, n_pts = 1e4, seed = [10,80,80], if_save = False):
            pts = np.asarray([seed])
            while pts.shape[0] < n_pts:
                new_pt = self.__generate_new__(pts[-1])
                if self.__if_accept__(new_pt, pts[-1]):
                    pts = np.append(pts, np.asarray([new_pt]), axis = 0)
                else:
                    continue
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            if if_save:
                fig = plt.figure()
                plt.style.use('seaborn-darkgrid')
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pts[:,0],pts[:,1],pts[:,2], color = "blue", s = 1)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.savefig("_421_hw10pr1a", dpi = 300)
                plt.show()
            return pts
        
        def __generate_new__(self, last_pts):
            import random
            new_pt = np.asarray(
                [   last_pts[0] + random.uniform(-self.c*self.sx, self.c*self.sx),
                    last_pts[1] + random.uniform(-self.c*self.sy, self.c*self.sy),
                    last_pts[2] + random.uniform(-self.c*self.k, self.c*self.k)     ]
                )
            return new_pt
        
        def __f__(self, pt):
            x, y, z = pt
            fac = math.exp(-z/self.k) \
                * math.exp( -0.5*(x-self.ux)**2/self.sx**2 ) \
                    * math.exp( -0.5*(y-self.uy)**2/self.sy**2 )
            return fac
                
        def __if_accept__(self, new_pt, last_pt):
            if new_pt[2] < 0:
                return False
            try:
                a = min(1, self.__f__(new_pt)/self.__f__(last_pt))
            except ZeroDivisionError:
                return False
            r = random.uniform(0, 1)
            if r < a:
                return True
            else:
                return False
            
    def __do_pr1a__(self):
        mcmc_object = self.MCMC()
        data = mcmc_object.__simulate__(if_save = True)
        self.data = data
        self.pr1_done = True
        print(np.mean(self.data[:,0][100:]), np.std(self.data[:,0]))
        print(np.mean(self.data[:,1][100:]), np.std(self.data[:,1]))
        print(np.mean(self.data[:,2][100:]), np.std(self.data[:,2]))
        return data
            
    def __do_pr1b__(self):
        if not self.pr1_done:
            raise ("Please do pr1 first!")
        plt.hist(self.data[:,0], color = "green" , edgecolor = 'black', bins = 80, density = True, label = "X")
        plt.hist(self.data[:,1], color = "orange" , edgecolor = 'black', bins = 80, density = True, label = "Y")
        plt.hist(self.data[:,2], color = "purple" , edgecolor = 'black', bins = 80, density = True, label = "Z")
        plt.xlim(0, 150)
        plt.xlabel("value")
        plt.ylabel("probability density")
        plt.legend()
        plt.savefig("_421_hw10_pr1b", dpi = 300)
        plt.show()
        
    def __do_pr1c__(self, lag = 5):
        if not self.pr1_done:
            raise ("Please do pr1 first!")
        # cov = np.cov(self.data[:,2][:-lag], self.data[:,2][lag:])
        mcmc_object = self.MCMC(c = 2)
        data = mcmc_object.__simulate__()
        datz = data[:,2]
        ch = 1/datz.shape[0] * np.sum( (datz[:-lag] - np.mean(datz)) * (datz[lag:] - np.mean(datz))  )
        c0 = 1/datz.shape[0] * np.sum( (datz - np.mean(datz)) **2   )
        print ("The autocorrelation coefficient is " + str(ch/c0))
        
        mcmc_object = self.MCMC(c = 1/2)
        data = mcmc_object.__simulate__()
        datz = data[:,2]
        ch = 1/datz.shape[0] * np.sum( (datz[:-lag] - np.mean(datz)) * (datz[lag:] - np.mean(datz))  )
        c0 = 1/datz.shape[0] * np.sum( (datz - np.mean(datz)) **2   )
        print ("The autocorrelation coefficient is " + str(ch/c0))
        
        mcmc_object = self.MCMC(c = 4)
        data = mcmc_object.__simulate__()
        datz = data[:,2]
        ch = 1/datz.shape[0] * np.sum( (datz[:-lag] - np.mean(datz)) * (datz[lag:] - np.mean(datz))  )
        c0 = 1/datz.shape[0] * np.sum( (datz - np.mean(datz)) **2   )
        print ("The autocorrelation coefficient is " + str(ch/c0))
            
    class Caunchy():
        def __init__(self, datpath = ".\data\cauchy.dat", r = 1):
            self.xi_list = np.loadtxt(datpath)
            self.r = r 
        
        def __pdf__(self, x_list):
            n = self.xi_list.shape[0]
            pdfs = np.zeros(x_list.shape[0])
            for i, x in enumerate(x_list):
                pdfs[i] = self.r / math.pi * n / (x**2 + self.r**2)
            return pdfs
                
    class KDE():
        def __init__(self, datpath = ".\data\cauchy.dat", s = 1/3):
            self.xi_list = np.loadtxt(datpath)
            self.s = s
            
        def __pdf__(self, x_list):
            n = x_list.shape[0]
            pdfs = np.zeros(x_list.shape[0])
            for i, x in enumerate(x_list):
                pdfs[i] = np.sum(        
                        1/math.sqrt(2*math.pi)/self.s *np.exp( -0.5*(x - self.xi_list)**2 / self.s**2 ) 
                    )
            return pdfs
        
    def __do_pr2a__(self):
        x_list = np.arange(-10, 10, 0.01)
        cauchy = self.Caunchy()
        cau_pdf = cauchy.__pdf__(x_list)
        plt.plot(x_list, cau_pdf, color = "black", label = "Cauchy")
        for si in [1/3, 1, 3]:
            kde = self.KDE(s = si)
            kde_pdf = kde.__pdf__(x_list)
            plt.plot(x_list, kde_pdf, label = "KDE; sigma ="+str(round(si,2)))
        plt.legend()
        plt.savefig("_421_hw10_pr2a", dpi = 300)
        plt.show()
        return cau_pdf
    
    def __do_pr2b__(self):
        x_list = np.arange(-10, 10, 0.01)
        cauchy = self.Caunchy()
        cau_pdf = cauchy.__pdf__(x_list)
        MISE = np.zeros(3)
        for i, si in enumerate([1/3, 1, 3]):
            kde = self.KDE(s = si)
            kde_pdf = kde.__pdf__(x_list)
            diff_2 = ( cau_pdf - kde_pdf ) ** 2
            MISE[i] = np.sum(diff_2) * 0.01
        print (MISE)
        
    def __do_pr2c__(self):
        x_list = np.arange(-10, 10, 0.01)
        cauchy = self.Caunchy()
        cau_pdf = cauchy.__pdf__(x_list)
        MISE = np.zeros(3)
        si_range = np.arange(-0.3, 3, 0.01)
        MISE = np.zeros(si_range.shape[0])
        for i, si in enumerate(si_range):
            kde = self.KDE(s = si)
            kde_pdf = kde.__pdf__(x_list)
            diff_2 = ( cau_pdf - kde_pdf ) ** 2
            MISE[i] = np.sum(diff_2) * 0.01
        plt.plot(si_range, MISE, color = "red")
        plt.xlabel("sigma")
        plt.ylabel("MISE")
        plt.savefig("_421_hw10_pr2c", dpi = 300)
        plt.show()
        
    class Mann_Whitney():
        def __init__(self):
            pass
        
        def __U_Test__(self, sample_X, sample_Y):
            sample_Merge = np.append(sample_X, sample_Y)
            Labels = np.append(np.full(sample_X.shape[0],'X'), np.full(sample_Y.shape[0],'Y'))
            sort = sorted(zip(sample_Merge, Labels))
            tuples = zip(*sort)
            sample_Merge, Labels = np.asarray([list(tu) for tu in tuples])
            Rx = np.sum(np.where(Labels == 'X')[0])
            Ry = np.sum(np.where(Labels == 'Y')[0])
            N = sample_X.shape[0]
            U = N **2 + 1/2 *N *(N+1) - Rx
            mu = 1/2 *N **2
            s = math.sqrt( 1/12 *N **2 *(2*N + 1) )
            z = (U - mu)/s
            return z, U, mu, s, Rx, Ry
            
    def __do_pr3abc__(self):
        self.datA = np.loadtxt("./data/valuesA.dat")
        self.datB = np.loadtxt("./data/valuesB.dat")
        self.datC = np.loadtxt("./data/valuesC.dat")
        mw = self.Mann_Whitney()
        ''' A vs B '''
        zAB, UAB, muAB, sAB, RABx, RABy = mw.__U_Test__(self.datA, self.datB)
        print (zAB, UAB, muAB, sAB, RABx, RABy)
        ''' A vs C '''
        zAC, UAC, muAC, sAC, RACx, RACy = mw.__U_Test__(self.datA, self.datC)
        print (zAC, UAC, muAC, sAC, RACx, RACy)
        ''' B vs C '''
        zBC, UBC, muBC, sBC, RBCx, RBCy = mw.__U_Test__(self.datB, self.datC)
        print (zBC, UBC, muBC, sBC, RBCx, RBCy)
        
        colors = ["red", "blue", "green"]
        labels = ["A", "B", "C"]
        for i, dat in enumerate([self.datA, self.datB, self.datC]):
            for value in dat:
                plt.plot(np.full(100, value), np.arange(0, 0.005, 0.005/100), color = colors[i])
            plt.plot(np.arange(20, 100, 0.1), scipy.stats.norm.pdf(np.arange(20, 100, 0.1), 
                         np.mean(dat), np.std(dat)), color = colors[i], linewidth = 1, linestyle = "--", label = labels[i])
        plt.legend()
        plt.savefig("_421_hw10pr3", dpi = 300)
        plt.show()
        
        
            
            
if __name__ == "__main__":
    
    sol = hw10()
    sol.__do_pr1a__()
    sol.__do_pr1b__()
    sol.__do_pr1c__()
    sol.__do_pr2a__()
    sol.__do_pr2b__()
    sol.__do_pr2c__()
    sol.__do_pr3abc__()
    
    # kde_data = hw10.KDE().data
    
    # import statsmodels.api as sm
    # import statsmodels.formula.api as smf
    
    
    
    
    # from sklearn.linear_model import LinearRegression
    # X = np.asarray([25, 50, 75])
    # sV = np.asarray([20, 15, 17])
    # reg = LinearRegression().fit(X.reshape(-1,1), sV)
    # print (reg.coef_[0])
    
    # sol = hw9()
    # hw9pr1 = sol.pr1()
    # hw9pr1._a_()
    # hw9pr1._b_()
    # hw9pr1._c_()
    # hw9pr1._d_()
    # hw9pr2 = sol.pr2()
    # hw9pr2._solve_()
   
   
    
    # hw9pr3 = sol.pr3()
    # hw9pr3._a_()
    # hw9pr3._b_()
    # hw9pr3._c_()
    
    # sol = hw8()
    # hw8pr1 = sol.pr1()
    # hw8pr1._a_()
    # hw8pr1._b_()
    # hw8pr1._c_()
    # hw8pr2 = sol.pr2()
    
    # hw8pr2._a_()
    # hw8pr2._b_()
    # hw8pr2._c_()
    # hw8pr3 = sol.pr3()
    # hw8pr3._ef_()
    # hw8pr3._g_()
    # hw8pr3._i_()
    # print(
    #     (14/75 - 24/110)/math.sqrt( 14*(75-14)/75**3 + 24*(110-24)/110**3 )
    #     )
    # print (
    #     (1-scipy.stats.norm.cdf(0.527,0,1)) * 2
    #     ) 
    
    # sol = hw7()
    # sol.prob1_linear()
    # sol.prob1_quadratic()
    # sol.pr2()._a_()
    # sol.pr2()._b_()
    # sol.pr2()._c_()
    # sol.pr2()._d_()
    # sol.pr2()._e_()
    # sol.pr2()._f_()
   
    # sol.pr3()._a_()
    # sol.pr3()._bc_()
 
        
    # sol = hw6()
    # sol.prob1_a()
    # sol.prob1_b()
    # sol.prob1_c()
    # sol.prob1_e()
    
    # sol.prob2()
    # sol.prob3()
    # sol.prob4()
    # sol.prob5()
    # sol.prob6()
    
    
    # print (np.random.normal(0, 1, 1)[0])

    
    
    
    
    # sol = hw5_1e()
    # # res = sol.__Generate_N_Estimates__(1000)
    # # sol.__Plot_N_Estimates__(res)
    # sol.__Sample_Std_Dev__()
    
    # sol2 = hw6_pr2()
    # sol2.__Calculate_C__()
    # sol2.__Plot_PDF__()
    # data = sol2.__Load_Dat__()
    # sol2.__K_vs_NLL__()
    # sol2.__Generate_Dataset_and_Estimate__()
    

    # print (np.sum(np.log(1-samples)))
    # ans = hw3_pr3()
    # outcomes = hw3_pr4()
    # hw3_pr5()
    # hw3_pr6_bcd()
    
    
    # hw4_a()
    # hw4_b()
    # hw4_c()
    # hw4_d()
    # hw4e_outcomes = hw4_e()
    
    # from sympy.solvers import solve
    # from sympy import Symbol
    # x = Symbol('x')
    # print (solve(-0.156*x**2 + 50*x - 1001.603))
    
    # print (np.asarray([0,1,2,3,4]) - 1)
    
    
    # print (factorial(500)  / ( factorial(500 - 281) * factorial(281)) *  (0.5)**500 )
    # unif = np.zeros(10000)
    # hists = np.zeros(200)
    # for i in range(10000):
    #    unif[i] = random.uniform(0, 1)
    #    hists[int (unif[i] / 0.05)] += 1
    # mean = np.mean(unif)
    # std = np.std(unif) ** 2
    # plt.hist(unif, color = "grey" , bins = 20, density = True)
    # plt.xlabel("Number")
    # plt.ylabel("Frequency")
    # plt.title("Uniform Distribution")
    # plt.show()

    