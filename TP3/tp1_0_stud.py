'''
File : tp1_0_stud.py

Creer un signal numerique et l'afficher
RMQ : On utilise ici le B,A,BA de Python.

     Si vous savez faire mieux : objets ?, numpy ? ... SURTOUT lachez vous !!!!!
     car la solution "pythonique" n'est pas là :-(
'''

import math
import matplotlib.pyplot as plt
import random
import numpy as np
import wave, struct
from matplotlib.ticker import MultipleLocator
#=======================================
class Signal:
    def __init__(self, amplitude, frequence, echantillonage, dureeech):
        self.amplitude = amplitude
        self.frequence = frequence
        self.echantillonage = echantillonage
        self.dureeech = dureeech
    
    def xyvalues(self):
        x = np.arange(0, self.dureeech, 1/self.echantillonage)
        y = [self.fonction(t) for t in x]
        return (x, y)

    def fourier(self, nmax):
        x = np.arange(0, self.dureeech, 1/self.echantillonage)
        y = []
        liste_n = range(nmax)
        liste_an = []
        liste_bn = []
        for n in range(nmax):
            liste_an.append(self.an(n))
            liste_bn.append(self.bn(n))
        for k in range(len(x)):
            n=0
            y.append(0)
            while n<nmax:
                y[k] += liste_an[n] * math.cos(2*math.pi*n*self.frequence*x[k])
                y[k] += liste_bn[n] * math.sin(2*math.pi*n*self.frequence*x[k])
                n+=1
        return (x,y, liste_n, liste_an, liste_bn)
'''
class Fourier(Signal):
    def __init__(self, amplitude, frequence, echantillonage, dureeech, an, bn, nmax):
        super().__init__(amplitude, frequence, echantillonage, dureeech)
        self.an = an
        self.bn = bn
        self.nmax = nmax
def fonction(self, t):
        valeur = 0
        n = 0
        while n<self.nmax:
            valeur += self.an(n,self.amplitude) * math.cos(2*math.pi*n*self.frequence*t)
            valeur += self.bn(n,self.amplitude) * math.sin(2*math.pi*n*self.frequence*t) 
            n += 1
        return valeur
    '''

class Sin(Signal):
    def __init__(self, amplitude, frequence, echantillonage, dureeech, dephasage):
        super().__init__(amplitude, frequence, echantillonage, dureeech)
        self.dephasage = dephasage

    def fonction(self, t):
        return self.amplitude*math.sin(((2*math.pi*self.frequence)*t)+self.dephasage)
    
    def label(self):
        return f"Sinusoide: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

    def an(self, n):
        return 0

    def bn(self, n):
        if(n==1) : return 1
        return 0

class Cos(Signal):
    def __init__(self, amplitude, frequence, echantillonage, dureeech, dephasage):
        super().__init__(amplitude, frequence, echantillonage, dureeech)
        self.dephasage = dephasage
    def fonction(self, t):
        return self.amplitude*math.cos(((2*math.pi*self.frequence)*t)+self.dephasage)
    
    def label(self):
        return f"Cosinus: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

    def an(self, n):
        if(n==1) : return 1
        return 0

    def bn(self, n):
        return 0

class Carre(Signal):
    def __init__(self, amplitude, frequence, echantillonage, dureeech, dephasage):
        super().__init__(amplitude, frequence, echantillonage, dureeech)
        self.dephasage = dephasage
        self.legende = f"Carré: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

    def fonction(self, t):
        return self.amplitude*(2*(2*math.floor(self.frequence*t)-math.floor(2*self.frequence*t))+1)

    def an(self, n):
        return 0

    def bn(self, n):
        if(n%2 == 0):
            return 0
        elif(n%2 == 1):
            return 2*self.amplitude*(1/math.pi)*(2)*(1/n)

class Triangle(Signal):
    def __init__(self, amplitude, frequence, echantillonage, dureeech, dephasage):
        super().__init__(amplitude, frequence, echantillonage, dureeech)
        self.dephasage = dephasage
        self.legende = f"Triangle: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

    def fonction(self, t):
        return self.amplitude*(4*(abs(self.frequence*t-math.floor(self.frequence*t + 1/2)))-1)

    def an(self, n):
        if(n%2 == 0):
            return 0
        elif(n%2 == 1):
            return self.amplitude * 8*(math.pi**-2) * (n**-2)

    def bn(self, n):
        return 0

class Dentscie(Signal):
    def __init__(self, amplitude, frequence, echantillonage, dureeech, dephasage):
        super().__init__(amplitude, frequence, echantillonage, dureeech)
        self.dephasage = dephasage
        self.legende = f"Dent de Scie: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

    def fonction(self, t):
        return 2*self.amplitude*(self.frequence*t - math.floor(self.frequence*t)-1/2)

    def an(self, n):
        return 0

    def bn(self, n):
        if(n == 0): return 0
        return -2*self.amplitude*(math.pi**-1)*(n**-1)

class Bruit():
    def __init__(self, x, mu, sigma):
        self.x = x
        self.mu = mu
        self.sigma = sigma
    
    def xyvalues(self):
        y = [self.fonction(t) for t in self.x]
        return (self.x, y)

class Gauss(Bruit):
    def __init__(self, x, mu, sigma):
        super().__init__(x, mu, sigma)
        self.legende = f"Bruit blanc: Moyenne={self.mu}, Ecart-Type={self.sigma}"

    def fonction(self, t):
        return np.random.normal(self.mu, self.sigma)

class Impulsif(Bruit):
    def __init__(self, x, mu, sigma, nbimpuls, dureeimpuls):
        super().__init__(x, mu, sigma)
        self.legende = f"Bruit impulsif: Moyenne={self.mu}, Ecart-Type={self.sigma}"
        self.nbimpuls = nbimpuls
        self.dureeimpuls = dureeimpuls

    def fonction(self, t):
        return np.random.normal(self.mu, self.sigma)
        
    def listebruit(self):    
        listepts = []
        sortie = []
        for k in range(self.nbimpuls):
            listepts.append(int(np.random.uniform(0, len(self.x))))
        for k in range(len(self.x)):
            sortie.append(0)
        for j in listepts:
                valeur = np.random.normal(self.mu, self.sigma)
                sortie[j] = valeur
        return sortie
#=======================================
def decorate_ax(ax, title, loca="lower left"):
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=loca)

def dessinerSinusoides(ListeSinusoides, titre, titrefichier, fig = None, ax= None, separes = 0):
    if(separes == 0 or len(ListeSinusoides) == 1):
        fig, ax = plt.subplots()
        fig.tight_layout()
        for Signal in ListeSinusoides:
            x,y = Signal.xyvalues()
            ax.plot(x, y, Signal.trace, label=Signal.label)
        decorate_ax(ax, titre)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('voltage (V)')
    else:
        fig, axs = plt.subplots(nrows=len(ListeSinusoides))
        fig.tight_layout()
        for k in range(len(ListeSinusoides)):
            Signal = ListeSinusoides[k]
            x,y = Signal.xyvalues()
            axs[k].plot(x, y, Signal.trace, label=Signal.label)
            axs[k].set_xlabel('time (s)')
            axs[k].set_ylabel('voltage (V)')
    plt.savefig(f"./{titrefichier}")
    plt.show()
#=======================================
def conversion_wav(x, y, sortie = "son.wav"):
    nbCanal = 2
    nbOctet = 1
    fe = 44100
    nbEchantillon = len(x)
    wave_file = wave.open(sortie,'w')
    parametres = (nbCanal, nbOctet, fe, nbEchantillon, 'NONE', 'not compressed')
    wave_file.setparams(parametres)
    for i in range(0,nbEchantillon):
        val = y[i]
        if val > 1.0:
            val = 1.0
        elif val < -1.0:
            val = -1.0
        val = int(127.5 + 127.5 * val)
        try:
            fr = struct.pack('BB', val,val)
        except struct.error as err:
            print(err)
            print("Sample {} = {}/{}".format(i,y[i],val))
        wave_file.writeframes(fr)
    wave_file.close()
#=======================================
def tracerSignal(x, y, style, titre, legende):
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(x, y, style, label = legende)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('voltage (V)')
    decorate_ax(ax, titre)
    plt.show()
    
def make_anoisysignal(amp,f,fe,ph,d):
    x,a = calculerSignal(Sinusoide(amp, f, fe, ph, d, "-bo", signal_sin))
    
    m = 0.0
    e = 0.05
    x,b = calculerSignal(Sinusoide(amp, f, fe, ph, d, "-bo", signal_bruit, "gauss", m, e))
    
    m1 = 0.0
    e1 = 1.6*amp
    nbimp = 2
    dureeimp = 2
    x,c = calculerSignal(Sinusoide(amp, f, fe, ph, d, "-bo", signal_bruit, "impulsif", m1, e1, nbimp, dureeimp))
    
    d = []
    for k in range(len(a)):
        d.append(a[k] + b[k] + c[k])
    return x,d

def plot_fft(N, freqStep, Fe, t, s_t, s_f):
    """
    s_f is the fft of s_t(t) on N samples

    """
    #==== Symetrisation et normalisation du spectre du signal
    #cf https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
    s_f = np.fft.fftshift(s_f)      # middles the zero-point's axis
    s_f = s_f/N    # Normalization => ainsi le module ne dependra
                   # de la longueur du signal ou de sa fe
    freq = freqStep * np.arange(-N/2, N/2)  # ticks in frequency domain

    #=== Affichage console des valeurs des raies
    for i,r in enumerate(list(s_f)):
        print("Raie {} \t= \t{:.5g}".format(freq[i],r))

    #==== Plot des spectres 
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=.6)
    # Plot time data ---------------------------------------
    plt.subplot(3,1,1)
    plt.plot(t, s_t, '.-', label="N={}, fe={}".format(N,Fe))
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Signal')
    plt.axis('tight')
    # Plot spectral magnitude ------------------------------
    plt.subplot(3,1,2)
    plt.plot(freq, np.abs(s_f), '.-b', label="freqStep={}".format(freqStep))
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('S(F) Magnitude (Linear)')
    # Plot phase -------------------------------------------
    plt.subplot(3,1,3)
    plt.plot(freq, np.angle(s_f), '.-b')
    plt.grid(True)
    plt.xlabel('Frequency')
    plt.ylabel('S(F) Phase (Radian)')
    plt.show()

def tracerFourier(x, y, n, an, bn, nmax):
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=.6)
    # le signal  ---------------------------------------
    plt.subplot(3,1,1)
    plt.plot(x, y, '.-', label="kmax={}".format(nmax))
    plt.grid(True)
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('s(kTe)')
    plt.title('Signal')
    plt.axis('tight')
    # les an ------------------------------------------
    plt.subplot(3,1,2)
    plt.plot(n, an, 'go')
    plt.grid(True)
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('an de s(kTe)')
    # les bn -------------------------------------------
    plt.subplot(3,1,3)
    plt.plot(n, bn, 'go')
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel('bn de s(kTe)')
    plt.show()
    
def Dictionnaire(Amax, bits):
    step = Amax / (2**bits)
    return np.linspace(-Amax+step, Amax-step, 2**bits)

def QuantifierSignal(signal, bits):
    Amax = np.amax(signal)
    D = Dictionnaire(Amax, bits)
    Q = []
    for k in range(len(signal)):
        Q.append(min(D, key=lambda x:abs(x-signal[k])))
    E = []
    for k in range(len(signal)):
        E.append(abs(signal[k]-Q[k]))
    return Q,E

def MSE(signal, quantification):
    N = len(signal)
    valeur = 0
    for k in range(len(signal)):
        valeur += ((signal[k] - quantification[k])**2)
    return valeur/N

def SNR(signal, quantification):
    variance = np.var(signal)
    return 10*np.log10(variance/MSE(signal, quantification))

def plot(inx, iny, leg, fmt='-bo', l=""):
    plt.plot(inx,iny,fmt,label=l)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (V)')
    plt.ylim([-5.5, +5.5])
    plt.legend()
    
def toBin(nombre, bits):
    val = "{:0{}b}".format(nombre, bits)
    if(nombre < 0):
        val = list(val)
        val[0] = "1"
        val = "".join(val)
    return val
#=======================================
if __name__ == '__main__':
    '''
    Fe = 1000
    N = 256
    d = N/Fe
    S1 = Cos(1, 10, Fe, d, 0)
    a,b = S1.xyvalues()
    #tracerSignal(a,b, "-.b", legende = "test", titre = "")

    S2 = Dentscie(1.5, 750, 8000, 3*1/750, 0)
    nmax = 32
    c,d,n,an,bn = S2.fourier(nmax)
    print(n, an, bn)
    tracerFourier(c,d, n, an, bn, nmax)

    Fe = 1000
    N = 256
    d = N/Fe
    S1 = Cos(1, 10, Fe, d, 0)
    S2 = Cos(1, 20, Fe, d, 0)
    S3 = Cos(1, 400, Fe, d, 0)
    a,b = S1.xyvalues()
    c,d = S2.xyvalues()
    e,f = S3.xyvalues()
    g = []
    for k in range(len(b)):
        g.append(b[k] + d[k] + f[k])

    #tracerSignal(a, g, "-.b", "Avant filtrage", legende = "")
    coef = [-6.849167e-003, 1.949014e-003, 1.309874e-002,1.100677e-002,
                -6.661435e-003,-1.321869e-002, 6.819504e-003, 2.292400e-002,7.732160e-004,
                -3.153488e-002,-1.384843e-002,4.054618e-002,3.841148e-002,-4.790497e-002,
                -8.973017e-002, 5.285565e-002,3.126515e-001, 4.454146e-001,3.126515e-001,
                5.285565e-002,-8.973017e-002,-4.790497e-002, 3.841148e-002, 4.054618e-002,
                -1.384843e-002,-3.153488e-002, 7.732160e-004,2.292400e-002,6.819504e-003,
                -1.321869e-002,-6.661435e-003, 1.100677e-002,1.309874e-002,1.949014e-003,
                -6.849167e-003]
    x = range(len(coef)) 

    Scie = Dentscie(10, 750, 8000, 3*1/750, 0)
    a,b,c,d,e = Scie.fourier(32)
    spectre = np.fft.fft(b)
    freqStep = 8000/32
    plot_fft(32, freqStep, 8000, a, b, spectre)

    #CONVOLUTION
    s = []
    for n in range(len(g)):
        valeur = 0
        for k in range(len(coef)):
            if(n-k > 0):
                valeur += coef[k] * g[n-k]
        s.append(valeur)

    plot_fft(N, freqStep, Fe, a, g, spectre)

    spectre_filtre = np.fft.fft(s)
    plot_fft(N, freqStep, Fe, a, s, spectre_filtre)
    
    a=5.0
    b=3
  
    # Signal à quantifier
    fe = 10000.0
    f = 50.0
    d = 0.04
    
    SignalTest = Dentscie(a, f, fe, d, 0)
    x,y = SignalTest.xyvalues()
    z, err = QuantifierSignal(y, b)
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    fig.tight_layout()
    step = 2*a/(2**b)
    majorLocator = MultipleLocator(step) # Choisir la graduation en y
    ax.yaxis.set_major_locator(majorLocator) 
    plot(x,y,"",'bo', l="Signal")
    plot(x,z,"",'ro', l="Quantized")
    plot(x,err,"",'--x', l="Diff")
    plt.show()   ''' 

    a=5.0
    b=4
  
    # Signal à quantifier
    fe = 10000.0
    f = 50.0
    d = 0.04
    
    SignalTest = Dentscie(a, f, fe, d, 0)
    x,y = SignalTest.xyvalues()
    for k in range(3, 10):
        z, err = QuantifierSignal(y, k)
        print(SNR(y, z))