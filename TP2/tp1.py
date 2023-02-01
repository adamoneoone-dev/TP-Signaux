import math
import matplotlib.pyplot as plt
import random
import numpy as np
import wave, struct
from scipy.io import wavfile
#=======================================
class Sinusoide:
    def __init__(self, amplitude, frequence, echantillonage, dephasage, dureeech, trace, fctsignal, bruit = "", mu=0, sigma=0, nbimpuls=0, dureeimpuls=0):
        self.amplitude = amplitude
        self.frequence = frequence
        self.echantillonage = echantillonage
        self.dephasage = dephasage
        self.dureeech = dureeech
        self.trace = trace
        self.label = f"s1 : a={amplitude}, f={frequence}, fe={echantillonage}, ph={dephasage}, d={dureeech}"
        self.fctsignal = fctsignal
        self.bruit = bruit
        self.mu = mu
        self.sigma = sigma
        self.nbimpuls = nbimpuls
        self.dureeimpuls = dureeimpuls
        
#=======================================
def calculerSignal(Signal):
    if(Signal.fctsignal != signal_bruit):
        x,y=make_signal(Signal.amplitude, Signal.frequence, Signal.echantillonage, Signal.dephasage, Signal.dureeech, Signal.fctsignal)
    elif(Signal.bruit == "gauss"):
        x,y=signal_bruit(Signal.echantillonage, Signal.dureeech, Signal.mu, Signal.sigma)
    elif(Signal.bruit == "impulsif"):
        x,y=signal_bruit(Signal.echantillonage, Signal.dureeech, Signal.mu, Signal.sigma)
        x,y=impulsif(x,y, Signal.nbimpuls, Signal.dureeimpuls)
    return x,y

def decorate_ax(ax, title, loca="lower left"):
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=loca)

def dessinerSinusoides(ListeSinusoides, titre, titrefichier, fig = None, ax= None, separes = 0):
    if(separes == 0 or len(ListeSinusoides) == 1):
        fig, ax = plt.subplots()
        fig.tight_layout()
        for Signal in ListeSinusoides:
            x,y = calculerSignal(Signal)
            ax.plot(x, y, Signal.trace, label=Signal.label)
        decorate_ax(ax, titre)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('voltage (V)')
    else:
        fig, axs = plt.subplots(nrows=len(ListeSinusoides))
        fig.tight_layout()
        for k in range(len(ListeSinusoides)):
            Signal = ListeSinusoides[k]
            x,y = calculerSignal(Signal)
            axs[k].plot(x, y, Signal.trace, label=Signal.label)
            axs[k].set_xlabel('time (s)')
            axs[k].set_ylabel('voltage (V)')
    plt.savefig(f"./{titrefichier}")
    plt.show()

def make_signal(a, f, fe, ph, d, fonction):
    sig_t = np.arange(0, d, 1/fe) #on crée la liste des abscisses, on prend un point à chaque intervalle te = 1/fe, sur la durée d
    sig_s = [fonction(a, f, fe, ph, d, t) for t in sig_t]
    return sig_t, sig_s

def signal_carre(a, f, fe, ph, d, t):
    return 2*(2*math.floor(f*t)-math.floor(2*f*t))+1

def signal_sin(a, f, fe, ph, d, t):
    omega = 2*math.pi*f #pulsation w=2pi*f
    return a*math.sin((omega*t)+ph)

def signal_dentscie(a, f, fe, ph, d, t):
    return 2*a*(f*t - math.floor(f*t)-1/2)

def signal_triangle(a, f, fe, ph, d, t):
    return a*(4*(abs(f*t-math.floor(f*t + 1/2)))-1)

def signal_bruit(fe, d, mu, sigma):
    sig_t = np.arange(0, d, 1/fe)
    sig_s = [np.random.normal(mu, sigma) for t in sig_t]
    return sig_t, sig_s

def impulsif(sig_t, sig_s, nbimpuls = 0, duree = 0):
    intervalle = float(sig_t[1]) # le premier élément de la liste des temps correspond à la durée d'un intervalle
    indexnonzero = random.sample(range(0, len(sig_s)), nbimpuls) #on choisit au hasard les points pour l'impulsion
    final = []
    for k in range(len(sig_s)):
        final.append(0)
        for j in indexnonzero:
            if k == j:
                final[k]=sig_s[k] 
                #si l'index de la liste correspond à un des points parmi ceux choisis pour l'impulsion, on ajoute la vraie valeur
                #sinon, on ajoute 0
    if(duree != 0):
        nbrepet = int(duree/intervalle)
        for j in range(1, nbrepet):
            for l in indexnonzero:
                if(l+j < len(final)):
                    final[l+j] = final[l] 
                    #si on veut une durée plus longue, on calcule le nombre de points qui auront pour valeur celle de l'impulsion
    sig_s = final
    return sig_t, sig_s
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
def partie1():
    Sin = []
    Sin.append(Sinusoide(2.0, 50.0, 1000, 0, 80/1000, "--g*", "sin"))
    dessinerSinusoides(Sin, "Une sinusoide ..", "basic_sin.png")

def partie2():
    Sin = []
    Sin.append(Sinusoide(1, 1/0.02, 1/0.002, 0, 20*0.002, "bo", "sin"))
    Sin.append(Sinusoide(0.5, 1/0.02, 1/0.001, math.pi, 40*0.001, "r.", "sin"))
    dessinerSinusoides(Sin, "Deux sinusoides ..", "basic_sin.png")
    '''
    Les amplitudes sont différentes: 1V pour la grande, 0.5V pour la petite
    Les fréquences sont identiques
    La sinusoide rouge est déphasée de pi radians
    La fréquence d'échantillonage est 2 fois plus importante pour la petite sinusoide    
    
    '''
def carre():
    Sin = []
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-bo", "carre"))    
    dessinerSinusoides(Sin, "Un carré ..", "carre.png")
     
def scie():
    Sin = []
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-bo", "dentscie"))
    dessinerSinusoides(Sin, "Dent de scie", "scie.png")
#impossible d'avoir un triangle parfait puisque cela voudrait dire que pour un t donné, on a deux valeurs

def bg():
    Sin = []
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-go", "bruitgauss", 0, 0.2))
    dessinerSinusoides(Sin, "Dent de scie", "bruitgauss.png")

def bi():
    Sin = []
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-go", "bruitimpuls", 0, 0.2, 20, 0.006))
    dessinerSinusoides(Sin, "Dent de scie", "bruitgauss.png")
    
def tracerSignal(x, y, style):
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(x, y, style)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('voltage (V)')
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


def fourier(A, nmax, f, fe, d, an, bn):
    sig_t = np.arange(0, d, 1/fe) #on crée la liste des abscisses, on prend un point à chaque intervalle te = 1/fe, sur la durée d
    sig_s = []
    ns = []
    ans = []
    bns = []
    for k in range(len(sig_t)):
        n=0
        sig_s.append(0)
        while n<nmax:
            ns.append(n)
            ans.append(an(n,A))
            bns.append(bn(n,A))
            sig_s[k] += an(n,A) * math.cos(2*math.pi*n*f*sig_t[k])
            sig_s[k] += bn(n,A) * math.sin(2*math.pi*n*f*sig_t[k])
            n+=1
        
    return sig_t, sig_s, ns, ans, bns

def an1(n,A):
    if(n%2 == 0):
        return 0
    elif(n%2 == 1):
        return 8*A*(math.pi**-2) * (n**-2)

def bn1(n, A):
    return 0

def an2(n, A):
    return 0

def bn2(n,A):
    if(n == 0): return 0
    return -2*A*(math.pi**-1)*(n**-1)

def an3(n, A):
    return 0

def bn3(n,A):
    if(n%2 == 0):
        return 0
    elif(n%2 == 1):
        return 2*A*(1/math.pi)*(2)*(1/n)


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
#=======================================
if __name__ == '__main__':
    '''
    x,y,z,t,u = fourier(3, 32, 750, 8000, 3*1/750, an2, bn2)
    tracerSignal(x,y,"-bo")

    filename = "La3guitare.wav"
    Fe, s_t = wavfile.read(filename, mmap = False)
    
    N=1024
    s_t = s_t[0: N]
    x = np.arange(N)/Fe
    
    Fe = 8000       # the sampling rate
    Te = 1./Fe       # the sampling period
    freqStep = Fe/N  # resolution of the frequency IN FREQUENCY DOMAIN
    f = 3.0*freqStep   # frequency of the sine 
    
    s_f = np.fft.fft(s_t)
    plot_fft(N, freqStep, Fe, x, s_t, s_f)
    

    Signal = Sinusoide(3, f, Fe, 0, N/Fe, "-bo", signal_dentscie)
    x,y = calculerSignal(Signal)
    
    s_f = np.fft.fft(y)
    
    plot_fft(N, freqStep, Fe, x, y, s_f)
    '''
    Fe = 1000
    N = 256
    Signal1 = Sinusoide(1, 10, Fe, 0, N/Fe, "-bo", signal_sin)
    Signal2 = Sinusoide(1, 20, Fe, 0, N/Fe, "-bo", signal_sin)
    Signal3 = Sinusoide(1, 400, Fe, 0, N/Fe, "-bo", signal_sin)
    x1,y1 = calculerSignal(Signal1)
    x2,y2 = calculerSignal(Signal2)
    x3,y3 = calculerSignal(Signal3)
    x4, y4 = [], []
    for k in range(len(y3)):
        x4.append(x1[k])
        y4.append(y1[k]+y2[k]+y3[k])
        
    
    Te = 1./Fe       # the sampling period
    freqStep = Fe/N  # resolution of the frequency IN FREQUENCY DOMAIN
    f = 3.0*freqStep   # frequency of the sine 
    
    s_f = np.fft.fft(y4)
    plot_fft(N, freqStep, Fe, x4, y4, s_f)
    '''
    coef = [-6.849167e-003, 1.949014e-003, 1.309874e-002,1.100677e-002,
            -6.661435e-003,-1.321869e-002, 6.819504e-003, 2.292400e-002,7.732160e-004,
            -3.153488e-002,-1.384843e-002,4.054618e-002,3.841148e-002,-4.790497e-002,
            -8.973017e-002, 5.285565e-002,3.126515e-001, 4.454146e-001,3.126515e-001,
            5.285565e-002,-8.973017e-002,-4.790497e-002, 3.841148e-002, 4.054618e-002,
            -1.384843e-002,-3.153488e-002, 7.732160e-004,2.292400e-002,6.819504e-003,
            -1.321869e-002,-6.661435e-003, 1.100677e-002,1.309874e-002,1.949014e-003,
            -6.849167e-003] 
    x5,y5 = [], []
    for k in range(len(x4)):
        x5.append(x4[k])
        sg = 0
        for j in range(len(coef)):
            if(j-k >= 0):
                val = y4[j-k]
            else: 
                val = 0
            sg += coef[j] * val
        y5.append(sg)
    
    s_f2 = np.fft.fft(y5)
    plot_fft(N, freqStep, Fe, x4, y5, s_f2)
    '''