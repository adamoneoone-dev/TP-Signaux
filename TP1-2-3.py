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
from scipy.io import wavfile
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

    def fonction(self, t):
        return self.amplitude*(2*(2*math.floor(self.frequence*t)-math.floor(2*self.frequence*t))+1)

    def label(self):
        return f"Carré: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

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

    def fonction(self, t):
        return self.amplitude*(4*(abs(self.frequence*t-math.floor(self.frequence*t + 1/2)))-1)

    def label(self):
        return f"Triangle: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

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

    def fonction(self, t):
        return 2*self.amplitude*(self.frequence*t - math.floor(self.frequence*t)-1/2)

    def label(self):
        return f"Dent de Scie: a={self.amplitude}, f={self.frequence}, ph={self.dephasage}, ech={self.echantillonage}, duree={self.dureeech}"

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
                for l in range(self.dureeimpuls):
                    if(j+l < len(sortie)):
                        sortie[j+l] = valeur
        return sortie
#=======================================
def plot_on_ax(ax, inx, iny, label, format='-bo'):
    ax.plot(inx,iny,format,label=label)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('voltage (V)')

def decorate_ax(ax, title, loca="lower left"):
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=loca)
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
def make_anoisysignal(a,f,fe,ph,d):
	Sig1 = Sin(a, f, fe, d, ph)
	x1,y1=Sig1.xyvalues()

	m = 0.0
	e = 0.05
	Br1 = Gauss(x1, m, e)
	x2,y2 = Br1.xyvalues()

	m1 = 0.0 # mean
	e1 = 1.6*a # ecart type
	nbi = 2 # nombre d’impulsions sur le signal
	di = 1000 # duree d’une impulsion en sample
	Br2 = Impulsif(x1, m1, e1, nbi, di)

	x3,y3 = x2, Br2.listebruit()

	x4, y4 = x3, []
	for k in range(len(x3)):
		y4.append(y1[k] + y2[k] + y3[k])
	return x4, y4
#=======================================
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
            ans.append(an(n))
            bns.append(bn(n))
            sig_s[k] += an(n) * math.cos(2*math.pi*n*f*sig_t[k])
            sig_s[k] += bn(n) * math.sin(2*math.pi*n*f*sig_t[k])
            n+=1
    return sig_t, sig_s, ns, ans, bns

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

def convolution(g, coef):
	s = []
	for n in range(len(g)):
		valeur = 0
		for k in range(len(coef)):
			if(n-k > 0):
				valeur += coef[k] * g[n-k]
		s.append(valeur)
	return s

def Dictionnaire(Amax, bits):
    step = 2*Amax / (2**bits)
    print(f"Step : {step}")
    dico = np.arange(-Amax+step/2, Amax-step/2 + step, step)
    dicopt = [(-2**bits)/2 + k for k in range(2**bits)]
    return dico, dicopt 

def QuantifierSignal(signal, bits):
    Amax = np.amax(signal)
    dico, dicopt = Dictionnaire(Amax, bits)
    Q = []
    Qpt = []
    for k in range(len(signal)):
        val = min(dico, key=lambda x:abs(x-signal[k]))
        Q.append(val)
        for v in range(len(dico)):
            if(dico[v] == val):
                Qpt.append(int(dicopt[v]))
    E = []
    for k in range(len(signal)):
        E.append(abs(signal[k]-Q[k]))
    print(f"MSE: {MSE(signal, Q)}")
    print(f"SNR: {SNR(signal, Q)}")
    
    print(f"Dictionnaire :{dico}")				#dictionnaire sous la forme card(dictionnaire)/step
    print(f"Quantification :{Q}")				#quantification 
    print(f"Dictionnaire points:{dicopt}")		#dictionnaire sous forme valeur entiere
    print(f"Quantification points:{Qpt}")		#quantification en points entiers
    return Q,Qpt,E,dico,dicopt

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

def convertirBinaire(quantificationpts, bits):  # ['0', '1', '2',...] > ['0000', '0001', '0010'] ex pour 4 bits (3+1 de signe)
    converti = []
    for k in range(len(quantificationpts)):
        converti.append(toBin(quantificationpts[k], bits+1))
    return converti

def eclaterConversion(quantif): # ['1001'...] > ['1', '0', '0', '1', ...]
    ec = []
    for k in range(len(quantif)):
        for j in range(len(quantif[0])):
            ec.append(int(quantif[k][j]))
    return ec

def MLT3(signal_bits, V):
    liste = []
    # valeur0 = 0
    # valeur1 = 0
    prochain_negatif = False
    liste.append(0)
    for k in range(1, len(signal_bits)):
        if(signal_bits[k] == 0): # si on recoit un 0, mettre la valeur précédente
            liste.append(liste[k-1])
        if(signal_bits[k] == 1): # si on recoit un 1
            if(liste[k-1] == +V):
                prochain_negatif = True
                liste.append(0)
            if(liste[k-1] == -V):
                prochain_negatif = False
                liste.append(0)
            elif(liste[k-1] == 0): #si la valeur précédente est 0, 
                if(prochain_negatif):
                    liste.append(-V)
                elif(not prochain_negatif):
                    liste.append(+V)
    return liste
#=======================================
if __name__ == '__main__':
	# a=2
	# f=1/0.02
	# fe=20/0.02
	# ph=0
	# d=0.08

	# Sig1 = Sin(a, f, fe, d, ph)
	# x,y=Sig.xyvalues()

	# # Representation graphique
	# fig,ax = plt.subplots()
	# plot_on_ax(ax,x,y,Sig1.label())
	# decorate_ax(ax,"Une sinusoide")

	# plt.savefig("./basic_sin.png")
	# plt.show()
	# ======================= PARTIE 2 ======================= 
	# a1=1.5
	# a2=0.5
	# f=1/0.02
	# fe1=10/0.02
	# fe2=20/0.02
	# ph=0
	# d=0.08

	# Sig1 = Sin(a1, f, fe1, d, ph)
	# Sig2 = Sin(a2, f, fe2, d, ph)
	# x,y=Sig1.xyvalues()
	# z,t=Sig2.xyvalues()

	# # Representation graphique
	# fig,ax = plt.subplots()
	# plot_on_ax(ax,x,y,Sig1.label())
	# plot_on_ax(ax,z,t,Sig2.label())
	# decorate_ax(ax,"Une sinusoide")

	# plt.savefig("./basic_sin.png")
	# plt.show()
	# ======================= PARTIE 3 ======================= 
	# a=3
	# f=50
	# fe=800
	# ph=0
	# d=0.08

	# Sig1 = Triangle(a, f, fe, d, ph)
	# x,y=Sig1.xyvalues()

	# # Representation graphique
	# fig,ax = plt.subplots()
	# plot_on_ax(ax,x,y,Sig1.label())
	# decorate_ax(ax,"Une sinusoide")

	# plt.savefig("./basic_sin.png")
	# plt.show()
	# ======================= PARTIE BRUIT ======================= 
	# a=3
	# f=50
	# fe=1000
	# ph=0
	# d=0.08

	# Sig1 = Sin(a, f, fe, d, ph)
	# x1,y1=Sig1.xyvalues()

	# m = 0.0
	# e = 0.2
	# Br1 = Gauss(x1, m, e)
	# x2,y2 = Br1.xyvalues()

	# m1 = 0.0 # mean
	# e1 = 2*a # ecart type
	# nbi = 3 # nombre d’impulsions sur le signal
	# di = 4 # duree d’une impulsion en sample
	# Br2 = Impulsif(x1, m1, e1, nbi, di)
	# x3,y3 = x2, Br2.listebruit()

	# x4, y4 = x3, []
	# for k in range(len(x3)):
	# 	y4.append(y1[k] + y2[k] + y3[k])
	# # Representation graphique
	# fig,ax = plt.subplots(4)
	# plot_on_ax(ax[0],x1,y1,"s1 : {} Hz, {} V".format(f,a),'bo-')
	# decorate_ax(ax[0],"")
	# plot_on_ax(ax[1],x2,y2,"Bruit blanc : Moyenne {}, Ecart Type {}".format(m,e),'g.-')
	# decorate_ax(ax[1],"")
	# plot_on_ax(ax[2],x3,y3,"Bruit impulsif : Moyenne {}, Ecart Type {}".format(m1,e1),'g.-')
	# decorate_ax(ax[2],"")
	# plot_on_ax(ax[3],x4,y4,"s1 bruite",'r.-')
	# decorate_ax(ax[3],"")

	# plt.savefig("./basic_sin.png")
	# plt.show()
	# ======================= PARTIE SON =======================
	# x,y = make_anoisysignal(a=0.2,f=440.0,fe=44100.0,ph=0,d=5)
	# conversion_wav(x,y,"lasono.wav")
	# ======================= FIN DU TP1 =======================

	# ======================= TP2 =======================
    # A = 1.5
    # f = 750
    # fe = 8000
    # nmax = 32
    # d = 3*1/f
    # Sig1 = Triangle(A, f, fe, d, 0)
    # sig_t, sig_s, ns, ans, bns = fourier(A, nmax, f, fe, d, Sig1.an, Sig1.bn)
    # tracerFourier(sig_t, sig_s, ns, ans, bns, nmax)

	# ======================= TP2 Partie 2 - FFT =======================
    # N = 32  # the number of points in signal s(n*te) et S(n*Fe)
    #         # Power of 2 !
    # Fe = 8000.       # the sampling rate
    # Te = 1./Fe       # the sampling period
    # freqStep = Fe/N  # resolution of the frequency IN FREQUENCY DOMAIN
    # f = 3*freqStep   # frequency of the sine wave
    #                  # On choisit un multiple de freqstep
    # T = 1.0/f        # periode de la sinusoide
    # a = 255
    # d = N*Te
    # Sig1 = Dentscie(a, f, Fe, d, 0)
    # x,y = Sig1.xyvalues()
    # #==== Calcul du spectre du signal 
    # s_f = np.fft.fft(y)       # Spectrum
    # print("fft result is a {} of len {} of type {}\n".format(type(s_f),len(s_f),type(s_f[0])))

    # # Plot
    # plot_fft(N, freqStep, Fe, x, y, s_f)

    # plt.savefig("fft_example{}.png".format(N))
    # plt.show()

	# ======================= TP2 Partie 2 - Analyse =======================
    # filename = "La3piano.wav"
    # Fe,s_t = wavfile.read(filename, mmap=False)    
    # # on ne fait la FFT que des N premiers samples
    # N = 1024 
    # freqStep = Fe/N
    # t = np.arange(N)/Fe
    # s_t = s_t[0:N]
    # s_f = np.fft.fft(s_t)   # Spectrum
    
    # plot_fft(N, freqStep, Fe, t, s_t, s_f)
    # plt.show()

	# ======================= TP2 Partie 3 - Signal Bruite 10, 20, 400 =======================
	# Fe = 1000
	# N = 256 
	# Sig1 = Cos(1, 10, Fe, N/Fe, 0)
	# Sig2 = Cos(1, 20, Fe, N/Fe, 0)
	# Sig3 = Cos(1, 400, Fe, N/Fe, 0)
	# x1,y1 = Sig1.xyvalues()
	# x2,y2 = Sig2.xyvalues()
	# x3,y3 = Sig3.xyvalues()	
	# x4, y4 = x3, []
	# for k in range(len(x3)):
	# 	y4.append(y1[k] + y2[k] + y3[k])
	# fig,ax = plt.subplots()

	# plot_on_ax(ax,x4,y4,Sig1.label(), "b--.")
	# decorate_ax(ax,"Une sinusoide")

	# plt.savefig("./basic_sin.png")
	# plt.show()

	# freqStep = Fe/N
	# s4 = np.fft.fft(y4)   # Spectrum
	
	# plot_fft(N, freqStep, Fe, x3, y4, s4)
	# plt.show()
	# ======================= TP2 Partie 3 - CONVOLUTION =======================
	# coef = [-6.849167e-003, 1.949014e-003, 1.309874e-002,1.100677e-002,
    #             -6.661435e-003,-1.321869e-002, 6.819504e-003, 2.292400e-002,7.732160e-004,
    #             -3.153488e-002,-1.384843e-002,4.054618e-002,3.841148e-002,-4.790497e-002,
    #             -8.973017e-002, 5.285565e-002,3.126515e-001, 4.454146e-001,3.126515e-001,
    #             5.285565e-002,-8.973017e-002,-4.790497e-002, 3.841148e-002, 4.054618e-002,
    #             -1.384843e-002,-3.153488e-002, 7.732160e-004,2.292400e-002,6.819504e-003,
    #             -1.321869e-002,-6.661435e-003, 1.100677e-002,1.309874e-002,1.949014e-003,
    #             -6.849167e-003]
	
	# sig_filtre = convolution(y4, coef)
	# sp_sig_filtre = np.fft.fft(sig_filtre)
	# plot_fft(N, freqStep, Fe, range(len(sig_filtre)), sig_filtre, sp_sig_filtre)
	# plt.show()
	# ======================= TP3 - QUANTIFICATION =======================
	# a=5.0
	# b=3
	# # Signal à quantifier
	# fe = 2000.0
	# f = 50.0
	# d = 0.04
	# Sig1 = Sin(a, f, fe, d, 0)
	# x,y = Sig1.xyvalues()
	# quantif, quantifpts, err , dico, dicopt = QuantifierSignal(y, b)

	# c1 = convertirBinaire(quantifpts, b)
	# c2 = eclaterConversion(c1)
	
	# fig,ax = plt.subplots()
	# plot_on_ax(ax,x,y,Sig1.label())
	# plot_on_ax(ax,x,quantif,Sig1.label(), "ro")
	# decorate_ax(ax,"Une sinusoide")
	# plt.show()
        
	# fig2, ax2 = plt.subplots()
	# plt.step(range(len(c2)),c2)
	# plt.show()
        
	# convmlt3 = MLT3(c2, 5)
	# plt.step(range(len(convmlt3)),convmlt3)
	# plt.show()