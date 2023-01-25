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
    return 2*math.floor(f*t)-math.floor(2*f*t)+1

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
#=======================================
if __name__ == '__main__':
    '''
    Sin = []
    
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-bo", signal_dentscie))
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-go", signal_bruit,"gauss", 0, 0.2))
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-ro", signal_bruit,"impulsif", 0, 0.2, 20))
    
    x,a = calculerSignal(Sinusoide(2, 50.0, 1000, 0, 0.08, "-bo", signal_sin))
    x,b = calculerSignal(Sinusoide(2, 50.0, 1000, 0, 0.08, "-bo", signal_bruit, "gauss", 0, 0.2))
    x,c = calculerSignal(Sinusoide(2, 50.0, 1000, 0, 0.08, "-bo", signal_bruit, "impulsif", 0, 4, 1, 0))
    d = []
    for k in range(len(a)):
        d.append(a[k] + b[k] + c[k])
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.tight_layout()

    ax1.plot(x, a, "-bo")
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('voltage (V)')
    ax2.plot(x, b, "-go")
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('voltage (V)')
    ax3.plot(x, c, "-go")   
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('voltage (V)')
    ax4.plot(x, d, "-ro")
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('voltage (V)')'''
    
    x,y = make_anoisysignal(amp=0.2,f=440.0,fe=44100.0,ph=0,d=5)
    conversion_wav(x,y)
