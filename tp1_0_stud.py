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
#---------------------------------------
def decorate_ax(ax, title, loca="lower left"):
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=loca)
#=======================================
class Sinusoide:
    def __init__(self, amplitude, frequence, echantillonage, dephasage, nbpts, trace, typesignal, mu=0, sigma=0, nbimpuls=0, dureeimpuls=0):
        self.amplitude = amplitude
        self.frequence = frequence
        self.echantillonage = echantillonage
        self.dephasage = dephasage
        self.nbpts = nbpts
        self.trace = trace
        self.label = f"s1 : a={amplitude}, f={frequence}, fe={echantillonage}, ph={dephasage}, d={nbpts}"
        self.typesignal = typesignal
        self.mu = mu
        self.sigma = sigma
        self.nbimpuls = nbimpuls
        self.dureeimpuls = dureeimpuls
#=======================================
def dessin(Signal):
    if(Signal.typesignal == "sin"):
        x,y=make_signal(Signal.amplitude, Signal.frequence, Signal.echantillonage, Signal.dephasage, Signal.nbpts, signal_sin)
    elif(Signal.typesignal == "dentscie"):
        x,y=make_signal(Signal.amplitude, Signal.frequence, Signal.echantillonage, Signal.dephasage, Signal.nbpts, signal_dentscie)
    elif(Signal.typesignal == "triangle"):
        x,y=make_signal(Signal.amplitude, Signal.frequence, Signal.echantillonage, Signal.dephasage, Signal.nbpts, signal_triangle)
    elif(Signal.typesignal == "carre"):
        x,y=make_signal(Signal.amplitude, Signal.frequence, Signal.echantillonage, Signal.dephasage, Signal.nbpts, signal_carre)
    elif(Signal.typesignal == "bruitgauss"):
        x,y=signal_bruit(Signal.echantillonage, Signal.nbpts, Signal.mu, Signal.sigma)
    elif(Signal.typesignal == "bruitimpuls"):
        x,y=signal_bruit(Signal.echantillonage, Signal.nbpts, Signal.mu, Signal.sigma)
        x,y=impulsif(x,y, Signal.nbimpuls, Signal.dureeimpuls)
    return x,y

def dessiner(ListeSinusoides, titre, titrefichier, fig = None, ax= None, separes = 0):
    if(separes == 0):
        fig, ax = plt.subplots()
        fig.tight_layout()
        for Signal in ListeSinusoides:
            x,y = dessin(Signal)
            ax.plot(x, y, Signal.trace, label=Signal.label)
        decorate_ax(ax, titre) 
        ax.set_xlabel('time (s)')
        ax.set_ylabel('voltage (V)')
    else:
        fig, axs = plt.subplots(nrows=len(ListeSinusoides))
        fig.tight_layout()
        for k in range(len(ListeSinusoides)):
            Signal = ListeSinusoides[k]
            x,y = dessin(Signal)
            axs[k].plot(x, y, Signal.trace, label=Signal.label)
            axs[k].set_xlabel('time (s)')
            axs[k].set_ylabel('voltage (V)')
    plt.savefig(f"./{titrefichier}")
    plt.show()

def signal_carre(a, f, fe, ph, d, t):
    return 2*math.floor(f*t)-math.floor(2*f*t)+1

def signal_sin(a, f, fe, ph, d, t):
    omega = 2*math.pi*f #pulsation w=2pi*f
    return a*math.sin((omega*t)+ph)

def signal_dentscie(a, f, fe, ph, d, t):
    return 2*a*(f*t - math.floor(f*t)-1/2)

def signal_triangle(a, f, fe, ph, d, t):
    return a*(4*(abs(f*t-math.floor(f*t + 1/2)))-1)

def make_signal(a, f, fe, ph, d, fonction):
    N = int(d*fe) #nombre de points
    te = 1.0/fe # intervalle d'échantillonage t=1/f

    sig_t = []
    sig_s = []
    for i in range(N):
        t = te*i #points auxquels on va évaluer la fonction pour tracer la sinusoide
        sig_t.append(t) # liste des abcisses 
        val = fonction(a, f, fe, ph, d, t)
        sig_s.append(val)
    return sig_t, sig_s

def signal_bruit(fe, d, mu, sigma):
    N = int(d*fe) #nombre de points
    te = 1.0/fe # intervalle d'échantillonage t=1/f

    sig_t = []
    sig_s = []
    for i in range(N):
        t = te*i #points auxquels on va évaluer la fonction pour tracer la sinusoide
        sig_t.append(t) # liste des abcisses 
        val = np.random.normal(mu, sigma)
        sig_s.append(val)
    return sig_t, sig_s

def impulsif(sig_t, sig_s, nbimpuls = 0, duree = 0):
    intervalle = float(sig_t[1])
    indexnonzero = random.sample(range(0, len(sig_s)), nbimpuls)
    final = []
    for k in range(len(sig_s)):
        final.append(0)
        for j in indexnonzero:
            if k == j:
                final[k]=sig_s[k]
    if(duree != 0):
        nbrepet = int(duree/intervalle)
        for j in range(1, nbrepet):
            for l in indexnonzero:
                if(l+j < len(final)):
                    final[l+j] = final[l]
    sig_s = final
    return sig_t, sig_s
#=======================================
def conversion_wav(x, y, sortie = "son.wav"):
    nbCanal = 2
    nbOctets = 1
#=======================================
Sin = []
def partie1():
    Sin.clear()
    Sin.append(Sinusoide(2.0, 50.0, 1000, 0, 80/1000, "--g*", "sin"))
    dessin(Sin, "Une sinusoide ..", "basic_sin.png")

def partie2():
    Sin.clear()
    Sin.append(Sinusoide(1, 1/0.02, 1/0.002, 0, 20*0.002, "bo", "sin"))
    Sin.append(Sinusoide(0.5, 1/0.02, 1/0.001, math.pi, 40*0.001, "r.", "sin"))
    dessin(Sin, "Deux sinusoides ..", "basic_sin.png")
    '''
    Les amplitudes sont différentes: 1V pour la grande, 0.5V pour la petite
    Les fréquences sont identiques
    La sinusoide rouge est déphasée de pi radians
    La fréquence d'échantillonage est 2 fois plus importante pour la petite sinusoide    
    
    '''

def carre():
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-bo", "carre"))    
    dessin(Sin, "Un carré ..", "carre.png")
     
def scie():
    Sin.clear()
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-bo", "dentscie"))
    dessin(Sin, "Dent de scie", "scie.png")

#impossible d'avoir un triangle parfait puisque cela voudrait dire que pour un t donné, on a deux valeurs

def bg():
    Sin.clear()
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-go", "bruitgauss", 0, 0.2))
    dessin(Sin, "Dent de scie", "bruitgauss.png")

def bi():
    Sin.clear()
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-go", "bruitimpuls", 0, 0.2, 20, 0.006))
    dessin(Sin, "Dent de scie", "bruitgauss.png")
#=======================================

if __name__ == '__main__':
    Sin.clear()
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-bo", "dentscie"))
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-go", "bruitgauss", 0, 0.2))
    Sin.append(Sinusoide(3, 50.0, 800, 0, 0.08, "-go", "bruitimpuls", 0, 0.2, 20, 0.006))
    dessiner(Sin, "Dent de scie2", "scie.png", None, None, 0)
