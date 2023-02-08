#-*- coding: utf-8 *-*
'''
File tp1_q_vide.py  : Created on 28 feb 2012
@author : menez
Illustration de la quantification
'''
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
sys.path.append('/home/menez/EnseignementsCurrent/Cours_Telecom/TPs/Src')
import Signaux.tp1_0 as tp1

#---------------------------------------------
def make_dico(vmax, b):
    """
    Crée un dictionnaire uniforme de quantification sur "b" bits
    pour des signaux d'amplitude max "vmax"
    """
    pass

#---------------------------------------------
def QS(sig_s, vmax, b):
    """
    Quantificateur sur b bits d'un signal sig_s d'amplitude max vmax.
    Rend le signal quantifie et le bruit sur chaque echantillon.
    """
    
    # Creation du dictionnaire de quantification = une liste de ""codewords"
    D = make_dico(vmax, b)
    
    # Pour l'instant le signal quantifie est le meme que le signal original. 
    Q = sig_s
    
    # Donc le signal de bruit est donc 0 puisqu'il n'y a pas de quantification.
    E = [0]*len(sig_s) 

    return Q, E

#---------------------------------------------
def plot(inx, iny, leg, fmt='-bo', l=""):
    plt.plot(inx,iny,fmt,label=l)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (V)')
    plt.ylim([-5.5, +5.5])

#============================================
if __name__ == '__main__':
    np.set_printoptions(linewidth=250) 
    np.set_printoptions(precision=3, suppress=True)
    a=5.0
    b=3
  
    # Signal à quantifier
    fe = 2000.0
    f = 50.0
    d = 0.04
    x,y=tp1.make_sin(a,0,f=f,fe=fe,d=d)

    # Signal quantifié et erreur
    z,err = QS(y,a,b)
   
    # Plot des signaux : 
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    step = 2*a/(2**b)
    majorLocator = MultipleLocator(step) # Choisir la graduation en y
    ax.yaxis.set_major_locator(majorLocator) 
    plot(x,y,"",'bo', l="Signal")
    plot(x,z,"",'rs', l="Quantized")
    plot(x,err,"",'--x', l="Diff")

    title = "Sinusoide : $f_e={}, f={}, d={}$".format(fe,f,d)
    tp1.plot_fig_decoration(title)
    
    plt.show()    
