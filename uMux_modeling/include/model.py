import math
import numpy as np


### ENTITIES ###

    #SQUID Entity: Acts as a non-linear inductor, simply calculates this inductance based off of input flux (given by mutual Inductance) Componants: Mutual Inductance, Inductance. Should be flexible to include multiple inputs with different couplings (Flux Ramp, TES ...) Params: SQUID params, Inputs: Flux, Outputs: Inductance

    #Resonator Entity: A resonant frequency function in a transmission. Resonant frequency changes based on inductive load. : Componants: Resonance Params, Will be coupled to an inductor componant, which is coupled to a SQUID. Params: Resonator Params, Inputs: Inductance, Outputs: Resonant Params, S21
    
    
class Resonator:
    """ Summary:
    _id -> resonator number id
    params:[a, tau, fr, Qr, Qc, phi0] -> resonator params as defined by the fit model:
            - fr: resonant frequency
            - bw: resonator bandwidth
            - Qr: quality factor (to expand)
            -  Theta0
    """
    RESONATOR_COUNT = 0
    
    def __init__(self, params):
        self._id = Resonator.RESONATOR_COUNT
        Resonator.RESONATOR_COUNT+=1
        
        self.params = params
    
    def getID(self):
        return self._id
    
    def setParams(self, params):
        self.params = params

    def getParams(self):
        return self.params
    
    def phase_model(self, f):
        fr, Qr, theta0 = self.params[2], self.params[3], self.params[5]
        return -theta0 + 2 * np.arctan(2 * Qr * (1 - f / fr))
    
    def t21_model(self, f):
        a, tau, fr, Qr, Qc, phi0 = self.params[0],self.params[1],self.params[2],self.params[3],self.params[4],self.params[5]
        delay = np.exp(-2 * np.pi * f * tau * 1.0j)
        resonator_term = (np.exp(1.0j * phi0) * Qr / Qc) / (1 + 2.0j * Qr * (f - fr) / fr)
        return a * delay * (1 - resonator_term)

### COMPONANTS ###

    #Mutual Inductance Componant: Simple, just computes a flux based on a current or vice versa

    #Inductor Componant: Maybe just a value, but could alse be defined through impedence value

    #Transmission Line Componant: idk yet, maybe not imporant
    
    #Impedance: general complex number impedance, can communicate with inductance, capacitance, and so on. 

    #input Current Componant, hold info about input current, simply outpluts a current on a wire, could be TES or Flux Ramp. Sort of depends. Takes in current responce in the form of a generato


### SYSTEMS ###

    #Channel System: Can simply add in line a resonator, inductor, SQUID, and input current

    #Readout System: Can couple many channels, stack their response, demodulate. Probably the most intensive part.


### HELPERS ###

    #Flux Ramp Generator: At first a simple current function gnerator, but could couple in resonant frequency as tone tracking and such.

    #Pulse Generator

    #TES Coupler
    
    #Resonance Loader


