
### ENTITIES ###

    #SQUID Entity: Acts as a non-linear inductor, simply calculates this inductance based off of input flux (given by mutual Inductance) Componants: Mutual Inductance, Inductance. Should be flexible to include multiple inputs with different couplings (Flux Ramp, TES ...) Params: SQUID params, Inputs: Flux, Outputs: Inductance

    #Resonator Entity: A resonant frequency function in a transmission. Resonant frequency changes based on inductive load. : Componants: Resonance Params, Will be coupled to an inductor componant, which is coupled to a SQUID. Params: Resonator Params, Inputs: Inductance, Outputs: Resonant Params, S21


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


