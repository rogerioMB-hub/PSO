# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:57:17 2022

@author: rmbra
"""
import numpy as np
#import copy
#import sys
import matplotlib.pyplot as plt


class Tanque(object):
    def __init__(self, vol_, vin_, vout_, li_, lo_, sp_nivel, nivel0=0, Ts=1):
        # gerando nova randomização
        self.litros=vol_
        self.litros_min_in=li_
        self.litros_min_out=lo_
        self.LIM_litros_min_in=vin_
        self.LIM_litros_min_out=vout_
        self.setpoint_litros=sp_nivel
        self.nivel=nivel0
        self.err=100.0
        self.Ts=Ts
        
    def calc_nivel(self):
        self.nivel+=((self.litros_min_in-self.litros_min_out)*self.Ts/60.0)
        
    def calc_erro(self):
        self.err=100.0*(self.setpoint_litros-self.nivel)/self.setpoint_litros
        
class Simul(object):
    def __init__(self, Tsim_, Kp_, Kd_, Ki_):
        self.Ts=1
        self.Tsim=Tsim_
        self.Kp=Kp_
        self.Kd=Kd_
        self.Ki=Ki_
        self.V=np.zeros(Tsim_)
        self.E=np.zeros(Tsim_)
        self.N=np.zeros(Tsim_)
        self.T=np.arange(Tsim_)
        

    def go(self, tq_):
        i=0
        for transcorrido in range(self.Tsim):
                
            lastN=tq_.nivel
            tq_.calc_nivel()
            
            tq_.calc_erro()
            
            p = self.Kp*tq_.err/100.0
            i=i+(self.Ki*tq_.err/100.0)
            d=self.Kd*(tq_.nivel-lastN)
            pid=p+i+d
                        
            tq_.litros_min_in=tq_.litros_min_in+pid
            if tq_.litros_min_in>tq_.LIM_litros_min_in:
                tq_.litros_min_in=tq_.LIM_litros_min_in
            if tq_.litros_min_in<0:
                tq_.litros_min_in=0
                
            self.V[transcorrido]=tq_.litros_min_in
            self.N[transcorrido]=tq_.nivel
            self.E[transcorrido]=tq_.err
        
            print("Tempo %2d, Nivel %3.3f, Err %3.3f, PID %.3f Vz_In %3.3f" % (transcorrido, tq1.nivel, tq1.err, pid, tq1.litros_min_in))
            
        E2=self.E.__ipow__(2)
        SE2=np.sum(E2)*self.T
        return np.sum(SE2)



capacidade_max=100
limite_vin=20
limite_vout=15
vin=0
vout=5
nivel_inicial=0
Ts=1
spoint=80

tq1=Tanque(capacidade_max,limite_vin,limite_vout,
           vin,vout,spoint,nivel_inicial,Ts)

Tsim=6000
[Kp, Ki, Kd]=[8835.9570956,   9.71800816, 1140.92252235]
IE2T=0

sim0=Simul(Tsim,Kp, Kd, Ki)
IE2T=sim0.go(tq1)
print(IE2T)

plt.plot(sim0.N)
plt.show()




        