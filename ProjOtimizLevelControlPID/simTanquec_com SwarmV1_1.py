# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:57:17 2022

@author: rmbra
"""
import numpy as np
import copy
import time
import sys

def show_vector(vector):
  for i in range(len(vector)):
    if i % 8 == 0: # 8 columns
      print("\n", end="")
    if vector[i] >= 0.0:
      print(' ', end="")
    print("%.4f" % vector[i], end="") # 4 decimals
    print(" ", end="")
  print("\n")


# Classes relacionadas as partículas e movimentos #

class Particle(object):
    def __init__(self, dim, minx, maxx, maxModVel):
        self.dim=dim
        # iniciando vetores de dimensão=dim da partícula
        self.pos = ((maxx - minx) * np.random.random(self.dim) + minx)
        self.vel = ((maxx - minx) * np.random.random(self.dim) + minx)
        self.best_part_pos = copy.copy(self.pos) 
        self.result=sys.float_info.max  #o maior valor possível
        self.bestResult=sys.float_info.max  #o maior valor possível
        self.minPos=copy.copy(minx)
        self.maxPos=copy.copy(maxx)
        self.LimVel=maxModVel
        
                
    def Calc_velocity(self, w, c1, c2, bestSwarmPos):
         # compute new velocity of curr particle
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        self.vel = ((w * self.vel) +
                  (c1 * r1 * (self.best_part_pos - self.pos )) +  
                  (c2 * r2 * (bestSwarmPos - self.pos)) )  
        
        for x in range(self.dim):
            sinal=1.0
            if self.vel[x]<0:
                    sinal=-1.0
                
            if abs(self.vel[x])>self.LimVel:
                self.vel[x]=sinal*self.LimVel 
                
    def Calc_Position(self):        
        # compute new position using new velocity
        self.pos += self.vel
        
        for x in range(self.dim):
            if self.pos[x]<self.minPos[x]:
                self.pos[x]=self.minPos[x]   
            if self.pos[x]>self.maxPos[x]:
                self.pos[x]=self.maxPos[x]   
   
    def Evaluate(self, tq0, sim0):
        # capacidade_max=100
        # limite_vin=20
        # limite_vout=10
        # vin=0
        # vout=5
        # nivel_inicial=0
        # Ts=5
        # spoint=30
        
        # tq1=Tanque(capacidade_max,limite_vin,limite_vout,
        #            vin,vout,spoint,nivel_inicial,Ts)

        # Tsim=3000
        # # Kp=100
        # # Kd=.5
        # # Ki=0.01
        
        # [Kp, Ki, Kd]= self.pos
        # sim0=Simul(Tsim, Kp, Ki, Kd)
        
        self.result=sim0.go(tq0)
        
    def IsTheBestParticlePos(self):
        status=False    
        if abs(self.result)<abs(self.bestResult):
            status=True
        return status
    
    def UpdateBest(self):
        self.best_part_pos = copy.copy(self.pos)
        self.bestResult=self.result
            
            
class Swarm:
    def __init__(self, n, dim, minx, maxx, LV):
        self.n=n
        self.dim=dim
        self.minP=minx
        self.maxP=maxx
        self.MaxVel=LV
        self.swarm_=list()
        self.max_epochs=100
        self.epochs = 0
        self.best_swarm_pos = np.zeros(self.dim)
        self.best_swarm_result = sys.float_info.max  #o maior valor possível
        self.w = 0.729    # inertia
        self.c1 = 1.49445 # cognitive (particle)
        self.c2 = 1.49445 # social (swarm)
    
        
    def Create_initial(self):
        # create n random particles
        self.swarm_ = [Particle(self.dim, self.minP, self.maxP, self.MaxVel) for i in range(self.n)]
        
    def IsTheBestSwarmPos(self, particle_):
        status=False    
        if abs(particle_.result)<abs(self.best_swarm_result):
            status=True
        return status
    
    def UpdateSwarmBest(self, particle_):
        self.best_swarm_pos = copy.copy(particle_.pos)
        self.best_swarm_result=particle_.result
    
    def Solve(self,max_epochs, tq_, sim_):
        self.max_epochs = max_epochs
        
        self.epoch = 0
        
        while self.epoch < self.max_epochs:
        
            if self.epoch % 10 == 0 and self.epoch > 1:
              print("Epoch = " + str(self.epoch) +
                " best result = %.3f" % self.best_swarm_result)
        
            # process each particle
            for i in range(self.n): 
              
                # compute particle's velocity
                self.swarm_[i].Calc_velocity(self.w, self.c1, self.c2, self.best_swarm_pos)
                      
       
                # compute new position using new velocity
                self.swarm_[i].Calc_Position()
          
                #Evaluate the new position
                self.swarm_[i].Evaluate(tq_, sim_)
                # print (self.swarm_[i].result)
                # is new position a new best for the particle?
                if self.swarm_[i].IsTheBestParticlePos():
                    self.swarm_[i].UpdateBest()
                
                # is new position a new best overall?
                if self.IsTheBestSwarmPos(self.swarm_[i]):
                    self.UpdateSwarmBest(self.swarm_[i])
            
            # for-each particle
            self.epoch += 1
            
        # while
        
        return self.best_swarm_pos
        # # end Solve

# Classes relacionadas a simulação do tanque #          
class Tanque(object):
    def __init__(self, vol_, Lvin_, Lvout_, li_, lo_, sp_nivel, nivel0=0, Ts=1):
        # gerando nova randomização
        self.litros=vol_
        self.litros_min_in=li_
        self.litros_min_out=lo_
        self.LIM_litros_min_in=Lvin_
        self.LIM_litros_min_out=Lvout_
        self.setpoint_litros=sp_nivel
        self.nivel=nivel0
        self.err=100.0
        self.Ts=Ts
        
    def calc_nivel(self):
        self.nivel+=((self.litros_min_in-self.litros_min_out)*self.Ts/60.0)
        
    def calc_erro(self):
        self.err=100.0*(self.setpoint_litros-self.nivel)/self.setpoint_litros
        
class Simul(object):
    def __init__(self, Tsim_, Kp_, Ki_, Kd_):
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
        
            #print("Tempo %2d, Nivel %3.3f, Err %3.3f, PID %.3f Vz_In %3.3f" % (transcorrido, tq1.nivel, tq1.err, pid, tq1.litros_min_in))
            
        E2=self.E.__ipow__(2)
        SE2=np.sum(E2)*self.T
        return np.sum(SE2)

start_time=time.time()
dim = 3
# definition of object tanq, to be optimized and simulated
capacidade_max=100
limite_vin=20
limite_vout=10
LpMin=0 #vazao inicial de entrada
LpMout=5  #vazao inicial de saida - mas fica fixo na simulação
nivel_inicial=0
Ts=1
spoint=30

tqX=Tanque(capacidade_max,limite_vin,limite_vout,
           LpMin,LpMout,spoint,nivel_inicial,Ts)

# simulation parameter's definition  
[Kp, Ki, Kd]=[80.0, 1.0, 1.0]
Tsim=24000 #40 minutos
simX=Simul(Tsim,Kp, Ki, Kd)

# PSO parameter's definition
minPos = np.asarray([1.0, 0.0, 0.0])     # limites inferiores dos parametros
maxPos = np.asarray([2000.0, 1.0, 5.0])  # limites superiores dos parametros
maxAbsV=1000  # limite do vetor velocidade
nparticulas = 50
mepochs = 100

swm=Swarm(nparticulas,dim,minPos,maxPos,maxAbsV)
swm.Create_initial()

print("------------------------\n")
print("Setting num_particles = " + str(nparticulas))
print("Setting max_epochs    = " + str(mepochs))
print("\nStarting PSO algorithm\n")

best_position = swm.Solve(mepochs, tqX, simX)

print("\nPSO completed\n")
print("\nBest solution found:")

print(swm.best_swarm_pos)
print("The best solution = %.6f" % swm.best_swarm_result)

print("\nEnd particle swarm demo\n")
print("--- %s seconds ---" % (time.time() - start_time))

[Kp, Ki, Kd]=swm.best_swarm_pos
simFinal=Simul(Tsim, Kp, Ki, Kd)
IE2T=simFinal.go(tqX)

plt.plot(simFinal.N)
plt.show()


        