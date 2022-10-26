# PSO algorithm to optimize PID parameters of a simulated control system (simple tank level control, one valve and a drain with a constant flow)

As the fist commit shows, the central idea is to share with other interested people the files for analisys and improvements, specially in the control code lines. 
One is a sketch of a simulator that, in the second file, is embedded in the evaluation of each particle. 
The objective is to simulate, considering each particle a sequence of parameters to be tested, to minimize the integral of the squared error multiplied by the time. 
The algorithm is still under development, however it already presents some interesting results.
