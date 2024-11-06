import numpy as np
import matplotlib.pyplot as plt
import time
import math

import circuit_fun as ct
import NR_fun as fx
import time_fun as tm

from scipy.linalg import expm

########################################################
fast = True
########################################################
print('#####################')
print("WARNING: this might take a couple of hours...")
########################################################
#define algorithms
#algorithms = ['NRC','NRCinv', 'NRD', 'NRX', 'IWA','FPPF','NRI','SC','GS', 'NRI0','NRX0','NRIM','NRIsum']#all
# algorithms = ['NRC','GS','FPPF']#some
if fast:
    algorithms = ['IWA','SC','NRI0','NRIM','NRIsum']#faster algorithms
    # algorithms = ['NRI0']#fastest algorithm for code testing
else:
    algorithms = ['NRC','NRCinv','NRR', 'NRD', 'NRX','FPPF', 'IWA','NRI','SC','GS']#algorithms from others
Nalgs = np.size(algorithms)#num of algs
M = 1 #for NRIsum
itmax = 25
prec = 0.00001
print('#####################')
print("Algorithms: %s" % (', '.join(algorithms)))
########################################################
if fast:
    Nnodos = np.arange(50, 401, 50)
else:
    Nnodos = np.arange(10, 201, 10)
# Nnodos = np.arange(10, 101, 10)

Nexample = 25
Dmin = 100  #min distance btwn nodes
Dmax = 200  #max distance btwn nodes
V = 13.2  #in kV
########################################################
#define time
t0 = 0  #begining of time in min
tf = 24 * 60  #end of time
T = 10  #time step
nn = int((tf - t0) / T) + 1  #num of instants
t = np.linspace(t0, tf, nn)  #time vector in mins
########################################################
prtvol = False
num_cts = 5#num of circuits for each N

cto_meaniterations = np.zeros((num_cts,Nalgs))
cto_meantime = np.zeros((num_cts,Nalgs))
cto_meanerror = np.zeros((num_cts,Nalgs))
cto_meanabsgues = np.zeros((num_cts,Nalgs))

meaniterations = np.zeros((len(Nnodos),Nalgs))
meantime = np.zeros((len(Nnodos),Nalgs))
meanerror = np.zeros((len(Nnodos),Nalgs))
meanabsgues = np.zeros((len(Nnodos),Nalgs))
########################################################
#load DG and load profiles
PVmodel = tm.read_pv_profile()
fpPV = 0.98
ComModel, ResModel, IndModel = tm.read_load_profile()
fpCom, fpRes, fpInd = (0.85, 0.90, 0.8)
########################################################
example_circuit_flag = True
index_Nnodos = 0
########################################################
for N in Nnodos:
    ########################################################
    print('#####################')
    print("Number of nodes with loads: %s" % ( N))
    index = 0
    for _ in range(num_cts):
        ########################################################
        print("    Circuit counter %s" % (_))
        ########################################################
        #create circuit topology
        start_time = time.time()
        nodes, lines = ct.create_circuit(N, Dmin, Dmax)
        #draw circuit to png
        ########################################################
        #loads at nodes
        load, loadmix = ct.load_circuit(nodes, 150, 10)  #in kW
        S = 10*np.ceil(load/10).sum()  #nominal power in kVA
        #generation at nodes
        pv = ct.DG_circuit(nodes, 0.50, 25, 4)  #in kW
        #impendances of the circuit
        Y, Y0, Y00 = ct.impendances_circuit(lines, load)
        Ybase = S / (V**2) / 1000  #divide by 1000 to obtain Ybase in Ohms
        ########################################################
        #normalized circuit
        barY = Y / Ybase
        barY0 = Y0 / Ybase
        barY00 = Y00 / Ybase
        
        barLoad = load / S
        barpv = pv / S
        ########################################################
        end_time = time.time()
        print("    Time needed for creation: %s [ms]" % (np.round(1000 * (end_time - start_time), 0)))
        ########################################################
        if N>=Nexample and example_circuit_flag and not(fast):
            ct.summary_circuit(nodes, lines,load,pv,'example_circuit')
        ########################################################
        #prefill vectors and matrices
        ve = np.zeros((N, nn), dtype=complex)
        ese = np.zeros((N, nn), dtype=complex)
        bars0 = np.zeros((1, nn), dtype=complex)
        Ppv = np.zeros((N, nn))
        Pload = np.zeros((N, nn))
        Qpv = np.zeros((N, nn))
        Qload = np.zeros((N, nn))
        fpPvVec = np.zeros((N, nn))
        fpLoadVec = np.zeros((N, nn))
        its = np.zeros((Nalgs, nn))
        ittime = np.zeros((Nalgs, nn))
        gues = np.zeros((Nalgs, nn))
        verrors = np.zeros((Nalgs - 1, nn))
        ########################################################
        #initial conditions for iterations
        R0 = np.eye(N)
        Phi0 = np.zeros((N, N))
        V0 = R0 @ expm(1j * Phi0)
        ########################################################
        #loop through time
        start_time_loop = time.time()
        for kk in range(nn):
            ##################################
            #instante
            tt = t[kk]
            ##################################
            #PV generation
            pv_gen = tm.pv_interpole(barpv, tt, PVmodel)
            
            Ppv[:, kk] = pv_gen
            Qpv[:, kk] = tm.reactive_power(pv_gen, fpPV)
            ##################################
            #Load
            load_com, load_res, load_ind = tm.load_interpole(barLoad, tt, loadmix, ComModel, ResModel, IndModel)
            Pload[:, kk] = load_com + load_res + load_ind
            Qload[:, kk] = tm.reactive_power(load_com, fpCom) + tm.reactive_power(load_res, fpRes) + tm.reactive_power(load_ind, fpInd)
            ##################################
            #Power balance
            Pbalance = Ppv[:, kk] - Pload[:, kk]
            Qbalance = Qpv[:, kk] - Qload[:, kk]
            
            barS = Pbalance + 1j * Qbalance
            ese[:, kk] = barS
            barS = np.diag(barS)
            ##################################
            #algs
            for alg in range(Nalgs):
                name = algorithms[alg]
                  
                arguments = '(barY, barY0, barS, V0, itmax, prec)'
                if name=='NRC' or name=='NRD' or name=='NRCinv':
                    arguments = '(barY, barY0, barS, R0, Phi0, itmax, prec)'
                if name=='NRIsum':
                    arguments = '(barY, barY0, barS, V0, itmax, prec,M)'
                if name=='FPPF':
                    arguments = '(barY, barY0, barS, V0, 6, prec)'
                
                function = 'fx.'+name+arguments
                
                vvv, its[alg, kk], ittime[alg, kk], gues[alg, kk] = eval(function)
                if alg == 0:
                    ve[:, kk] = vvv
                else:
                    verrors[alg-1, kk] = np.linalg.norm(ve[:, kk] - vvv)/np.linalg.norm(ve[:, kk])*100
            ##################################
            #power at root node
            VVV = np.diag(ve[:, kk])
            bars0[:, kk] = np.ones((1, N)) @ np.conj(barY0) @ (np.conj(VVV) - np.eye(N)) @ np.ones((N, 1))
            ##################################
            #initial conditions
            #V0 = VVV
            #R0 = abs(VVV)
            #Phi0 = np.angle(VVV)
        end_time_loop = time.time()
        loop_time = (np.round(1000 * (end_time_loop - start_time_loop), 0))
        print("    Time needed for loop: %s [ms]" %loop_time)
        ############################################
        if N>=Nexample and example_circuit_flag and not(fast):
            tm.print_example_quantities(t, S*ese, S*bars0,ve,'example_circuit_quantities')
            example_circuit_flag = False
        ############################################
        cto_meaniterations[index,:], cto_meantime[index,:], cto_meanerror[index,:], cto_meanabsgues[index,:] = tm.analize_voltages(algorithms,its, ittime, verrors,gues,prtvol)
        print("    Fraction of time needed for each algorithm w/r to loop: %s [%%]" % (np.round(100*nn*cto_meantime[index,:]/loop_time)))
        print("    Fraction of time for non-algorithm: %s [%%]" % (100-np.round(100*nn*cto_meantime[index,:]/loop_time).sum()))
        index = index + 1
        ############################################
    meaniterations[index_Nnodos,:] = np.mean(cto_meaniterations, axis=0)
    meantime[index_Nnodos,:] = np.mean(cto_meantime, axis=0)
    meanerror[index_Nnodos,:] = np.mean(cto_meanerror, axis=0)
    meanabsgues[index_Nnodos,:] = np.mean(cto_meanabsgues, axis=0)
    index_Nnodos = index_Nnodos + 1
########################################################
if fast:
    subsets_algorithms = [
        algorithms
        ]
else: 
    subsets_algorithms = [
        ['NRC','NRCinv','NRR','NRD','NRX','IWA'],
        ['NRC','FPPF','GS'],
        ['NRX','IWA','NRI','SC'],
        ]
########################################################
for subset in subsets_algorithms:
    print('#####################')
    print('    Subset: '+', '.join(subset))
    indices = np.array([i for i, string in enumerate(algorithms) if string in subset])
    file_name_sfx = '_'.join(subset)+'.eps'
    if fast:
        file_name_sfx = 'fast_'+file_name_sfx
        
    # plt.rcParams.update({'font.size': 14})
    font_size = 16
    ############################################
    #time vs nodes
    plt.plot(Nnodos,meantime[:,indices], marker='*', linestyle='-', markersize=5)
    plt.grid(True)
    plt.legend(subset,loc='upper left', fontsize=font_size-2)
    plt.xticks(Nnodos[1::2])
    plt.title('Mean iteration time', fontsize=font_size)
    plt.ylabel('[ms]', fontsize=font_size)
    plt.xlabel('N', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.savefig('time_'+file_name_sfx, format='eps')
    plt.show()
    plt.close()
    ############################################
    #iterations vs nodes
    plt.plot(Nnodos,meaniterations[:,indices], marker='*', linestyle='-', markersize=5)
    plt.grid(True)
    plt.legend(subset,loc='upper left', fontsize=font_size-2)
    plt.xticks(Nnodos[1::2])
    plt.title('Mean iterations', fontsize=font_size)
    #plt.ylabel('iterations')
    plt.xlabel('N', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.savefig('iterations_'+file_name_sfx, format='eps')
    plt.show()
    plt.close()
    ############################################
    #gues vs nodes
    plt.plot(Nnodos,meanabsgues[:,indices], marker='*', linestyle='-', markersize=5)
    plt.grid(True)
    plt.legend(subset,loc='upper left', fontsize=font_size-2)
    plt.xticks(Nnodos[1::2])
    plt.tight_layout()
    plt.title('Mean ||g(.)||', fontsize=font_size)
    plt.ylabel('[p.u.]', fontsize=font_size)
    plt.xlabel('N', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.show()
    plt.close()
    ############################################
    #voltage errors w/r to first ABSOLUTE algorithm  vs nodes
    plt.plot(Nnodos,meanerror[:,indices], marker='*', linestyle='-', markersize=5)
    plt.grid(True)
    plt.legend(subset,loc='upper left', fontsize=font_size-2)
    plt.xticks(Nnodos[1::2])
    plt.tight_layout()
    plt.title('Mean voltage error w/r to '+algorithms[0], fontsize=font_size)
    plt.xlabel('N', fontsize=font_size)
    plt.ylabel('%', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.show()
    plt.close()
    ############################################
    #Time rate w/r to first subset algorithm vs nodes
    plt.plot(Nnodos,meantime[:,indices[1:]]/(meantime[:,indices[0]].reshape(-1,1)@np.ones((1,len(subset)-1))), marker='*', linestyle='-', markersize=5)
    plt.grid(True)
    plt.legend(subset[1:],loc='upper left', fontsize=font_size-2)
    plt.xticks(Nnodos[1::2])
    plt.tight_layout()
    plt.title('Time rate w/r to '+subset[0], fontsize=font_size)
    plt.ylabel('[ms/ms]', fontsize=font_size)
    plt.xlabel('N', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.savefig('timerate_'+file_name_sfx, format='eps')
    plt.show()
    plt.close()
    ############################################