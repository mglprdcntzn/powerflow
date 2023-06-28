%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The following is the description of all the files in this github.

The Python 3.10 scripts are for obtaining all the figures in the paper:

	Parada Contzen, M. "Enhancing performance of power flow algorithms for distribution smart grids with renewable energy sources." 2023

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
File: main.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	The main file to obtain all figures in the paper. It has the following structure:
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	Preable:
		1.- Defines algorithms
		2.- Defines nodes numbers and other topological and electric parameters.
		3.- Defines time
		4.- Generates some vectors to save the simulated data
		5.- Reads load and PV profiles
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	Main:
		1.- Loops over the defined number of nodes:
			a) Loops over the number of circuits (5 in the paper):
				i) creates a circuit with loads and impedances.
				ii) Normalizes circuit parameters.
				iii) Generates vectors to save simulated data
				iv) Generates initial conditions for all instants
				v) Loops over 24hrs time:
					- Interpoles load and generation
					- Calculates balance
					- Loops over defined algorithms:
						+ solves power flow with algorithm
					- Determines injected power according to first algorithm
				vi) Analize quantities of 24hrs period for all algorithms
			b) Average values for each number of nodes.
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	Results:
		1.- Defines subsets of algorithms
		2.- Loops over subsets:
			a) Plots different graphs for each subset of algorithms with respect to number of nodes
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
File: NR_fun.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	Functions to implement the different power flow algorithms described in the paper. The variables and function names are directly taken from the paper. For example,
	NRC(barY, barY0, barS, R0, Phi0, itmax, prec): uses the classic Newton-Raphson algorithm to determine the voltages in a given circuit at a given precision. As inputs it accepts the matrices that describe the circuit and the aggregated load, the initial condition, and the maximum iteration number and precision. It returns the complex voltages, the number of iterations, and the time (in ms) needed to complete the procedure. All algorithms have a similar structure.
	
	gue(VVV, barY, barY0, barS): calculates the complex power balance for a given voltage.
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
File: circuit_fun.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	Functions related to the creation, plotting, and parametrization of distribution circuits.

	create_circuit(N, Dmin, Dmax): Creates a random circuit topology with N non-root nodes. The structure is mostly a radial tree, but allows certain cycles by connecting geographically close branches.

	print_circuit(nodes, lines, plt_name,cts_data): Plots a circuit to an eps image. The example circuit in the paper is plotted with this function.

	impendances_circuit(lines, load): given a set of lines/edges and maximum values for the loads at the nodes that define this edges, this function defines the impedances for each line according to the line length and the maximum possible power that it might distribute. It returns the addmitances matrices that describe the circuit.

	load_circuit(nodes, LoadMean, dev): For a given set of nodes, this function generates random load rates and the fraction that each of three classes (residential, industrial, commertial) represent on the total rated value. 

	DG_circuit(nodes, PVprob, PVmean, PVdev): For a given set of nodes, this functions generate random PV rates.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
File: time_fun.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	Functions related to the simulation in time of a circuit. Some of these functions are:

	def read_load_profile() and read_pv_profile(): respectively read from csv files the splin polynomial parameters of a daily profile of the load classes and the photo-voltaic generation.

	load_interpole and pv_interpole: given installed or rated values and a splin model, these functions interpole the load or the PV generation at the nodes at a given instant of the day.

	print_example_quantities: prints the daily simulation seen in the paper

	analize_voltages(headers, its, ittime, verrors,gues,prtvol): this functions average different measuremets tanken from an entire period of the circuit simulations and generates a report.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Files:
	load_com_profile.csv
	load_ind_profile.csv
	load_res_profile.csv
	pv_profile.csv
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	These files content as comma separted values the parameters of the splin models of the load according to their class, and of the photovoltaic generation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%