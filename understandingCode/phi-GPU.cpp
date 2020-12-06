#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <utility>
#include <algorithm>

#ifdef FOURTH
#  include "headers/hermite4.h"
#elif defined SIXTH
#  include "headers/hermite6.h"
#elif defined EIGHTH
#  include "headers/hermite8.h"
#else
#  error
#endif

// Define max number of bodies
#define N_MAX     (1 << 20) // Global
#define N_MAX_loc (1 << 20) // Local

// Instantiate the particle class
static Particle ptcl[N_MAX];       // Global
static Jparticle jptcl[N_MAX_loc]; // Local

// Define a dt for every time step
std::pair<double, int> t_plus_dt[N_MAX];

// Instantiate the force vector class
static Force force_tmp[N_MAX], force[N_MAX];

// List of active nodes (bodies)
static int active_list[N_MAX];

// Initialize bodies and declare the disk step
static int nbody, diskstep;

// MPI parameters
static int myRank, n_proc, name_proc;
static char processor_name[MPI_MAX_PROCESSOR_NAME];

// Performance variables
static int jstart, jend;
static double eps2;
static double time_cur, Timesteps=0.0, n_act_sum=0.0, g6_calls=0.0;
static double CPU_time_real0, CPU_time_user0, CPU_time_syst0;
static double CPU_time_real,  CPU_time_user,  CPU_time_syst;

// Define the method of getting the process time
#define PROFILE
#ifdef PROFILE
extern double wtime() {
	return MPI_Wtime();
}
#else
extern double wtime() {
	return 0;
}
#endif

/* Desc: Update real, user nad system time */
static void get_CPU_time(double *time_real, double *time_user, double *time_syst) {
	// Store the resource usage of a process
    struct rusage rscs_usage;

    // Declare time variables
	double sec_u, microsec_u, sec_s, microsec_s;

    // Examine the resource usage of a process
	getrusage(RUSAGE_SELF, &rscs_usage);

    // Time spent executing user instructions (in seconds)
	sec_u = rscs_usage.ru_utime.tv_sec;

    // Time spent in operating system code on behalf of processes (in seconds)
	sec_s = rscs_usage.ru_stime.tv_sec;

    // Time spent executing user instructions (in microseconds)
	microsec_u = rscs_usage.ru_utime.tv_usec;
    
    // Time spent in operating system code on behalf of processes (in microseconds)
	microsec_s = rscs_usage.ru_stime.tv_usec;

    // Sum to get a more precise measurement
	*time_user = sec_u + microsec_u * 1.0E-06;
	*time_syst = sec_s + microsec_s * 1.0E-06;

    // Get real time of the process
	struct timeval tv;
	gettimeofday(&tv, NULL);
	*time_real = tv.tv_sec + 1.0E-06 * tv.tv_usec;
}

// If a file is defined, use this function
/* Desc: Save new particles' state */
static void outputsnap(FILE *out){
	// Save algorithm parameters to the output file
    fprintf(out,"%04d \n", diskstep);
	fprintf(out,"%06d \n", nbody);
	fprintf(out,"%.8E \n", time_cur);
	
    // Save the final state of each particle
    for (int i = 0; i < nbody; i++){
		Particle &p = ptcl[i];
	   	fprintf(out,"%06d  %.8E  % .8E % .8E % .8E  % .8E % .8E % .8E \n", 
				p.id, p.mass, p.pos[0], p.pos[1], p.pos[2], p.vel[0], p.vel[1], p.vel[2]);
	}
}

// If a file is not defined, use this function instead
/* Desc: Create file, save new particles' state and also save on history file */
static void outputsnap(){
	// Name of the output file
    static char out_fname[256];
	sprintf(out_fname, "%04d.dat", diskstep);

    // Open desired file for writing or create it
	FILE *out = fopen(out_fname,"w");
	
    // Verify that the file was successfully created
    assert(out);

    // Recursive call with file declared (see function above)
	outputsnap(out);
	
    // Close file
    fclose(out);

    // Open the file data.con for history
	out = fopen("data.con","w");
	assert(out);
	outputsnap(out);
	fclose(out);
}

/* Desc: */
static void energy(int myRank){
	static bool init_call = true;
	static double einit;

	double E_pot = 0.0; // Accumulate Potential energy
	for (int i = 0; i < nbody; i++)
        E_pot += ptcl[i].mass * ptcl[i].pot;

	E_pot *= 0.5; // Divide potential energy by 2

	double E_kin = 0.0; // Accumulate kinetic energy
	for(int i = 0; i < nbody; i++)
        E_kin += ptcl[i].mass * ptcl[i].vel.norm2();
	E_kin *= 0.5; // Divide kinetic energy by 2

	assert(E_pot == E_pot); // ??
	assert(E_kin == E_kin);

	double mcm = 0.0;
	dvec3 xcm = 0.0;
	dvec3 vcm = 0.0;
	for (int i = 0; i < nbody; i++) {
		mcm += ptcl[i].mass; // Accumulate mass
		xcm += ptcl[i].mass * ptcl[i].pos; // Accumulate kg*m
		vcm += ptcl[i].mass * ptcl[i].vel; // Accumulate kg *m/s
	}
	// Total accumulated position and speed
	xcm /= mcm;
	vcm /= mcm;

	double rcm_mod = xcm.abs();
	double vcm_mod = vcm.abs();

	// Accumulate  momentum
	dvec3 mom = 0.0;
	for (int i = 0; i < nbody; i++)
        mom += ptcl[i].mass * (ptcl[i].pos % ptcl[i].vel);

	get_CPU_time(&CPU_time_real, &CPU_time_user, &CPU_time_syst); // 6 FLOPS


	if (init_call) {
        einit = E_pot + E_kin; // 1 FLOP
        init_call = false;
  	}

    double eerr = (E_pot+E_kin-einit)/einit; // 3 FLOPS
	einit = E_pot + E_kin; // 2 FLOPS
	

	if (myRank == 0) {
		printf("%.4E   %.3E %.3E   % .6E % .6E % .6E % .6E   %.6E \n",
			   time_cur, Timesteps, n_act_sum,
			   E_pot, E_kin, E_pot+E_kin, 
			   eerr,
			   CPU_time_user-CPU_time_user0);
		fflush(stdout);

		FILE *out = fopen("contr.dat","a");
		assert(out);

		fprintf(out,"%.8E  %.8E  %.8E   % .16E % .16E % .16E   % .16E   % .16E % .16E   % .16E % .16E % .16E   %.8E %.8E %.8E \n",
				time_cur, Timesteps, n_act_sum,
				E_pot, E_kin, E_pot+E_kin, eerr, 
				rcm_mod, vcm_mod, mom[0], mom[1], mom[2], 
				CPU_time_real-CPU_time_real0, CPU_time_user-CPU_time_user0, CPU_time_syst-CPU_time_syst0);
		fclose(out);
	}

    #ifdef CMCORR
        for(int i=0; i<nbody; i++) {
            ptci[i].pos -= xcm;
            ptci[i].vel -= vcm;
        }
    #endif
}

int main(int argc, char *argv[]) {
	// MPI configuration
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);	

    // Define the processor name
    MPI_Get_processor_name(processor_name, &name_proc);

    // Print the rank and the name of each processor
    printf("Rank of the processor %02d on %s \n", myRank, processor_name);

    // Instantiate the state predictor for every particle
	Predictor *ipred = Predictor::allocate(N_MAX);     // Global
    Predictor *jpred = Predictor::allocate(N_MAX_loc); // Local

    // Algorithm variables
	double eps, t_end, dt_disk, dt_contr, eta, eta_BH;
	
	MPI_Barrier(MPI_COMM_WORLD);
	if (myRank == 0) {
        // Load parallel configuration
		#ifdef FOURTH
				std::ifstream ifs("phi-GPU4.cfg");
		#endif
		#ifdef SIXTH
				std::ifstream ifs("phi-GPU6.cfg");
		#endif
		#ifdef EIGHTH
				std::ifstream ifs("phi-GPU8.cfg");
		#endif

		// Import configuration variables
		static char inp_fname[256];
		ifs >> eps >> t_end >> dt_disk >> dt_contr >> eta >> eta_BH >> inp_fname;
		
		// Abort execution if the configuration file could not be opened
		if(ifs.fail())
			MPI_Abort(MPI_COMM_WORLD, -1);
		
		// Import particles' parameters
		std::ifstream inp(inp_fname);
		inp >> diskstep >> nbody >> time_cur;
		
		// Make sure the number of particles do not exced the previoulsy defined maximum size
		assert(nbody <= N_MAX);
		assert(nbody/n_proc <= N_MAX_loc);
		
		// Abort execution if the particles' parameters do not fit the variable type
        if (inp.fail())
            MPI_Abort(MPI_COMM_WORLD, -2);
		
		// Import particles' initial state to each particle object
        for (int i = 0; i < nbody; i++) {
			Particle &p = ptcl[i];
			inp >> p.id >> p.mass >> p.pos >> p.vel;
			p.t = time_cur;
		}
		
		// Abort execution if the particles' initial states do not fit the variable type
        if(inp.fail())
            MPI_Abort(MPI_COMM_WORLD, -3);

        printf("\nBegin the calculation of phi-GPU4 program on %03d processors\n\n", n_proc); 
        printf("N       = %06d \t eps      = %.6E \n", nbody, eps);
        printf("t_beg   = %.6E \t t_end    = %.6E \n", time_cur, t_end);
        printf("dt_disk = %.6E \t dt_contr = %.6E \n", dt_disk, dt_contr);
        printf("eta     = %.6E \t eta_BH   = %.6E \n\n", eta, eta_BH);
        fflush(stdout); // Wait for the printing process to finish

		// Save particles' states if the particles' file has 0 as an "id"
        if (diskstep == 0)
            outputsnap();
        
		// Update times
        get_CPU_time(&CPU_time_real0, &CPU_time_user0, &CPU_time_syst0); // 6 FLOPS
    }

	// Send configuration parameters to every process
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&nbody,    1, MPI_INT,    0, MPI_COMM_WORLD);
	MPI_Bcast(&eps,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	MPI_Bcast(&eta,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eta_BH,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&t_end,    1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt_disk,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	MPI_Bcast(&dt_contr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&time_cur, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// NOTE: Every process does the following

	// Divide maximum diferential by 2 while it is greater than the minimum value of
	// interval of snapshot files output and interval for the energy control output
	double dt_max = 1. / (1 << 3); // 1/2^3 // 2 FLOPS
	while (dt_max >= std::min(dt_disk, dt_contr)) 
        dt_max *= 0.5; // 1 FLOP
	
	// Minimum diferential
    double dt_min = 1. / (1 << 23); // 1/2^23
	
	// Update controlled times
	double t_disk  = time_cur + dt_disk;
	double t_contr = time_cur + dt_contr;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(ptcl, nbody*sizeof(Particle), MPI_CHAR, 0, MPI_COMM_WORLD);

	// Start and end local particle index number 
	jstart = (myRank * nbody) / n_proc;
	jend   = ((1+myRank) * nbody) / n_proc;

	// Interval of particles for each process
	int n_loc = jend - jstart;
	eps2 = eps*eps;

	MPI_Barrier(MPI_COMM_WORLD);
	if (myRank == 0) // Update times
		get_CPU_time(&CPU_time_real0, &CPU_time_user0, &CPU_time_syst0);
	
	// Initialize particles' sates
	for (int l = 0; l < Particle::init_iter; l++) {
        #pragma omp parallel for
            for(int j = 0; j < n_loc; j++){
				// create a predictor object for each local particle
				// it estimates the position and speed of the particle for the next time step
                jpred[j] = Predictor(time_cur, Jparticle(ptcl[j+jstart]));
			}

        #pragma omp parallel for
            for (int i = 0; i < nbody; i++){
				// create a predictor object for each active particle, globally
				// it estimates the position and speed of the particle for the next time step
                ipred[i] = Predictor(time_cur, Jparticle(ptcl[i]));
			}

			// Compute the exerted forces by each particle from the processor subset
            calc_force(nbody, n_loc, eps2, ipred, jpred, force_tmp);

			// Calculate the resulting force for each particle
            MPI_Allreduce(force_tmp, force, nbody*Force::nword, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			// Update each particle's state
            for(int i = 0; i < nbody; i++){
                ptcl[i].init(time_cur, dt_min, dt_max, eta, force[i]);
                t_plus_dt[i].first = ptcl[i].t + ptcl[i].dt;	
                t_plus_dt[i].second = i;

				// Update local particle subset
                jptcl[i] = Jparticle(ptcl[i]);
		    }

		// Sort particles by time
		std::sort(t_plus_dt, t_plus_dt + nbody);
	}

	// Calcultate and print necessary Gflops to execute simulation
	if (myRank == 0) {
		get_CPU_time(&CPU_time_real, &CPU_time_user, &CPU_time_syst);
		double Gflops = Particle::flops * 1.e-9 * Particle::init_iter * double(nbody) * nbody 
			            / (CPU_time_real - CPU_time_real0);
		fprintf(stdout, "Initialized, %f Gflops\n", Gflops);
	}

	// Update energy
	energy(myRank);

	if (myRank == 0) // Update times
		get_CPU_time(&CPU_time_real0, &CPU_time_user0, &CPU_time_syst0);
	
	// Task time variables
	double t_scan = 0.0, t_pred = 0.0, t_jsend = 0.0, t_isend = 0.0, t_force = 0.0, t_recv = 0.0, t_comm = 0.0, t_corr = 0.0;

	// Update particles' states for each calculated time step
	while (time_cur <= t_end) {
		double t0 = wtime();
		double min_t = t_plus_dt[0].first; // New time of the first particle
		int n_act = 0;
		
		// Find all particles affected on the same time step and save them on active_list
        while (t_plus_dt[n_act].first == min_t){
			active_list[n_act] = t_plus_dt[n_act].second;
			n_act++;
		}
		double t1 = wtime();

    	#pragma omp parallel for
		for (int j = 0; j < n_loc; j++) {
			jptcl[j+jstart+1].prefetch(); // does nothing
			// create a predictor object for each local particle
			// it estimates the position and speed of the particle for the next time step
		   	jpred[j] = Predictor(min_t, jptcl[j+jstart]);
		}
		int ni = n_act;

    	#pragma omp parallel for
		for (int i = 0; i < ni; i++) {
			jptcl[active_list[i+1]].prefetch(); // does nothing
			// create a predictor object for each active particle, globally
			// it estimates the position and speed of the particle for the next time step
			ipred[i] = Predictor(min_t, jptcl[active_list[i]]);
		}

		double t2 = wtime();
		double t3;

		// Compute the exerted forces by each particle from the processor subset
		calc_force(ni, n_loc, eps2, ipred, jpred, force_tmp);

		double t4 = wtime();

		// Calculate the resulting force for each particle
		MPI_Allreduce(force_tmp, force, ni*Force::nword, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		double t5 = wtime();

    #pragma omp parallel for
		for (int i = 0; i < ni; i++){
			ptcl[active_list[i+1]].prefetch(); // does nothing
			Particle &p = ptcl[active_list[i]];
			// Correct position and velocity and update acceleration, jerk and pot
			p.correct(dt_min, dt_max, eta, force[i]);
			t_plus_dt[i].second = active_list[i]; // update index
			t_plus_dt[i].first  = p.t + p.dt; // update t + dt 
			jptcl[active_list[i]] = Jparticle(p); // save updated particle into local array
		}

		double t6 = wtime();

		// Sort particles by time
		std::sort(t_plus_dt, t_plus_dt + ni);
		
		double t7 = wtime();
		
		// Calculate different exceution times
		t_scan  += t1 - t0;
		t_pred  += t2 - t1;
		t_jsend += t3 - t2;
		t_force += t4 - t3;
		t_comm  += t5 - t4;
		t_corr  += t6 - t5;
		t_scan  += t7 - t6;


		time_cur = min_t;
		Timesteps += 1.0;
		n_act_sum += n_act;

		if (time_cur >= t_contr) {
			energy(myRank);
			t_contr += dt_contr;
		}

		// If disk time is reached, output results and update time disk
		if (time_cur >= t_disk) {
			if (myRank == 0) {
				diskstep++;
				outputsnap();
			}
			t_disk += dt_disk;
		}
	}

	// Next 2 lines do nothing
	double g6_calls_sum;
	MPI_Reduce(&g6_calls, &g6_calls_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	// Print results
	if (myRank == 0) {
		printf("\n");
		printf("Timesteps = %.0f   Total sum of integrated part. = %.0f   n_act average = %.0f\n", 
				Timesteps, n_act_sum, n_act_sum/Timesteps);
		printf("\n");
		double Gflops  = Particle::flops*1.e-9*double(nbody)*double(n_act_sum)/(CPU_time_real - CPU_time_real0);
		printf("Real Speed = %.3f GFlops \n", Gflops);

        #ifdef PROFILE
            double t_tot = t_scan + t_pred + t_jsend + t_force + t_comm + t_corr;
            t_force -= t_isend + t_recv;
            printf("        sec_tot     usec/step   ratio\n"); 
            printf("scan :%12.4E%12.4E%12.4E\n", t_scan, t_scan/Timesteps*1.e6, t_scan/t_tot);
            printf("pred :%12.4E%12.4E%12.4E\n", t_pred, t_pred/Timesteps*1.e6, t_pred/t_tot);
            printf("jsend:%12.4E%12.4E%12.4E\n", t_jsend, t_jsend/Timesteps*1.e6, t_jsend/t_tot);
            printf("isend:%12.4E%12.4E%12.4E\n", t_isend, t_isend/Timesteps*1.e6, t_isend/t_tot);
            printf("force:%12.4E%12.4E%12.4E\n", t_force, t_force/Timesteps*1.e6, t_force/t_tot);
            printf("recv :%12.4E%12.4E%12.4E\n", t_recv, t_recv/Timesteps*1.e6, t_recv/t_tot);
            printf("comm :%12.4E%12.4E%12.4E\n", t_comm, t_comm/Timesteps*1.e6, t_comm/t_tot);
            printf("corr :%12.4E%12.4E%12.4E\n", t_corr, t_corr/Timesteps*1.e6, t_corr/t_tot);
            printf("tot  :%12.4E%12.4E%12.4E\n", t_tot, t_tot/Timesteps*1.e6, t_tot/t_tot);
        #endif
        fflush(stdout);
	}

	MPI_Finalize();
	return 0;
}