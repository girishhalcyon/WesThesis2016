/**
 * Simulation Archive
 *
 * This example shows how to use the Simulation Archive.
 * We integrate a two planet system forward in time using
 * the WHFast integrator. The simulation can be interrupted
 * at any time. On the next run, the program will try to reload
 * the latest data from the Simulation Archive.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <omp.h>
#include "rebound.h"
#include "reboundx.h"

void heartbeat(struct reb_simulation* r);

int main(int argc, char* argv[]) {
    int np = 20;
    omp_set_num_threads(np);
    FILE *fp2;
    char hash_name[] = "hash.csv";
    fp2 = fopen(hash_name, "w+");
    fprintf(fp2, "Hash, Mass, Radius\n");
    char filename[512] = "veras_no_frag.bin";
    // Trying to restart from the Simulation Archive.
    struct reb_simulation* r = reb_create_simulation_from_simulationarchive(filename);
    printf("Loaded Simulation Successfully\n");
    printf("Time is: %.16f\n", r->t);
    printf("N_active is: %i\n", r->N);
    printf("Timestep is: %.3f\n", r->dt);
    r->heartbeat	= heartbeat;
    r->dt = 10.0;
    r->gravity = REB_GRAVITY_TREE;
    r->integrator = REB_INTEGRATOR_WHFAST;
    r->collision = REB_COLLISION_TREE;
    r->boundary     = REB_BOUNDARY_OPEN;
    const double boxsize = 2.5e10; // about 0.15 AU, with fragments at 0.0054 AU
    reb_configure_box(r,boxsize,1,1,1);
    struct rebx_extras* rebx = rebx_init(r);
    /*
    struct rebx_effect* rad_params = rebx_add(rebx, "radiation_forces");
    double* c = rebx_add_param(rad_params, "c", REBX_TYPE_DOUBLE);
    *c = 3.e8;                          // speed of light in SI units    */


    struct reb_particle* wd = reb_get_particle_by_hash(r, reb_hash("wd"));
    int wd_index = reb_get_particle_index(wd);
    /*
    int* rad_source = rebx_add_param(&r->particles[wd_index], "radiation_source", REBX_TYPE_INT);
    *rad_source = 1;    */


    struct rebx_effect* gr_params = rebx_add(rebx, "gr");
    double* c_2 = rebx_add_param(gr_params, "c", REBX_TYPE_DOUBLE);
    *c_2 = 3.e8;
    int* source = rebx_add_param(&r->particles[wd_index], "gr_source", REBX_TYPE_INT);
    *source = 1;
    double wd_rad = wd->r;
    double disk_rad_in = 10.0*wd_rad;
    double disk_rad_out = 90.0*wd_rad;
    printf("Inner disk radius is: %f AU\n", disk_rad_in/1.496e11);
    printf("Outer disk radius is: %f AU\n", disk_rad_out/1.496e11);
    int N_disk = 2000;
    double disk_inc_low = 0.0;
    double disk_inc_high = 0.0;
    double e_low = 0.0;
    double e_high = 0.0;
    r->N_active = r->N;
    double* add_beta;
    while(r->N<N_disk + r->N_active){
        struct reb_particle pt = {0};
        double a = reb_random_uniform(disk_rad_in, disk_rad_out);
        double e = reb_random_uniform(e_low, e_high);
        double inc = reb_random_uniform(disk_inc_low, disk_inc_high);
        double Omega = reb_random_uniform(0, 2.*M_PI);
        double apsis = reb_random_uniform(0, 2.*M_PI);
        double phi = reb_random_uniform(0.0, 2.*M_PI);
        pt = reb_tools_orbit_to_particle(r->G, r->particles[wd_index], 0.0, a, e, inc, Omega, apsis, phi);
        int N_count = r->N - r->N_active;
        pt.hash = N_count + 8000;
        reb_add(r, pt);
        add_beta = rebx_add_param(&r->particles[N_count], "beta", REBX_TYPE_DOUBLE);
        *add_beta = 0.01;
    }
    r->simulationarchive_interval = 6.32e6;
    char sim_name[1000];
    filename[strlen(filename) - 4] = 0;
    sprintf(sim_name, "%s_months.bin", filename);
    r->simulationarchive_filename = sim_name;

    for (int i=0; i < r->N; i=i+1){
        fprintf(fp2, "%u, %f, %f\n", r->particles[i].hash, r->particles[i].m, r->particles[i].r);
     }
    fclose(fp2);
    reb_integrate(r, 1.6e8); // ~5 years
    rebx_free(rebx);
}


void heartbeat(struct reb_simulation* r){
    double P_output = 2.628e6;
    double P_light = 2.16e4;
    double test = r->t/P_output;
    double floor_test = floor(test);
    double temp_test = test - floor_test;
    double ratio = P_light/P_output;
    printf("Time is: %.6f days\n", r->t/86400.0);
    if (temp_test <= ratio){
        FILE *fp;
        char f_name[100];
        char Buffer[1300000];
        int temp_name = floor_test;
        // double save_name = (r->t - floor_test*P_output)/r->dt;
        sprintf(f_name,"%i_months.csv", temp_name);
        fp = fopen(f_name, "a+");
        snprintf(Buffer, strlen(Buffer)+200,"Hash, x, y, z\n");
        printf("N=%i\n", r->N);
        for (int i=0; i < r->N; i++){
            sprintf(strlen(Buffer)+Buffer, "%u, %.16f, %.16f, %.16f\n", r->particles[i].hash, r->particles[i].x, r->particles[i].y, r->particles[i].z);
    }
        fprintf(fp, Buffer);
        fclose(fp);
}
        //printf("dt=%.16f\n", r->dt);
}
