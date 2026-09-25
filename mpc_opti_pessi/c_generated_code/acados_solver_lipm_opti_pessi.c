/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "lipm_opti_pessi_model/lipm_opti_pessi_model.h"


#include "lipm_opti_pessi_constraints/lipm_opti_pessi_constraints.h"
#include "lipm_opti_pessi_cost/lipm_opti_pessi_cost.h"



#include "acados_solver_lipm_opti_pessi.h"

#define NX     LIPM_OPTI_PESSI_NX
#define NZ     LIPM_OPTI_PESSI_NZ
#define NU     LIPM_OPTI_PESSI_NU
#define NP     LIPM_OPTI_PESSI_NP
#define NP_GLOBAL     LIPM_OPTI_PESSI_NP_GLOBAL
#define NY0    LIPM_OPTI_PESSI_NY0
#define NY     LIPM_OPTI_PESSI_NY
#define NYN    LIPM_OPTI_PESSI_NYN

#define NBX    LIPM_OPTI_PESSI_NBX
#define NBX0   LIPM_OPTI_PESSI_NBX0
#define NBU    LIPM_OPTI_PESSI_NBU
#define NG     LIPM_OPTI_PESSI_NG
#define NBXN   LIPM_OPTI_PESSI_NBXN
#define NGN    LIPM_OPTI_PESSI_NGN

#define NH     LIPM_OPTI_PESSI_NH
#define NHN    LIPM_OPTI_PESSI_NHN
#define NH0    LIPM_OPTI_PESSI_NH0
#define NPHI   LIPM_OPTI_PESSI_NPHI
#define NPHIN  LIPM_OPTI_PESSI_NPHIN
#define NPHI0  LIPM_OPTI_PESSI_NPHI0
#define NR     LIPM_OPTI_PESSI_NR

#define NS     LIPM_OPTI_PESSI_NS
#define NS0    LIPM_OPTI_PESSI_NS0
#define NSN    LIPM_OPTI_PESSI_NSN

#define NSBX   LIPM_OPTI_PESSI_NSBX
#define NSBU   LIPM_OPTI_PESSI_NSBU
#define NSH0   LIPM_OPTI_PESSI_NSH0
#define NSH    LIPM_OPTI_PESSI_NSH
#define NSHN   LIPM_OPTI_PESSI_NSHN
#define NSG    LIPM_OPTI_PESSI_NSG
#define NSPHI0 LIPM_OPTI_PESSI_NSPHI0
#define NSPHI  LIPM_OPTI_PESSI_NSPHI
#define NSPHIN LIPM_OPTI_PESSI_NSPHIN
#define NSGN   LIPM_OPTI_PESSI_NSGN
#define NSBXN  LIPM_OPTI_PESSI_NSBXN



// ** solver data **

lipm_opti_pessi_solver_capsule * lipm_opti_pessi_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(lipm_opti_pessi_solver_capsule));
    lipm_opti_pessi_solver_capsule *capsule = (lipm_opti_pessi_solver_capsule *) capsule_mem;

    return capsule;
}


int lipm_opti_pessi_acados_free_capsule(lipm_opti_pessi_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int lipm_opti_pessi_acados_create(lipm_opti_pessi_solver_capsule* capsule)
{
    int N_shooting_intervals = LIPM_OPTI_PESSI_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return lipm_opti_pessi_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int lipm_opti_pessi_acados_update_time_steps(lipm_opti_pessi_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "lipm_opti_pessi_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for lipm_opti_pessi_acados_create: step 1
 */
void lipm_opti_pessi_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = NONLINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = NONLINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = NONLINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* lipm_opti_pessi_acados_create_setup_dimensions(lipm_opti_pessi_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 22;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for lipm_opti_pessi_acados_create: step 3
 */
void lipm_opti_pessi_acados_create_setup_functions(lipm_opti_pessi_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        MAP_CASADI_FNC(nl_constr_h_0_fun_jac, lipm_opti_pessi_constr_h_0_fun_jac_uxt_zt);
        MAP_CASADI_FNC(nl_constr_h_0_fun, lipm_opti_pessi_constr_h_0_fun);
        // constraints.constr_type == "BGH" and dims.nh > 0
        capsule->nl_constr_h_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun_jac[i], lipm_opti_pessi_constr_h_fun_jac_uxt_zt);
        }
        capsule->nl_constr_h_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun[i], lipm_opti_pessi_constr_h_fun);
        }
    
        // nonlinear least squares function
        MAP_CASADI_FNC(cost_y_0_fun, lipm_opti_pessi_cost_y_0_fun);
        MAP_CASADI_FNC(cost_y_0_fun_jac_ut_xt, lipm_opti_pessi_cost_y_0_fun_jac_ut_xt);



    
        // discrete dynamics
        capsule->discr_dyn_phi_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun[i], lipm_opti_pessi_dyn_disc_phi_fun);
        }

        capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++)
        {
            MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], lipm_opti_pessi_dyn_disc_phi_fun_jac);
        }

    

    
        // nonlinear least squares cost
        capsule->cost_y_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(cost_y_fun[i], lipm_opti_pessi_cost_y_fun);
        }

        capsule->cost_y_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++)
        {
            MAP_CASADI_FNC(cost_y_fun_jac_ut_xt[i], lipm_opti_pessi_cost_y_fun_jac_ut_xt);
        }
    } // N > 0
    // nonlinear least square function
    MAP_CASADI_FNC(cost_y_e_fun, lipm_opti_pessi_cost_y_e_fun);
    MAP_CASADI_FNC(cost_y_e_fun_jac_ut_xt, lipm_opti_pessi_cost_y_e_fun_jac_ut_xt);

#undef MAP_CASADI_FNC
}


/**
 * Internal function for lipm_opti_pessi_acados_create: step 5
 */
void lipm_opti_pessi_acados_create_set_default_parameters(lipm_opti_pessi_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        lipm_opti_pessi_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for lipm_opti_pessi_acados_create: step 5
 */
void lipm_opti_pessi_acados_setup_nlp_in(lipm_opti_pessi_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        lipm_opti_pessi_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 0.05;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.05;
        cost_scaling[1] = 0.05;
        cost_scaling[2] = 0.05;
        cost_scaling[3] = 0.05;
        cost_scaling[4] = 0.05;
        cost_scaling[5] = 0.05;
        cost_scaling[6] = 0.05;
        cost_scaling[7] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    yref_0[0] = 2;
    yref_0[1] = 2;
    yref_0[2] = 0.7853981633974483;
    yref_0[19] = 2;
    yref_0[20] = 2;
    yref_0[21] = 0.7853981633974483;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 10;
    W_0[1+(NY0) * 1] = 10;
    W_0[2+(NY0) * 2] = 10;
    W_0[3+(NY0) * 3] = 5;
    W_0[4+(NY0) * 4] = 5;
    W_0[5+(NY0) * 5] = 0.5;
    W_0[6+(NY0) * 6] = 0.000001;
    W_0[7+(NY0) * 7] = 10;
    W_0[8+(NY0) * 8] = 10;
    W_0[9+(NY0) * 9] = 10;
    W_0[10+(NY0) * 10] = 10;
    W_0[11+(NY0) * 11] = 500;
    W_0[12+(NY0) * 12] = 500;
    W_0[13+(NY0) * 13] = 500;
    W_0[14+(NY0) * 14] = 500;
    W_0[15+(NY0) * 15] = 0.5;
    W_0[16+(NY0) * 16] = 0.05;
    W_0[17+(NY0) * 17] = 0.05;
    W_0[18+(NY0) * 18] = 10;
    W_0[19+(NY0) * 19] = 0.01;
    W_0[20+(NY0) * 20] = 0.01;
    W_0[21+(NY0) * 21] = 0.01;
    W_0[22+(NY0) * 22] = 0.005;
    W_0[23+(NY0) * 23] = 0.005;
    W_0[24+(NY0) * 24] = 0.0005;
    W_0[25+(NY0) * 25] = 0.000000001;
    W_0[26+(NY0) * 26] = 0.01;
    W_0[27+(NY0) * 27] = 0.01;
    W_0[28+(NY0) * 28] = 0.01;
    W_0[29+(NY0) * 29] = 0.01;
    W_0[30+(NY0) * 30] = 0.5;
    W_0[31+(NY0) * 31] = 0.5;
    W_0[32+(NY0) * 32] = 0.5;
    W_0[33+(NY0) * 33] = 0.5;
    W_0[34+(NY0) * 34] = 0.0005;
    W_0[35+(NY0) * 35] = 0.00005;
    W_0[36+(NY0) * 36] = 0.00005;
    W_0[37+(NY0) * 37] = 0.01;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:
    yref[0] = 2;
    yref[1] = 2;
    yref[2] = 0.7853981633974483;
    yref[19] = 2;
    yref[20] = 2;
    yref[21] = 0.7853981633974483;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 10;
    W[1+(NY) * 1] = 10;
    W[2+(NY) * 2] = 10;
    W[3+(NY) * 3] = 5;
    W[4+(NY) * 4] = 5;
    W[5+(NY) * 5] = 0.5;
    W[6+(NY) * 6] = 0.000001;
    W[7+(NY) * 7] = 10;
    W[8+(NY) * 8] = 10;
    W[9+(NY) * 9] = 10;
    W[10+(NY) * 10] = 10;
    W[11+(NY) * 11] = 500;
    W[12+(NY) * 12] = 500;
    W[13+(NY) * 13] = 500;
    W[14+(NY) * 14] = 500;
    W[15+(NY) * 15] = 0.5;
    W[16+(NY) * 16] = 0.05;
    W[17+(NY) * 17] = 0.05;
    W[18+(NY) * 18] = 10;
    W[19+(NY) * 19] = 0.01;
    W[20+(NY) * 20] = 0.01;
    W[21+(NY) * 21] = 0.01;
    W[22+(NY) * 22] = 0.005;
    W[23+(NY) * 23] = 0.005;
    W[24+(NY) * 24] = 0.0005;
    W[25+(NY) * 25] = 0.000000001;
    W[26+(NY) * 26] = 0.01;
    W[27+(NY) * 27] = 0.01;
    W[28+(NY) * 28] = 0.01;
    W[29+(NY) * 29] = 0.01;
    W[30+(NY) * 30] = 0.5;
    W[31+(NY) * 31] = 0.5;
    W[32+(NY) * 32] = 0.5;
    W[33+(NY) * 33] = 0.5;
    W[34+(NY) * 34] = 0.0005;
    W[35+(NY) * 35] = 0.00005;
    W[36+(NY) * 36] = 0.00005;
    W[37+(NY) * 37] = 0.01;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    yref_e[0] = 2;
    yref_e[1] = 2;
    yref_e[2] = 0.7853981633974483;
    yref_e[7] = 2;
    yref_e[8] = 2;
    yref_e[9] = 0.7853981633974483;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 10;
    W_e[1+(NYN) * 1] = 10;
    W_e[2+(NYN) * 2] = 10;
    W_e[3+(NYN) * 3] = 5;
    W_e[4+(NYN) * 4] = 5;
    W_e[5+(NYN) * 5] = 0.5;
    W_e[6+(NYN) * 6] = 0.000001;
    W_e[7+(NYN) * 7] = 0.01;
    W_e[8+(NYN) * 8] = 0.01;
    W_e[9+(NYN) * 9] = 0.01;
    W_e[10+(NYN) * 10] = 0.005;
    W_e[11+(NYN) * 11] = 0.005;
    W_e[12+(NYN) * 12] = 0.0005;
    W_e[13+(NYN) * 13] = 0.000000001;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &capsule->cost_y_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &capsule->cost_y_fun_jac_ut_xt[i-1]);
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);




    // slacks initial
    double* zlu0_mem = calloc(4*NS0, sizeof(double));
    double* Zl_0 = zlu0_mem+NS0*0;
    double* Zu_0 = zlu0_mem+NS0*1;
    double* zl_0 = zlu0_mem+NS0*2;
    double* zu_0 = zlu0_mem+NS0*3;

    // change only the non-zero elements:
    Zl_0[10] = 1000000;
    Zl_0[21] = 1000000;
    Zu_0[0] = 10000;
    Zu_0[1] = 10000;
    Zu_0[2] = 10000;
    Zu_0[3] = 10000;
    Zu_0[4] = 1000000;
    Zu_0[5] = 1000000;
    Zu_0[6] = 1000000;
    Zu_0[7] = 1000000;
    Zu_0[8] = 1000000;
    Zu_0[9] = 1000000;
    Zu_0[10] = 1000000;
    Zu_0[11] = 10000;
    Zu_0[12] = 10000;
    Zu_0[13] = 10000;
    Zu_0[14] = 10000;
    Zu_0[15] = 1000000;
    Zu_0[16] = 1000000;
    Zu_0[17] = 1000000;
    Zu_0[18] = 1000000;
    Zu_0[19] = 1000000;
    Zu_0[20] = 1000000;
    Zu_0[21] = 1000000;
    zl_0[10] = 100000;
    zl_0[21] = 100000;
    zu_0[0] = 1000;
    zu_0[1] = 1000;
    zu_0[2] = 1000;
    zu_0[3] = 1000;
    zu_0[4] = 100000;
    zu_0[5] = 100000;
    zu_0[6] = 100000;
    zu_0[7] = 100000;
    zu_0[8] = 100000;
    zu_0[9] = 100000;
    zu_0[10] = 100000;
    zu_0[11] = 1000;
    zu_0[12] = 1000;
    zu_0[13] = 1000;
    zu_0[14] = 1000;
    zu_0[15] = 100000;
    zu_0[16] = 100000;
    zu_0[17] = 100000;
    zu_0[18] = 100000;
    zu_0[19] = 100000;
    zu_0[20] = 100000;
    zu_0[21] = 100000;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Zl", Zl_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Zu", Zu_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "zl", zl_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "zu", zu_0);
    free(zlu0_mem);
    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    Zl[0] = 1000;
    Zl[1] = 1000;
    Zl[2] = 1000;
    Zl[3] = 1000;
    Zl[4] = 1000;
    Zl[5] = 1000;
    Zl[16] = 1000000;
    Zl[27] = 1000000;
    Zu[0] = 1000;
    Zu[1] = 1000;
    Zu[2] = 1000;
    Zu[3] = 1000;
    Zu[4] = 1000;
    Zu[5] = 1000;
    Zu[6] = 10000;
    Zu[7] = 10000;
    Zu[8] = 10000;
    Zu[9] = 10000;
    Zu[10] = 1000000;
    Zu[11] = 1000000;
    Zu[12] = 1000000;
    Zu[13] = 1000000;
    Zu[14] = 1000000;
    Zu[15] = 1000000;
    Zu[16] = 1000000;
    Zu[17] = 10000;
    Zu[18] = 10000;
    Zu[19] = 10000;
    Zu[20] = 10000;
    Zu[21] = 1000000;
    Zu[22] = 1000000;
    Zu[23] = 1000000;
    Zu[24] = 1000000;
    Zu[25] = 1000000;
    Zu[26] = 1000000;
    Zu[27] = 1000000;
    zl[0] = 100;
    zl[1] = 100;
    zl[2] = 100;
    zl[3] = 100;
    zl[4] = 100;
    zl[5] = 100;
    zl[16] = 100000;
    zl[27] = 100000;
    zu[0] = 100;
    zu[1] = 100;
    zu[2] = 100;
    zu[3] = 100;
    zu[4] = 100;
    zu[5] = 100;
    zu[6] = 1000;
    zu[7] = 1000;
    zu[8] = 1000;
    zu[9] = 1000;
    zu[10] = 100000;
    zu[11] = 100000;
    zu[12] = 100000;
    zu[13] = 100000;
    zu[14] = 100000;
    zu[15] = 100000;
    zu[16] = 100000;
    zu[17] = 1000;
    zu[18] = 1000;
    zu[19] = 1000;
    zu[20] = 1000;
    zu[21] = 100000;
    zu[22] = 100000;
    zu[23] = 100000;
    zu[24] = 100000;
    zu[25] = 100000;
    zu[26] = 100000;
    zu[27] = 100000;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);



    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;
    idxbx0[13] = 13;
    idxbx0[14] = 14;
    idxbx0[15] = 15;
    idxbx0[16] = 16;
    idxbx0[17] = 17;
    idxbx0[18] = 18;
    idxbx0[19] = 19;
    idxbx0[20] = 20;
    idxbx0[21] = 21;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    lbx0[6] = 0.2;
    ubx0[6] = 0.2;
    lbx0[7] = -0.2;
    ubx0[7] = -0.2;
    lbx0[8] = -0.2;
    ubx0[8] = -0.2;
    lbx0[9] = 0.2;
    ubx0[9] = 0.2;
    lbx0[17] = 0.2;
    ubx0[17] = 0.2;
    lbx0[18] = -0.2;
    ubx0[18] = -0.2;
    lbx0[19] = -0.2;
    ubx0[19] = -0.2;
    lbx0[20] = 0.2;
    ubx0[20] = 0.2;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(22 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    idxbxe_0[13] = 13;
    idxbxe_0[14] = 14;
    idxbxe_0[15] = 15;
    idxbxe_0[16] = 16;
    idxbxe_0[17] = 17;
    idxbxe_0[18] = 18;
    idxbxe_0[19] = 19;
    idxbxe_0[20] = 20;
    idxbxe_0[21] = 21;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);



    // set up nonlinear constraints for last stage
    double* luh_0 = calloc(2*NH0, sizeof(double));
    double* lh_0 = luh_0;
    double* uh_0 = luh_0 + NH0;
    lh_0[12] = -1000000;
    lh_0[13] = -1000000;
    lh_0[15] = -1000000;
    lh_0[16] = -1000000;
    lh_0[17] = -1000000;
    lh_0[18] = -1000000;
    lh_0[19] = -1000000;
    lh_0[20] = -1000000;
    lh_0[26] = -1000000;
    lh_0[27] = -1000000;
    lh_0[29] = -1000000;
    lh_0[30] = -1000000;
    lh_0[31] = -1000000;
    lh_0[32] = -1000000;
    lh_0[33] = -1000000;
    lh_0[34] = -1000000;
    uh_0[8] = 0.03;
    uh_0[9] = 0.03;
    uh_0[10] = 0.03;
    uh_0[11] = 0.03;
    uh_0[14] = 1;
    uh_0[21] = 1000000;
    uh_0[22] = 0.03;
    uh_0[23] = 0.03;
    uh_0[24] = 0.03;
    uh_0[25] = 0.03;
    uh_0[28] = 1;
    uh_0[35] = 1000000;

    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun_jac", &capsule->nl_constr_h_0_fun_jac);
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun", &capsule->nl_constr_h_0_fun);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lh", lh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "uh", uh_0);
    
    
    free(luh_0);






    // set up soft bounds for nonlinear constraints
    int* idxsh_0 = malloc(NSH0 * sizeof(int));
    idxsh_0[0] = 8;
    idxsh_0[1] = 9;
    idxsh_0[2] = 10;
    idxsh_0[3] = 11;
    idxsh_0[4] = 15;
    idxsh_0[5] = 16;
    idxsh_0[6] = 17;
    idxsh_0[7] = 18;
    idxsh_0[8] = 19;
    idxsh_0[9] = 20;
    idxsh_0[10] = 21;
    idxsh_0[11] = 22;
    idxsh_0[12] = 23;
    idxsh_0[13] = 24;
    idxsh_0[14] = 25;
    idxsh_0[15] = 29;
    idxsh_0[16] = 30;
    idxsh_0[17] = 31;
    idxsh_0[18] = 32;
    idxsh_0[19] = 33;
    idxsh_0[20] = 34;
    idxsh_0[21] = 35;
    double* lush_0 = calloc(2*NSH0, sizeof(double));
    double* lsh_0 = lush_0;
    double* ush_0 = lush_0 + NSH0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxsh", idxsh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lsh", lsh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ush", ush_0);
    free(idxsh_0);
    free(lush_0);



    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 4;
    idxbu[1] = 5;
    idxbu[2] = 6;
    idxbu[3] = 7;
    idxbu[4] = 8;
    idxbu[5] = 9;
    idxbu[6] = 10;
    idxbu[7] = 15;
    idxbu[8] = 16;
    idxbu[9] = 17;
    idxbu[10] = 18;
    idxbu[11] = 19;
    idxbu[12] = 20;
    idxbu[13] = 21;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = 0.1;
    ubu[0] = 0.9;
    lbu[1] = -100;
    ubu[1] = 100;
    lbu[2] = -100;
    ubu[2] = 100;
    lbu[3] = 0.05;
    ubu[3] = 0.0875;
    lbu[4] = -1;
    ubu[4] = 1;
    lbu[5] = -1;
    ubu[5] = 1;
    lbu[6] = -10;
    ubu[6] = 10;
    lbu[7] = 0.1;
    ubu[7] = 0.9;
    lbu[8] = -100;
    ubu[8] = 100;
    lbu[9] = -100;
    ubu[9] = 100;
    lbu[10] = 0.05;
    ubu[10] = 0.0875;
    lbu[11] = -1;
    ubu[11] = 1;
    lbu[12] = -1;
    ubu[12] = 1;
    lbu[13] = -10;
    ubu[13] = 10;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);






    /* Path constraints */

    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 3;
    idxbx[1] = 4;
    idxbx[2] = 5;
    idxbx[3] = 10;
    idxbx[4] = 14;
    idxbx[5] = 15;
    idxbx[6] = 16;
    idxbx[7] = 21;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    lbx[0] = -0.6;
    ubx[0] = 0.6;
    lbx[1] = -0.3;
    ubx[1] = 0.3;
    lbx[2] = -0.8;
    ubx[2] = 0.8;
    ubx[3] = 100;
    lbx[4] = -0.6;
    ubx[4] = 0.6;
    lbx[5] = -0.3;
    ubx[5] = 0.3;
    lbx[6] = -0.8;
    ubx[6] = 0.8;
    ubx[7] = 100;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);


    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;
    lh[4] = -1000000;
    lh[5] = -1000000;
    lh[7] = -1000000;
    lh[8] = -1000000;
    lh[9] = -1000000;
    lh[10] = -1000000;
    lh[11] = -1000000;
    lh[12] = -1000000;
    lh[18] = -1000000;
    lh[19] = -1000000;
    lh[21] = -1000000;
    lh[22] = -1000000;
    lh[23] = -1000000;
    lh[24] = -1000000;
    lh[25] = -1000000;
    lh[26] = -1000000;
    uh[0] = 0.03;
    uh[1] = 0.03;
    uh[2] = 0.03;
    uh[3] = 0.03;
    uh[6] = 1;
    uh[13] = 1000000;
    uh[14] = 0.03;
    uh[15] = 0.03;
    uh[16] = 0.03;
    uh[17] = 0.03;
    uh[20] = 1;
    uh[27] = 1000000;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "uh", uh);
        
        
    }
    free(luh);







    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxsbx", idxsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lsbx", lsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "usbx", usbx);

    // soft bounds on x
    int* idxsbx = malloc(NSBX * sizeof(int));
    idxsbx[0] = 0;
    idxsbx[1] = 1;
    idxsbx[2] = 2;
    idxsbx[3] = 4;
    idxsbx[4] = 5;
    idxsbx[5] = 6;

    double* lusbx = calloc(2*NSBX, sizeof(double));
    double* lsbx = lusbx;
    double* usbx = lusbx + NSBX;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxsbx", idxsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lsbx", lsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "usbx", usbx);
    }
    free(idxsbx);
    free(lusbx);


    // set up soft bounds for nonlinear constraints
    int* idxsh = malloc(NSH * sizeof(int));
    idxsh[0] = 0;
    idxsh[1] = 1;
    idxsh[2] = 2;
    idxsh[3] = 3;
    idxsh[4] = 7;
    idxsh[5] = 8;
    idxsh[6] = 9;
    idxsh[7] = 10;
    idxsh[8] = 11;
    idxsh[9] = 12;
    idxsh[10] = 13;
    idxsh[11] = 14;
    idxsh[12] = 15;
    idxsh[13] = 16;
    idxsh[14] = 17;
    idxsh[15] = 21;
    idxsh[16] = 22;
    idxsh[17] = 23;
    idxsh[18] = 24;
    idxsh[19] = 25;
    idxsh[20] = 26;
    idxsh[21] = 27;
    double* lush = calloc(2*NSH, sizeof(double));
    double* lsh = lush;
    double* ush = lush + NSH;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ush", ush);
    }
    free(idxsh);
    free(lush);



    /* terminal constraints */




















}


static void lipm_opti_pessi_acados_create_set_opts(lipm_opti_pessi_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 7;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for lipm_opti_pessi_acados_create: step 7
 */
void lipm_opti_pessi_acados_set_nlp_out(lipm_opti_pessi_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    x0[6] = 0.2;
    x0[7] = -0.2;
    x0[8] = -0.2;
    x0[9] = 0.2;
    x0[17] = 0.2;
    x0[18] = -0.2;
    x0[19] = -0.2;
    x0[20] = 0.2;


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for lipm_opti_pessi_acados_create: step 9
 */
int lipm_opti_pessi_acados_create_precompute(lipm_opti_pessi_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int lipm_opti_pessi_acados_create_with_discretization(lipm_opti_pessi_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != LIPM_OPTI_PESSI_N && !new_time_steps) {
        fprintf(stderr, "lipm_opti_pessi_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, LIPM_OPTI_PESSI_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    lipm_opti_pessi_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = lipm_opti_pessi_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    lipm_opti_pessi_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    lipm_opti_pessi_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    lipm_opti_pessi_acados_create_setup_functions(capsule);
    lipm_opti_pessi_acados_setup_nlp_in(capsule, N, new_time_steps);
    lipm_opti_pessi_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = lipm_opti_pessi_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int lipm_opti_pessi_acados_update_qp_solver_cond_N(lipm_opti_pessi_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from lipm_opti_pessi_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = lipm_opti_pessi_acados_create_precompute(capsule);
    return status;
}


int lipm_opti_pessi_acados_reset(lipm_opti_pessi_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int lipm_opti_pessi_acados_update_params(lipm_opti_pessi_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 8;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int lipm_opti_pessi_acados_update_params_sparse(lipm_opti_pessi_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int lipm_opti_pessi_acados_set_p_global_and_precompute_dependencies(lipm_opti_pessi_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, lipm_opti_pessi_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int lipm_opti_pessi_acados_solve(lipm_opti_pessi_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int lipm_opti_pessi_acados_setup_qp_matrices_and_factorize(lipm_opti_pessi_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int lipm_opti_pessi_acados_free(lipm_opti_pessi_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        
    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);
  
  

    // cost
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_external_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun_jac_ut_xt);

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_0_fun_jac);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_0_fun);



    return 0;
}


void lipm_opti_pessi_acados_print_stats(lipm_opti_pessi_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[1616];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int lipm_opti_pessi_acados_custom_update(lipm_opti_pessi_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *lipm_opti_pessi_acados_get_nlp_in(lipm_opti_pessi_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *lipm_opti_pessi_acados_get_nlp_out(lipm_opti_pessi_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *lipm_opti_pessi_acados_get_sens_out(lipm_opti_pessi_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *lipm_opti_pessi_acados_get_nlp_solver(lipm_opti_pessi_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *lipm_opti_pessi_acados_get_nlp_config(lipm_opti_pessi_solver_capsule* capsule) { return capsule->nlp_config; }
void *lipm_opti_pessi_acados_get_nlp_opts(lipm_opti_pessi_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *lipm_opti_pessi_acados_get_nlp_dims(lipm_opti_pessi_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *lipm_opti_pessi_acados_get_nlp_plan(lipm_opti_pessi_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
