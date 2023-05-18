#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include "complex.h"

#define N         4		// number of reaction
#define M         3		// number of chemical species

static void init(int x[]){

	// population of chemical species
	x[0] = 1;
	x[1] = 0;
	x[2] = 0;

}

static void update_p(double t, double p[], double k_on, double k_off, double k_syn, double k_deg, int x[]){

	p[0] = k_on*x[0];
	p[1] = k_off*x[1];
	p[2] = k_syn*x[1];
	p[3] = k_deg*x[2];
	//printf("%f, %f, %f, %f\n",p[0],p[1],p[2],p[3]);
}


static int select_reaction(double p[], int pn, double sum_propencity, double r){
	int reaction = -1;
	double sp = 0.0;
	int i;
	r = r * sum_propencity;
	for(i=0; i<pn; i++){
		sp += p[i];
		if(r < sp){
			reaction = i;
			break;
		}
	}
	return reaction;
}

static void update_x(int x[], int reaction){
	if (reaction == 0){
	   x[0] = 0; 
	   x[1] = 1;
	} 
	else if (reaction == 1){
	   x[0] = 1;
	   x[1] = 0;
	   }
	else if (reaction == 2){
	    x[2] = x[2] + 1;
	}
	else if (reaction == 3){
	    x[2] = x[2] - 1;
	    if (x[2] < 0){
	       x[2] = 0;
	       }
	    }

}

static double sum(double a[], int n){
	int i;
	double s=0.0;
	for(i=0; i<n; i++) 
		s += a[i];
	return(s);
}

static int telegraph_constant_ssa(int** x1_ptr, int** x2_ptr, int** x3_ptr, double** times_ptr, int* length, 
                           double end_time, double k_on, double k_off, double k_syn, double k_deg){

    *length = 1;
    double* times = (double*) calloc(*length, sizeof(double));
    int* x1 = (int*) calloc(*length, sizeof(int));
    int* x2 = (int*) calloc(*length, sizeof(int));
    int* x3 = (int*) calloc(*length, sizeof(int));
    
	double sum_propencity = 0.0;
	double tau=0.0;
	double t=0.0;
	double r;
	int reaction;

    int i = 0;
    time_t current_time = time(NULL);
    double seed = (double)current_time + (double)clock() / CLOCKS_PER_SEC;
    srand((unsigned int)(seed * 1000000.0));

	int x[M];
	init(x);
    x1[0] = x[0];
    x2[0] = x[1];
    x3[0] = x[2];
	double p[N];

	while(t < end_time){
 
        (*length)++;
        times = (double*) realloc(times, *length*sizeof(double));
        x1 = (int*) realloc(x1, *length*sizeof(int));
        x2 = (int*) realloc(x2, *length*sizeof(int));
        x3 = (int*) realloc(x3, *length*sizeof(int));

		update_p(t, p, k_on, k_off, k_syn, k_deg, x);
		sum_propencity = sum(p, N);
	    i += 1;

		if(sum_propencity > 0){
			tau = -log((double)rand()/INT_MAX) / sum_propencity;
		}else{
			break;
		}
	
		r = (double)rand()/INT_MAX;
		reaction = select_reaction(p, N, sum_propencity, r);
		update_x(x, reaction);

		x1[i] = x[0];
		x2[i] = x[1];
		x3[i] = x[2];
		t += tau;
		times[i] = t;


	}

    *times_ptr = times;
    *x1_ptr = x1;
    *x2_ptr = x2;
    *x3_ptr = x3;
 	
	return(0);
}


static PyObject* telegraph_constant(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;
    
    double end_time = PyFloat_AsDouble(PyList_GetItem(list, 0));
    double k_on = PyFloat_AsDouble(PyList_GetItem(list, 1));
    double k_off = PyFloat_AsDouble(PyList_GetItem(list, 2));
    double k_syn = PyFloat_AsDouble(PyList_GetItem(list, 3));
    double k_deg = PyFloat_AsDouble(PyList_GetItem(list, 4));
 
    double *times = NULL;
    int *x1 = NULL;
    int *x2 = NULL;
    int *x3 = NULL;
    int *length = (int*) calloc(1, sizeof(int));

    telegraph_constant_ssa(&x1,&x2,&x3,&times,length,end_time,k_on,k_off,k_syn,k_deg);

	npy_intp dims[1] = {*length};
	PyObject *x1_out = PyArray_SimpleNew(1, dims, NPY_INT);
	PyObject *x2_out = PyArray_SimpleNew(1, dims, NPY_INT);
	PyObject *x3_out = PyArray_SimpleNew(1, dims, NPY_INT);
	PyObject *times_out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    memcpy(PyArray_DATA(x1_out), x1, *length*sizeof(int));
	memcpy(PyArray_DATA(x2_out), x2, *length*sizeof(int));
	memcpy(PyArray_DATA(x3_out), x3, *length*sizeof(int));
	memcpy(PyArray_DATA(times_out), times, *length*sizeof(double));

	return Py_BuildValue("(OOOO)", x1_out, x2_out, x3_out, times_out);
	free(x1);
	free(x2);
	free(x3);
	free(times);
    free(length);
}
