#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <gsl_randist.h>
#include <gsl_rng.h>
#include <sys/time.h>
#include "complex.h"

#define N         6		// number of reaction
#define M         4		// number of chemical species

static void switch_init(int x[]){

	// population of chemical species
	x[0] = 1;
	x[1] = 0;
	x[2] = 0;
	x[3] = 0;

}

static void switch_update_p(double t, double p[], double k12, double k23, double k34, double k41, double k31, double k21, int x[]){

	p[0] = k12*x[0];
	p[1] = k23*x[1];
	p[2] = k34*x[2];
	p[3] = k21*x[1];
	p[4] = k31*x[2];
	p[5] = k41*x[3];

}


static int switch_select_reaction(double p[], int pn, double sum_propencity, double r){
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

static void switch_update_x(int x[], int reaction){
	if (reaction == 0){
	   x[0] = 0; 
	   x[1] = 1;
	   x[2] = 0;
	   x[3] = 0;
	} 
	else if (reaction == 1){
	   x[0] = 0;
	   x[1] = 0;
	   x[2] = 1;
	   x[3] = 0;
	   }
	else if (reaction == 2){
	    x[0] = 0;
	    x[1] = 0;
	    x[2] = 0;
	    x[3] = 1;
	}
	else if (reaction == 3){
	    x[0] = 1;
	    x[1] = 0;
	    x[2] = 0;
	    x[3] = 0;
	    }
	else if (reaction == 4){
	    x[0] = 1;
	    x[1] = 0;
	    x[2] = 0;
	    x[3] = 0;
	    }
	else if (reaction == 5){
	    x[0] = 1;
	    x[1] = 0;
	    x[2] = 0;
	    x[3] = 0;
	    }
}

static double switch_sum(double a[], int n){
	int i;
	double s=0.0;
	for(i=0; i<n; i++) 
		s += a[i];
	return(s);
}

static int photoswitch_ssa(int** x1_ptr, int** x2_ptr, int** x3_ptr, int** x4_ptr, double** times_ptr, int* length, 
                           double end_time, double k12, double k23, double k34, double k41, double k31, double k21){

    *length = 1;
    double* times = (double*) calloc(*length, sizeof(double));
    int* x1 = (int*) calloc(*length, sizeof(int));
    int* x2 = (int*) calloc(*length, sizeof(int));
    int* x3 = (int*) calloc(*length, sizeof(int));
    int* x4 = (int*) calloc(*length, sizeof(int));
    
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
    x4[0] = x[3];
	double p[N];

	while(t < end_time){
 
        (*length)++;
        times = (double*) realloc(times, *length*sizeof(double));
        x1 = (int*) realloc(x1, *length*sizeof(int));
        x2 = (int*) realloc(x2, *length*sizeof(int));
        x3 = (int*) realloc(x3, *length*sizeof(int));
        x4 = (int*) realloc(x4, *length*sizeof(int));

		switch_update_p(t,p,k12,k23,k34,k41,k31,k21,x);
		sum_propencity = switch_sum(p, N);
	    i += 1;

		if(sum_propencity > 0){
			tau = -log((double)rand()/INT_MAX) / sum_propencity;
		}else{
			break;
		}
	
		r = (double)rand()/INT_MAX;
		reaction = switch_select_reaction(p, N, sum_propencity, r);
		switch_update_x(x, reaction);

		x1[i] = x[0];
		x2[i] = x[1];
		x3[i] = x[2];
		x4[i] = x[3];
		t += tau;
		times[i] = t;

	}

    *times_ptr = times;
    *x1_ptr = x1;
    *x2_ptr = x2;
    *x3_ptr = x3;
 	*x4_ptr = x4;
	return(0);
}


static PyObject* photoswitch(PyObject* Py_UNUSED(self), PyObject* args) {

  PyObject* list;

  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list))
    return NULL;
    
    double end_time = PyFloat_AsDouble(PyList_GetItem(list, 0));
    double k12 = PyFloat_AsDouble(PyList_GetItem(list, 1));
    double k23 = PyFloat_AsDouble(PyList_GetItem(list, 2));
    double k34 = PyFloat_AsDouble(PyList_GetItem(list, 3));
    double k41 = PyFloat_AsDouble(PyList_GetItem(list, 4));
    double k31 = PyFloat_AsDouble(PyList_GetItem(list, 5));
    double k21 = PyFloat_AsDouble(PyList_GetItem(list, 6));
 
    double *times = NULL;
    int *x1 = NULL;
    int *x2 = NULL;
    int *x3 = NULL;
    int *x4 = NULL;
    int *length = (int*) calloc(1, sizeof(int));

    photoswitch_ssa(&x1,&x2,&x3,&x4,&times,length,end_time,k12,k23,k34,k41,k31,k21);

	npy_intp dims[1] = {*length};
	PyObject *x1_out = PyArray_SimpleNew(1, dims, NPY_INT);
	PyObject *x2_out = PyArray_SimpleNew(1, dims, NPY_INT);
	PyObject *x3_out = PyArray_SimpleNew(1, dims, NPY_INT);
	PyObject *x4_out = PyArray_SimpleNew(1, dims, NPY_INT);
	PyObject *times_out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    memcpy(PyArray_DATA(x1_out), x1, *length*sizeof(int));
	memcpy(PyArray_DATA(x2_out), x2, *length*sizeof(int));
	memcpy(PyArray_DATA(x3_out), x3, *length*sizeof(int));
	memcpy(PyArray_DATA(x4_out), x4, *length*sizeof(int));
	memcpy(PyArray_DATA(times_out), times, *length*sizeof(double));

	return Py_BuildValue("(OOOOO)", x1_out, x2_out, x3_out, x4_out, times_out);
	free(x1);
	free(x2);
	free(x3);
	free(x4);
	free(times);
    free(length);
}
