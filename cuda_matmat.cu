#include <stdio.h>
#include <stdlib.h>
extern "C" {
	#include "c_timer.h"
}

int main(int argc, char* argv[]) {
	//PROTOTIPI DI FUNZIONE
	__global__ void matrixMult (float *, float *, float *, int , int, int, int, int);
	void printMatrix(float *, const char[], int, int);

	//DICHIARAZIONE VARIABILI
	float *h_A=NULL, *h_B=NULL, *h_C=NULL, *d_A=NULL, *d_B=NULL, *d_C=NULL;
	int N=0, M=0, P=0, ntrow=0, ntcol=0, i=0;
	double inizio, fine;

	//CONTROLLI SUI PARAMETRI IN INGRESSO
	if(argc != 6){
		printf("Errore: eseguire nel seguente modo [N][M][P][ntrow][ntcol] ");
		return 0;
	}
	if(atoi(argv[1]) % atoi(argv[4]) != 0 || atoi(argv[3]) % atoi(argv[5]) != 0){
		printf("Errore: La matrice non puo' essere divisa in una griglia di %dx%d blocchi di %dx%d thread",(int)ceil(atoi(argv[1])/atoi(argv[4])),(int)ceil(atoi(argv[3])/atoi(argv[5])),atoi(argv[4]),atoi(argv[5]));
		return 0;
	}

	N = atoi(argv[1]);
	M = atoi(argv[2]);
	P = atoi(argv[3]);
	ntrow = atoi(argv[4]);
	ntcol = atoi(argv[5]);

	//INIZIALIZZAZIONE DELLE MATRICI
	h_A = (float *) malloc(N * M * sizeof(float));
	h_B = (float *) malloc(M * P * sizeof(float));
	h_C = (float *) malloc(N * P * sizeof(float));
	cudaMalloc((void **) &d_A, N * M * sizeof(float));
	cudaMalloc((void **) &d_B, M * P * sizeof(float));
	cudaMalloc((void **) &d_C, N * P * sizeof(float));

	for(i=0; i < N * M; i++)
		*(h_A + i) = i+1;
	for(i=0; i < M * P; i++)
		*(h_B + i) = i+1;
	for(i=0; i < N * P; i++)
		*(h_C + i) = 0;

	//COPIA DELLE MATRICI SULLA GPU
	cudaMemcpy(d_A, h_A, N * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, M * P * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, N * P * sizeof(float), cudaMemcpyHostToDevice);

	//DEFINZIONE DIMENSIONI DEI BLOCCHI
	dim3 dimBlock(ntcol, ntrow);
	dim3 dimGrid((int)ceil(P/dimBlock.x), (int)ceil(N/dimBlock.y));

	//ESECUZIONE DELL'ALGORITMO CON IL CALCOLO DEL TEMPO
	inizio = get_cur_time();
		matrixMult<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, M, P, ntrow, ntcol);
    cudaDeviceSynchronize();
	fine = get_cur_time();

	printf("GPU Computation Time: %lfs\n", fine - inizio);
	printf("Performance: %e\n",((double)2*N*M*P)/(fine-inizio));

	cudaMemcpy(h_C, d_C, N * P * sizeof(float), cudaMemcpyDeviceToHost);

	//STAMPA DELLE MATRICI
	//printMatrix(h_A, "A", N, M);
	//printf("\n");
	//printMatrix(h_B, "B", M, P);
	//printf("\n");
	//printMatrix(h_C, "C", N, P);
	//printf("\n");

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

__global__ void matrixMult(float* A, float* B, float* C,int N, int M, int P, int ntrow,int ntcol){
	int k, temp = 0;
	int col = blockIdx.x*ntcol + threadIdx.x;
	int row = blockIdx.y*ntrow + threadIdx.y;

	if(col < P && row < N) {
		for (k = 0; k < M; k++)
			temp += A[row * M + k] * B[k * P + col];
		C[row * P + col] = temp;
		//printf("%d|%d %d|%d C(%d,%d):%.2f\n",blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x, row, col, C[row*P+col]);
	}
} 

void printMatrix(float *M, const char name[], int row, int col) {
	int i;
	printf("%s:", name);
	for(i=0; i < row * col; i++) {
		if(i % col == 0) printf("\n");
		printf("%.2f ", *(M + i));
	}
	printf("\n");
}