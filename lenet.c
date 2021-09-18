#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

void mat_add(float **result, float **mat1, float **mat2, int m, int n)
{	
	//float **result;
	int i, j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			result[i][j] = mat1[i][j] + mat2[i][j];
	return;
}

int main() 
{	
	float **matrix1, **matrix2, **sum;
	int m = 3, n = 4, i, j;
	//mat1
	matrix1 = (float **)malloc(m * sizeof(float*));
	for(int i = 0; i < m; i++)
		*(matrix1 + i) = (float *)malloc(n * sizeof(float));
	//mat2	
	matrix2 = (float **)malloc(m * sizeof(float*));
	for(int i = 0; i < m; i++)
		*(matrix2 + i) = (float *)malloc(n * sizeof(float));
	sum = (float **)malloc(m * sizeof(float*));
	for(int i = 0; i < m; i++)
		*(sum + i) = (float *)malloc(n * sizeof(float));	
	//input random values to matrix
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			matrix1[i][j] = ((uint8_t)rand())%256;
			matrix2[i][j] = ((uint8_t)rand())%256;
		}	
	}
	for(int i = 0; i < m; i++)
	{
		free(matrix1[i]);
		free(matrix2[i]);	
		free(sum[i]);
	}	
	free(matrix1);
	free(matrix2);
	free(sum);
}