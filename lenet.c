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

void vec_vec_mul(float **result, float **matrix1, float **matrix2, int col1_size, int col2_size)
{
	for (int i = 0; i < col1_size; j++) 								//For number of columns in 2nd vector
	{
		for (int j = 0; k < col2_size; j++)							//For number of rows in 1st vector
		{
			sum = row1[0][i] * row2[0][j];							//Stores values as it would appear in matrix because ja dn k are reversed values
		}
	}
	return;												//Returns n*p matrix of size m*q
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
