/*
 * nnn.cpp
 *
 *  Created on: 15-Jul-2021
 *      Author: gaurav
 */
/*
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <bits/stdc++.h>
#include<ctime>
#include<math.h>
#include<malloc.c>
using namespace std;
typedef vector<float> Row;
typedef vector<Row> Matrixx;

//To read pixel values
Matrixx read_pixel(string file)
{
	string line;                    // string to hold each line
	Row label_val;
	Matrixx pixel;      // vector of vector<int> for 2d array
	int count=0;

	ifstream f (file);
	if (!f.is_open()) {     // validate file open for reading
		cout << "error opening file";
		exit(0);
	}

	while (getline (f, line)) // read each line
	{
		if(count != 0)						//To ignore first row containing unnecessary info
		{
			string val;                     // string to hold value
			int counter = 0;
			vector<float> pixel_row;        // vector for row of values
			stringstream s (line);          // stringstream to parse csv
			while (getline (s, val, ','))   // for each value
			{
				if(counter != 0)
					pixel_row.push_back (stof(val));  //pushes pixel values into "pixel_row"
				counter++;
			}
			pixel.push_back (pixel_row);          // add row to pixel array
		}
		count++;
	}
	f.close();
	return pixel;
}

//Use to intialize Weight matrix
Matrixx initialize_matrix(int row_size, int col_size)
{
	// specify the default value to fill the vector elements

	float rand_val = 0, extra_val, final_val;
	Matrixx result;
	srand(time(NULL)); //picks random time
	for (int i = 0; i < row_size; i++)
	{
		// construct a vector of floats with the given default value
		Row row;
		for (int j = 0; j < col_size; j++) {
			extra_val = (float)rand() / (float)RAND_MAX;
			final_val = rand_val + extra_val;
			row.push_back(final_val);
		}

		// push back above one-dimensional vector
		result.push_back(row);
	}
	return result;
}

//To normalize Matrix
Matrixx normalize_matrix(Matrixx m)
{
	float temp = 0;
	Matrixx result;
	for(int i = 0; i < m.size(); i++)
	{
		Row row2;										//Always intialize the Row vector in a 'for' loop before it is used
		for(int j = 0; j < m[i].size(); j++)
		{
			temp = m[i][j]/255;							//Divide it by 255 to get pixel values in normalized form
			row2.push_back(temp);
		}
		result.push_back(row2);
	}
	return result;
}

// Matrix Multiplication
Matrixx mul_matrix(Matrixx& matrix1, Matrixx& matrix2)
{
	int n = matrix1.size(); //row size for matrix 1
	int m = matrix1[0].size();//col size for matrix 1
	int q = matrix2.size();//row size for matrix 2
	int p = matrix2[0].size();//col size for matrix 2
	Matrixx result(n, vector<float>(p));
	if(m != q)
	{
		cout << "Invalid size for mul_matrix \n";						//Get all dimensions so multiplication is correct
		exit(0);
	}
	else{

		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < p; ++j)
			{
				float sum = 0;
				for (int k = 0; k < m; ++k)
				{
					sum = sum + matrix1[i][k] * matrix2[k][j];
				}
				result[i][j] = sum; 					//Returns n*p matrix for matices of size n*m and q*p
			}
		}
	}
	return result;
}

//Matrix vector multiplication
Matrixx mat_vec_mul(Row& row1, Row& row2)
{
	int m = row1.size();										//row1 size for vector
	int q = row2.size();										//row2 size for vector
	Matrixx result(m, vector<float>(q));
	for (int j = 0; j < q; j++) 								//For number of columns in 2nd vector
	{
		float sum = 0;
		for (int k = 0; k < m; k++)								//For number of rows in 1st vector
		{
			sum = row1[j] * row2[k];
			result[j][k] = sum;									//Stores values as it would appear in matrix because ja dn k are reversed values
		}
	}
	return result;												//Returns n*p matrix of size m*q
}

//Transpose matrix
Matrixx transpose_matrix(Matrixx& matrix)
{
	int row_size = matrix.size();
	Matrixx tpose(matrix[0].size(),vector<float>());
	for (int i = 0; i < row_size; i++)
	{
		for (int j = 0; j < matrix[i].size(); j++)
		{
			tpose[j].push_back(matrix[i][j]);
		}
	}
	return tpose;
}

//addition of matices
Matrixx sum_matrix(Matrixx& matrix1,Matrixx& matrix2)
{
	int arows = matrix1.size();
	int acols = matrix1[0].size();
	int brows = matrix2.size();
	int bcols = matrix2[0].size();								//Read and store all size values to check matrix dimensions
	Matrixx result(arows, Row(acols));
	if(arows!=brows || acols!=bcols)							//If dimensions arent the same, exit
	{
		cout<<"Invalid size for sum_matrix \n";
	}
	else
	{
		for (int j = 0; j < acols; ++j)
		{
			for (int i = 0; i < arows; ++i)
			{
				result[i][j] = matrix1[i][j] + matrix2[i][j];				//Addition function
			}
		}
	}
	return result;
}

//addition of matices
Matrixx sub_matrix(Matrixx& matrix1,Matrixx& matrix2)
{
	int arows = matrix1.size();
	int acols = matrix1[0].size();
	int brows = matrix2.size();
	int bcols = matrix2[0].size();								//Read and store all size values to check matrix dimensions
	Matrixx result(arows, Row(acols));
	if(arows!=brows || acols!=bcols)							//If dimensions arent the same, exit
	{
		cout<<"Invalid size for sub)matrix \n";
	}
	else
	{
		for (int j = 0; j < acols; ++j)
		{
			for (int i = 0; i < arows; ++i)
			{
				result[i][j] = fabs(matrix1[i][j] - matrix2[i][j]);				//Subtraction function
			}
		}
	}
	return result;
}

//To perform one hot encoding for MNIST dataset labels and store all the vectors in another vector
Matrixx hotencode(Row label)
{
	Matrixx hotvecs;									//To store rows of one hot encoded vectors
	Row mnist {0,1,2,3,4,5,6,7,8,9};					//Only 10 output values are possible hence define them prior
	for(int i = 0; i < label.size(); i++)				//Check label size, usually 60,000 in MNIST
	{
		Row onehot;										//To store the one hot encoded values in rows
		for (int j = 0; j < mnist.size(); j++)
		{
			if (label[i] == mnist[j])
			{
				onehot.push_back(0.99);					//push 0.99 if the values match
			}
			else
			{
				onehot.push_back(0.011);				//push 0.011 if values dont match
			}
		}
		hotvecs.push_back(onehot);
	}
	return hotvecs;
}

//Use to update weight
Matrixx update_weight(Matrixx& weight_old, Matrixx& err_weight, float learn_rate)
{
	int arows = weight_old.size();
	int acols = weight_old[0].size();
	Matrixx weight_upd(arows, Row(acols));
	for (int j = 0; j < acols; ++j)
	{
		for (int i = 0; i < arows; ++i)
		{
			weight_upd[i][j] = weight_old[i][j] - (learn_rate * err_weight[i][j]); //Upd weight = Old weight - LR*New weight
		}
	}
	return weight_upd;
}

//To read labels
Row read_label(string file)
{
	string line;             // string to hold each line
	Row label_val;
	int count=0;

	ifstream f (file);
	if (!f.is_open()) {     // validate file open for reading
		cout << "error opening file";
		exit(0);
	}

	while (getline (f, line)) // read each line
	{
		if(count != 0)
		{
			string val;
			int counter = 0;
			vector<int> row;                		// vector for row of values
			stringstream s (line);                  // stringstream to parse csv
			while (getline (s, val, ','))			// for each value
			{
				if(counter == 0)
					label_val.push_back (stof(val));  // convert to float, add to row and pushes label value
				counter++;
			}
		}
		count++;
	}
	f.close();
	return label_val;
}

//Use to initialise bias vector
Row initialize_bias(int col_size)

{
	// specify the default value to fill the vector elements
	float rand_val = 0, extra_val, final_val_prev, final_val;
	srand(time(NULL));								//Compute random time
	Row row;
	for (int i = 0; i < col_size; i++) {
		extra_val = (float)rand() / (float)RAND_MAX;//For particular time, compute float value
		final_val_prev = rand_val + extra_val;
		final_val = final_val_prev/10;
		row.push_back(final_val);
	}
	return row;
}

//To normalize vector
Row normalize_vector(Row row)
{
	float temp = 0;
	Row row2;										//Always intialize the Row vector in a 'for' loop before it is used
	for(int i = 0; i < row.size(); i++)
	{
		temp = row[i]/255;							//Divide it by 255 to get pixel values in normalized form
		row2.push_back(temp);
	}
	return row2;
}

//Subtract vector
Row sub_vector(Row& vector1,Row& vector2)
{
	Row result_vec(vector1.size());
	for(int i = 0; i < vector1.size(); i++)
	{
		result_vec[i] = fabs(vector1[i] - vector2[i]);
	}
	return result_vec;
}

//Add vectors
Row sum_vector(Row& vector1,Row& vector2)
{
	Row result_vec(vector1.size());
	for(int i = 0; i < vector1.size(); i++)
	{
		result_vec[i] = vector1[i] + vector2[i];
	}
	return result_vec;
}

//sigmoid fucntion
Row sigmoid(Row row)
{
	Row result_vec(row.size());
	for(int i = 0; i < row.size(); i++)
	{
		result_vec[i] = (1.0/(1.0 + exp(-row[i])));
	}
	return result_vec;
}

//Hadamard product
Row hadamard(Row row1, Row row2)				//Used to multiply 2 vectos of the same size
{
	Row result_vec(row1.size());
	for(int i = 0; i < row1.size(); i++)
	{
		result_vec[i] = row1[i] * row2[i];
	}
	return result_vec;
}

//gradient sigmoid
Row sigmoid_derivative(Row row)
{
	Row result_vec(row.size());
	float sig;
	for(int i = 0; i < row.size(); i++)
	{
		sig = (1.0/(1.0 + exp(-row[i])));
		result_vec[i] = (sig*(1-sig));			//sigmoid(x)*[1-sigmoid(x)]-derivative
	}
	return result_vec;
}

//Relu function
Row relu(Row row)
{
	Row result_vec(row.size());
	for(int i = 0; i < row.size(); i++)
	{
		if (row[i] > 0)
		{
			result_vec[i] = row[i];
		}
		else
			result_vec[i] = 0;
	}
	return result_vec;
}

//Relu derivate
Row relu_derivative(Row row)
{
	Row result_vec(row.size());
	for(int i = 0; i < row.size(); i++)
	{
		if (row[i] > 0 )
		{
			result_vec[i] = 1;
		}
		else
			result_vec[i] = 0;
	}
	return result_vec;
}

//Vector Matrix multiplication
Row vec_mat_mul(Row& row, Matrixx& matrix)
{
	float m = row.size();//row size for vector
	float q = matrix.size();//number of rows
	float p = matrix[0].size();//number of cols
	Row result(p);
	if(m != q)
	{
		cout << "Invalid size for vec_mat_mul\n";						//Get all dimensions so multiplication is correct
		exit(0);
	}
	else
	{
		for (int i = 0; i < p; i++)
		{
			float sum = 0;
			for (int j = 0; j < m; j++)
			{
				sum = sum + row[j] * matrix[j][i];
			}
			result[i] = sum; 											//Returns n*p matrix for matices of size n*m and q*p
		}
	}
	return result;
}

//Softmax function
Row softmax(Row fin_lyr)
{
	Row y_hat;
	float sftmx;										//sftmax stores the float values obtained for each value in final layer
	float sum_exp_y = 0;								//Sum of all value's exponenets in finaly layer
	for (int i =0; i < fin_lyr.size(); i++)
	{
		sum_exp_y = sum_exp_y + exp(fin_lyr[i]); 		//Sum of all value's exponenets in finaly layer in formula
	}
	for(int j = 0; j < fin_lyr.size(); j++)
	{
		sftmx = (exp(fin_lyr[j])/sum_exp_y); 			//Softmax given by e^y(i)/sum(e^y(all))
		y_hat.push_back(sftmx);
	}
	return y_hat;
}

//Update bias
Row update_bias(Row bias_old, Row err_bias, float learn_rate)//Use to update bias and calculate error (Gradient descent)
{
	Row bias_upd(bias_old.size());
	for(int i = 0; i < bias_old.size(); i++)
	{
		bias_upd[i] = bias_old[i] - (learn_rate * err_bias[i]); // (b - eta*err_b)
	}
	return bias_upd;
}

float arg_max(Row y_hatt)
{
	Row y_hat_temp;
	y_hat_temp = y_hatt;
	int size = y_hatt.size();
	float max_val;
	float correct_val;
	sort(y_hat_temp.begin(), y_hat_temp.end());	//Sorts a copy of the input vector in asceding order
	max_val = y_hat_temp[size-1];				//Stores maximum value from sorted vector into max_val
	for(int i = 0; i < size; i++)
	{
		if (y_hatt[i] == max_val)
			correct_val = i;					//returns index of maximum value in the original vector
	}
	return correct_val;
}

int main()
{

	//Variables
	Row labels_train(60000,0),labels_test(10000,0),x(784,0),x_prev(784,0),z1(300,0),z1_prev(300,0),z1_derivative(10,0),z2(100,00),z2_prev(100,0),z2_derivative(100,0),z3(10,0),z3_prev(10,0),h1(300,0),h2(100,0),b1(300,0),b2(100,0),b3(10,0),y(10,0),y_hat(10,0),delta1(10,0),delta2(100,0),delta3(300,0),err_h1(300,0),err_h2(100,0),err_b1(300,0),err_b2(100,0),err_b3(10,0);
	Matrixx pixel_val_train(60000,Row(784,0)),pixel_val_test(10000,Row(784,0)),ohe_matrix_train(60000,Row(10,0)),w1(784,Row(300,0)),w2(300,Row(100,0)),w3(100,Row(10,0)),w2_transpose(100,Row(300,0)),w3_transpose(100,Row(10,0)),err_w1(784,Row(300,0)),err_w2(300,Row(100,0)),err_w3(100,Row(10,0));
	float value, accuracy, correct_value = 0;

	//Main part
	labels_train = read_label("train.csv");				//Read label values for train data
	cout << "Size of labels_train is " << labels_train.size() << endl;
	pixel_val_train = read_pixel("train.csv");			//Read pixel values for train data
	cout << "Size of pixel_values_train is " << pixel_val_train.size() << "**" << pixel_val_train[0].size() << endl;
	labels_test = read_label("test.csv");				//Read label values for test data
	cout << "Size of labels_test is " << labels_test.size() << endl;
	pixel_val_test = read_pixel("test.csv");			//Read pixel values for test data
	cout << "Size of pixel_values_test is " << pixel_val_test.size() << "**" << pixel_val_test[0].size() << endl;
	ohe_matrix_train = hotencode(labels_train);			//One hot encoded matrix for train data
	cout << "Size of ohe_matrix is " << ohe_matrix_train.size() << "**" << ohe_matrix_train[0].size() << endl;
	w1 = initialize_matrix(784,300);					//Initialize first weight matrix(W1) ,Matrix 784*300
	cout << "Size of w1 is " << w1.size() << "**" << w1[0].size() << endl;
	w2 = initialize_matrix(300,100);					//Initialize second weight matrix(W2), Matrix 300*100
	cout << "Size of w2 is " << w2.size() << "**" << w2[0].size() << endl;
	w3 = initialize_matrix(100,10);						//Initialize third weight matrix(W3), Matrix 100*10
	cout << "Size of w3 is " << w3.size() << "**" << w3[0].size() << endl;
	b1 = initialize_bias(300);							//Initialize first bias vector (b1), Vector 1*300
	cout << "Size of b1 is " << b1.size() << endl;
	b2 = initialize_bias(100);							//Initialize second bias vector (b2), Vector 1*100
	cout << "Size of b2 is " << b2.size() << endl;
	b3 = initialize_bias(10);							//Initialize third bias vector (b3), Vector 1*10
	cout << "Size of b3 is " << b3.size() << endl;

	//Training using input data
	//for(int i = 0; i < pixel_val_train.size(); i++)
	for(int i = 0; i < 1; i++)
	{
		x_prev = pixel_val_train[i];
		x = normalize_vector(x_prev);
		//Forward Propagation:
		//Hidden layer 1
		z1_prev = vec_mat_mul(x, w1); 					//Perform z = x*W1, Returns a vector 1*300
		z1 = sum_vector(z1_prev,b1);					//Perform z = x*W1 + b1, Returns a vector 1*300
		h1 = sigmoid(z1);								//Obatain first hidden layers values by applying sigmoid funcition to matrix, Returns a vector 1*300
		//Hidden layer 2
		z2_prev = vec_mat_mul(h1, w2);					//Perform z2 = h1*W2, Returns a vector 1*100
		z2 = sum_vector(z2_prev,b2);					//Perform z2 = h1*W2 + b2, Returns vector 1*100
		h2 = sigmoid(z2);								//Hidden layer 2 from sigmoid activation, Returns vector 1*100
		//Output layer
		z3_prev = vec_mat_mul(h2, w3);					//Perform z2 = h2*W3, Returns a vector 1*10
		z3 = sum_vector(z3_prev,b3);					//Perform z2 = h1*W2 + b2, Returns a vector 1*10
		y_hat = softmax(z3);							//Peform softmax operation on y_hat, Returns a vector 1*10
		y = ohe_matrix_train[i];						//Pick respective y from one hot encoded matrix

		//Backward Propagation
		//Output layer
		delta1 = sub_vector(y_hat,y);					//delta1 = (y_hat - y), Returns a vector 1*10

		//Hidden layer 2
		cout << "Size of w3 is " << w3.size() << "**" << w3[0].size() << endl;
		w3_transpose = transpose_matrix(w3);			//tranpose matrix w3 to bring it to correct dimension, Returns a matrix 10*100
		cout << "Size of w3 transpose is " << w3_transpose.size() << "**" << w3_transpose[0].size() << endl;
		err_h2 = vec_mat_mul(delta1, w3_transpose);		//error wrt to hidden layer 2, Returns a vector 1*100
		err_w3 = mat_vec_mul(h2, delta1);				//error wrt to weight matrix 3,Returns a vector 100*10
		err_b3 = delta1;								//error wrt to bias vector 3,Returns a vector 1*10

		//Hidden layer 1
		z2_derivative = sigmoid_derivative(z2);			//sigmoid derivative of z2, Returns a vector 1*100
		delta2 = hadamard(err_h2, z2_derivative);		//delta2 = hadmard product err_h2 * z2_derivative, Returns a vector 1*100
		cout << "Size of w2 is " << w2.size() << "**" << w2[0].size() << endl;
		w2_transpose = transpose_matrix(w2);			//tranpose matrix w2 to bring it to correct dimension, Returns a vector 1*10
		err_h1 = vec_mat_mul(delta2, w2_transpose);		//error wrt to h1, Returns a vector 1*300
		err_w2 = mat_vec_mul(h1, delta2);				//error wrt to w2, Returns a vector 300*100
		err_b2 = delta2;								//error wrt to b2, Returns a vector 1*100

		//input layer
		z1_derivative = sigmoid_derivative(z1);			//sigmoid derivative of z1, Returns a vector 1*10
		delta3 = hadamard(err_h1, z1_derivative);		//delta3 = hadmard product err_h1 * z1_derivative, Returns a vector 1*300
		err_w1 = mat_vec_mul(x, delta3);				//error wrt to w1, Returns a vector 784*300
		err_b1 = delta3;								//error wrt to b1, Returns a vector 1*300

		//Gradient descent
		//For w3 and b3
		w3 = update_weight(w3, err_w3, 0.1);			//w3_new = w3_old - eta*err_w3, 784*300
		b3 = update_bias(b3, err_b3, 0.01);				//b3_new = b3_old - eta*err_b3, 1*300
		//For w3 and b3
		w2 = update_weight(w2, err_w2, 0.1);			//w2_new = w2_old - eta*err_w2, 300*100
		b2 = update_bias(b2, err_b2, 0.01);				//b2_new = b2_old - eta*err_b2, 1*100
		//For w3 and b3
		w1 = update_weight(w1, err_w1, 0.1);			//w1_new = w1_old - eta*err_w1, 100*10
		b1 = update_bias(b1, err_b1, 0.01);				//b1_new = b1_old - eta*err_b1	1*10
		cout << "i is " << i << endl;
	}
	cout << "Exit first for loop " << endl;
	//Testing for test data
	for(int j = 0; j < pixel_val_test.size(); j++)
	{
		x_prev = pixel_val_test[j];
		x = normalize_vector(x_prev);
		//forward propagation:
		//Hidden layer 1
		z1_prev = vec_mat_mul(x, w1); 					//Perform z = x*W1, Returns a vector 1*300
		z1 = sum_vector(z1_prev,b1);					//Perform z = x*W1 + b1, Returns a vector 1*300
		h1 = sigmoid(z1);								//Obatain first hidden layers values by applying sigmoid funcition to matrix, Returns a vector 1*300
		//Hidden layer 2
		z2_prev = vec_mat_mul(h1, w2);					//Perform z2 = h1*W2, Returns a vector 1*100
		z2 = sum_vector(z2_prev,b2);					//Perform z2 = h1*W2 + b2, Returns vector 1*100
		h2 = sigmoid(z2);								//Hidden layer 2 from sigmoid activation, Returns vector 1*100
		//Output layer
		z3_prev = vec_mat_mul(h2, w3);					//Perform z2 = h2*W3, Returns a vector 1*10
		z3 = sum_vector(z3_prev,b3);							//Perform z2 = h1*W2 + b2, Returns a vector 1*10
		y_hat = softmax(z3);							//Peform softmax operation on y_hat, Returns a vector 1*10
		value = arg_max(y_hat);							//Check the maximum value's index in y_hat
		if (labels_test[j] == value)
			correct_value++;
		cout << "j is " << j << endl;
	}
	cout << "Exit second for loop " << endl;
	accuracy = (correct_value/labels_test.size())*100;
	cout << "The accuracy is " << accuracy << endl;
	return 0;
}

