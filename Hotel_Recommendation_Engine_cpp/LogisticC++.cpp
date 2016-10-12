#include <iostream>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <exception>
#include <sstream>
#include <cmath>
#include <Rcpp.h>
using namespace Rcpp;

using namespace std;

//Global declarations
const int Max_iter = 400;
const float Alpha  = 0.3;
const int Num_labels = 29;

int M;
int N;
int Pred_cols = 2;

//Function Prototypes
void readCSVFile(string argv, float matrix[][100]);

void oneVsAll( float *combined_X, float *Y, float *all_theta);

void predictOneVsAll(float *combined_X, float *all_theta, float *pred_z, float *pred);

float trainingAccuracy(float *pred, float *Y);

void gradientDescent(float *combined_X, float *initial_theta, float *new_Y);

/**********************************************************************
 ** Function Name : readCSVFile()                                    **
 ** Description   : To read the contents of the input csv file and   **
 **                 store it in a matrix                             **
 *********************************************************************/
void readCSVFile(string argv, float matrix[][100]) 
{
    
    string line;
    char *token    = NULL;
    char *temp_var = NULL;
    ifstream fp;
    int r = 0;
    int c = 0;
   

    //reading from csv into matrix
    fp.open(argv, ios::in);
    
    if (fp.is_open())
    {
        while (getline(fp,line))
        {
            temp_var = (char *)line.c_str();
            token = strtok(temp_var, ",");
            
            c = 0;
            while(token!=NULL)
            {
                matrix[r][c] = atoi(token); //converting char* to integer            
                c = c + 1;
                token= strtok(NULL,",");
            }
            r = r + 1;
        }
    }
    M = r;
    N = c;
         
}

/**********************************************************************
 ** Function Name : gradientDescent()                                **
 ** Description   : Gradient Descent finds theta for every Label     **
 **                                                                  **
 *********************************************************************/
void gradientDescent(float *combined_X, float *initial_theta, float *new_Y)
{
    float temp[N][1];   
    float z[M][1];
    float predictions[M][1];
    float srt_error[M][1];
    float inter[1][1];
    float transpose[1][M];
     int sub = 0;
    float var;
    float e = 2.718282;
    memset(temp, 0.0, N * sizeof(float));
    for(int iter = 0; iter < Max_iter; iter++)
    {
        memset(z, 0.0, M * sizeof(float));
        //matrix multiplication part z = x%*% theta
        for(int outer = 0; outer < M; outer++)
        {
            for(int center = 0; center < 1 ; center++)
            {
                for(int inner = 0; inner < N; inner++)
                {
                    z[outer][center] += *((combined_X + (N * outer))+ inner) * 
                                   (*((initial_theta + (1 * inner))+ center));  
                }
            }
        }
       
        //sigmoid matrix and srt_error calculation
        for(int sig = 0; sig < M; sig++)
        {
            var = (pow(e,-(z[sig][0]))) + 1;
            var = float(1.0)/var;
            predictions[sig][0] = var;
            srt_error[sig][0]   = (predictions[sig][0] - (*((new_Y + (1 * sig))+ 0)));  
        }
           
        //temp calculation
        sub = 0;
        for(int tem = 0; tem < N; tem++)
        {
            inter[0][0] = 0.0;
            
            //transpose calculation
            for(int trnso = 0; trnso < M; trnso++)
            {
                transpose[0][trnso] = *((combined_X + (N * trnso))+ tem);
            }
           
            //calculation of final temp
            for(int i = 0; i < 1; i++)
            {
                for(int k = 0; k < 1; k++)
                {
                    for(int l = 0; l < M; l++)
                    {
                        inter[i][k] +=  transpose[i][l] * srt_error[l][k];
                    }
                    inter[i][k] = inter[i][k] * (Alpha * (float(1.0)/M));
                }
            }
            temp[sub][0] = (*((initial_theta + (1 * sub)) + 0)) - inter[0][0]; 
            sub++;
                
                
        }//end of temp calculation
          
        for(int eq = 0; eq < N; eq++)
        {
            (*((initial_theta + (1 * eq)) + 0)) = temp[eq][0]; 
        }  
           
    }//end of max iter calculation
}
/**********************************************************************
 ** Function Name : oneVsAll()                                       **
 ** Description   : To compute theta values for every label          **
 **                                                                  **
 *********************************************************************/

void oneVsAll( float *combined_X, float *Y, float *all_theta)
{
    float initial_theta[N][1];
    float C_mat[M][1];
	float new_Y[M][1];
    float cur_Y[M][1];
    
    //comparing with every category
    for(int comp = 1; comp <= Num_labels; comp++)
    {
        
        memset(initial_theta, 0.0, N * sizeof(float));
        
        //copying Y into cur_Y
        for(int i = 0; i < M; i++)
        {
            
            cur_Y[i][0] = *((Y + (1 * i))+ 0);
            
        } // end of copying Y into cur_Y
        
        //initializing C_mat
        for(int i = 0; i < M; i++)
        {
            
            C_mat[i][0] = comp;
            
        }
        //end of initializing C_mat
        
        
        //checking matching values
        for(int row2 = 0; row2 < M ; row2++)	
    	{
			if(cur_Y[row2][0] == C_mat[row2][0])
    		{
                new_Y[row2][0] = 1;
            }
            else
            {
                new_Y[row2][0] = 0;
            }
        }// end of checking matching values
        
        //gradient descent starts here
        gradientDescent(combined_X, *initial_theta, *new_Y );
        
        for(int all = 0; all < N; all++)
        {
            *((all_theta + (N * (comp - 1)))+ all) = initial_theta[all][0];
        }
    
     }//end of comp loop*/
    
}

/**********************************************************************
 ** Function Name : predictOneVsAll()                                **
 ** Description   : To compute the prediction matrix for all the     **
 **                 training set based on all_theta values           **
 **                                                                  **
 *********************************************************************/

void predictOneVsAll(float *combined_X, float *all_theta, float *pred_z, float *pred)
{
     float max_value;
     int pos;
    
     memset(pred_z, 0.0, M * Num_labels * sizeof(float));
    
     // X multiplication with all_theta
     for(int pred_zi = 0 ; pred_zi < M; pred_zi++)
     {
         for(int pred_zj = 0; pred_zj < Num_labels; pred_zj++)
         {
             for(int pred_zk = 0; pred_zk < N; pred_zk++)
             {
                 *((pred_z + (Num_labels * pred_zi)) + pred_zj) += 
                 (*((combined_X + (N * pred_zi)) + pred_zk)) * (*((all_theta + (N * pred_zj)) + pred_zk));
             }
         }
     } //X multiplication with all_theta ends here
    
    //find max values from each row and finding the position.
    for(int i = 0; i < M; i++)
    {
        max_value = *((pred_z + (Num_labels * i)) + 0); 
        pos = 0;
        for(int j=0;j<Num_labels;j++)
        {
            if(*((pred_z + (Num_labels * i)) + j) > max_value)
            {
                max_value = *((pred_z + (Num_labels * i)) + j);
                pos = j;
            }            
        }
        *((pred + (Pred_cols * i)) + 0) = max_value;
        *((pred + (Pred_cols * i)) + 1) = pos + 1;
    } 
}

/**********************************************************************
 ** Function Name : trainingAccuracy()                               **
 ** Description   : To compute the training accuracy                 **
 **                                                                  **
 *********************************************************************/

float trainingAccuracy(float *pred, float *Y)
{
    float total = 0.0;
    for(int i = 0;i < M;i++)
    {
        *((pred + (Pred_cols * i)) + 1) = 
        ((*((pred + (Pred_cols * i)) + 1)) == (*((Y + (1 * i)) + 0))) ? 1.0 : 0.0;  
        total += *((pred + (Pred_cols * i)) + 1);
    }
    total  = (total/M) * 100;
    return total;
}

/**********************************************************************
 ** Function Name : main()                                           **
 ** Description   : Beginning of Execution                           **
 **                                                                  **
 *********************************************************************/
// [[Rcpp::export]]
float startMain(string fileName)
{
    float matrix[100][100];
    
    //reading input matrix from CSV file
    readCSVFile(fileName, matrix);

    float X[M][N - 1];
    float Y[M][1];
    float combined_X[M][N];
    float all_theta[Num_labels][N];
    float pred_z[M][Num_labels];
    float pred[M][Pred_cols];
    float training_accuracy=0.0;
      
    //splitting into X and Y matrices

    for(int row = 0 ; row < M ; row++)
    {
    	for(int col = 0; col < N; col++)
    	{
            //populate Y if last column is reached
    		if(col == N - 1)
    		{
    			Y[row][0] = matrix[row][col];
    		}
            //populate X
    		else
    		{

    			X[row][col] = matrix[row][col];
    		}
    	}
    } // end of splitting into X and Y matrices
    
    
    //combining onesMatrix and X
    for(int row1 = 0 ; row1 < M; row1++)
    {
        for(int col1 = 0; col1 < N; col1++)
        {
            if(col1 == 0)
            {
                combined_X[row1][0] = 1;
            }
            else
            {
                combined_X[row1][col1] = X[row1][col1-1];
            }
            
        } 
    } // end of combining onesMatrix and X
    

	//one vs all function call
    oneVsAll(*combined_X, *Y, *all_theta);
    
    //predictOneVsAll starts here
    predictOneVsAll(*combined_X, *all_theta, *pred_z, *pred);
     
    
    //Calculating training accuracy
    training_accuracy = trainingAccuracy(*pred, *Y);
    
    cout << "\ntraining accuracy is " << training_accuracy << "%\n";
    
    return training_accuracy;
}

