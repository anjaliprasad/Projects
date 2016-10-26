import numpy as np
import matplotlib.pyplot as plt

def random_sample(size):
    x = np.random.uniform(-1,1,size)
    mu, sigma = 0, 0.1 # mean and standard deviation
    epsilon = np.random.normal(mu, sigma, size)
    y = 2*pow(x,2)+epsilon
    return x,y

def linear_reg(X,y):
    theta = (np.linalg.pinv((X.T * X))) * (X.T * y)
    prediction = np.mat(np.zeros(X.shape[0])).transpose(((1, 0)))
    for i in range(0, X.shape[0]):
        prediction[i] = X[i] * theta

    MSE_train = MSE(y, prediction)
    #print theta
    return MSE_train,theta

def ridge_reg(X,y, lam):
    theta = (np.linalg.pinv((X.T * X) + (lam * np.identity(X.shape[1])))) * (X.T * y)
    prediction = np.mat(np.zeros(X.shape[0])).transpose(((1, 0)))
    for i in range(0, X.shape[0]):
        prediction[i] = X[i] * theta

    MSE_train = MSE(y, prediction)
    #print theta
    return MSE_train,theta

def MSE(y,prediction):
    mse = (sum(np.square(y - prediction)))/len(y)
    return mse[0,0]

def plotHistogram1(MSE_List):
        MSE_List = np.array(MSE_List)
        plt.hist(MSE_List)
        plt.title("Histogram for MSE")
        plt.xlabel("MSE")
        plt.ylabel("Value")
        plt.show()

def bias(g_avg, y, model):
    bias_g = np.mean(pow((g_avg - np.array(y)),2))
    print model ,bias_g


def plotHistogram(MSE_List_Set):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
    ax1.set_ylabel("g1(x)")
    ax1.hist(MSE_List_Set[0])
    ax2.set_ylabel("g2(x)")
    ax2.hist(MSE_List_Set[1])
    ax3.set_ylabel("g3(x)")
    ax3.hist(MSE_List_Set[2])
    ax4.set_ylabel("g4(x)")
    ax4.hist(MSE_List_Set[3])
    ax5.set_ylabel("g5(x)")
    ax5.hist(MSE_List_Set[4])
    ax6.set_ylabel("g6(x)")
    ax6.hist(MSE_List_Set[5])
    plt.show()

def getThetaMatrix(theta_List_g4):
    rows = len(theta_List_g4[0][0])
    intermediate_g4 = map(list, zip(*theta_List_g4))
    g4_model = np.ones((rows, 1))
    for i in range(0, len(intermediate_g4[0])):
        if i == 0:
            g4_model = np.transpose(intermediate_g4[0][i])
            g4_model = g4_model.reshape(rows, 1)
        else:
            new_col = np.transpose(intermediate_g4[0][i]).reshape(rows, 1)
            g4_model = np.concatenate((g4_model, new_col), axis=1)
    return (g4_model)

def bias_intermediate_calculation(X2,g2_thetas,i,g2_avg):

    g2_avg_intermediate = X2 * g2_thetas
    point_mean_g2 = np.mean(g2_avg_intermediate[0])
    # print point_mean_g2
    if i == 0:
        g2_avg = point_mean_g2.reshape(1, 1)
    else:
        point_mean_g2 = point_mean_g2.reshape(1, 1)
        g2_avg = np.concatenate((g2_avg, point_mean_g2), axis=1)

    return g2_avg,point_mean_g2

def var_intermediate_calculation(X2,g2_thetas,point_mean_g2,g2_var,i):
    se_list = []
    for vr in range(0, 100):
        rows = len(g2_thetas)
        model = g2_thetas[:, vr].reshape(rows,1)
        g2 = (X2 * model)
        diff = pow((g2 - point_mean_g2), 2)
        se_list.append(diff)
    se_list = np.mean(se_list)
    if i == 0:
        g2_var = se_list.reshape(1, 1)
    else:
        se_list = se_list.reshape(1, 1)
        g2_var = np.concatenate((g2_var, se_list), axis=1)

    return g2_var

def main(sample):
    #Training/Ridge----------------------------------------------------------------------------

    MSE_List_g1=[]
    MSE_List_g2 = []
    MSE_List_g3 = []
    MSE_List_g4 = []
    MSE_List_g5 = []
    MSE_List_g6 = []
    theta_List_g1 = []
    theta_List_g2 = []
    theta_List_g3 = []
    theta_List_g4 = []
    theta_List_g5 = []
    theta_List_g6 = []
    for i in range(0,100):
        x,y=random_sample(sample)
        x2 = pow(x,2)
        x3 = pow(x,3)
        x4 = pow(x,4)
        X = (np.mat(x)).transpose((1, 0))
        Y = (np.mat(y)).transpose((1, 0))
        X = np.insert(X, 0, 1, axis=1)
        X = np.insert(X, 2, x2, axis=1)
        X = np.insert(X, 3, x3, axis=1)
        X = np.insert(X, 4, x4, axis=1)
        #Y_New =
        #For g1----------------------------------------------
        y1 = np.mat(np.ones(Y.shape[0])).transpose(((1, 0)))

        MSE_g1 = MSE(y,y1)
        MSE_List_g1.append(MSE_g1)
        #----------------------------------------------------

        #For g2----------------------------------------------
        x2 = X[:,0]
        MSE_g2, theta_g2 = linear_reg(x2, Y)
        theta_g2 = theta_g2.reshape(1, 1)
        theta_List_g2.append(np.array(theta_g2.transpose(1, 0)))
        MSE_List_g2.append(MSE_g2)
        #-----------------------------------------------------
        # For g3----------------------------------------------
        x3 = X[:, [0,1]]
        MSE_g3, theta_g3 = linear_reg(x3, Y)
        theta_g3 = theta_g3.reshape(2, 1)
        theta_List_g3.append(np.array(theta_g3.transpose(1, 0)))
        MSE_List_g3.append(MSE_g3)
        # ----------------------------------------------------
        # For g4----------------------------------------------
        x4 = X[:, [0, 1, 2]]
        MSE_g4,theta_g4 = linear_reg(x4, Y)
        theta_g4= theta_g4.reshape(3,1)
        theta_List_g4.append(np.array(theta_g4.transpose(1,0)))
        MSE_List_g4.append(MSE_g4)
        # ----------------------------------------------------
        # For g5----------------------------------------------
        x5 = X[:, [0, 1, 2,3]]
        MSE_g5,theta_g5 = linear_reg(x5, Y)
        theta_g5 = theta_g5.reshape(4, 1)
        theta_List_g5.append(np.array(theta_g5.transpose(1, 0)))
        MSE_List_g5.append(MSE_g5)
        # ----------------------------------------------------
        # For g6----------------------------------------------
        x6 = X[:, [0, 1, 2, 3,4]]
        MSE_g6,theta_g6 = linear_reg(x6, Y)
        theta_g6 = theta_g6.reshape(5, 1)
        theta_List_g6.append(np.array(theta_g6.transpose(1, 0)))
        MSE_List_g6.append(MSE_g6)
        # -----------------------------------------------------
    MSE_ConList = [MSE_List_g1,MSE_List_g2,MSE_List_g3,MSE_List_g4,MSE_List_g5,MSE_List_g6]
    g2_thetas = getThetaMatrix(theta_List_g2)
    g3_thetas= getThetaMatrix(theta_List_g3)
    g4_thetas =getThetaMatrix(theta_List_g4)
    g5_thetas = getThetaMatrix(theta_List_g5)
    g6_thetas = getThetaMatrix(theta_List_g6)

    plotHistogram(MSE_ConList)
    #---------------------------------------------------------------------------------------
    print "Please wait for a few seconds for results"
    lam =[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    #Testing set implementation
    g2_avg = np.zeros((1,1))
    g3_avg = np.zeros((1, 1))
    g4_avg = np.zeros((1, 1))
    g5_avg = np.zeros((1, 1))
    g6_avg = np.zeros((1, 1))

    g2_var = np.zeros((1, 1))
    g3_var = np.zeros((1, 1))
    g4_var = np.zeros((1, 1))
    g5_var = np.zeros((1, 1))
    g6_var = np.zeros((1, 1))

    sizeOfTest = 1000
    x,y=random_sample(sizeOfTest)
    x2 = pow(x, 2)
    x3 = pow(x, 3)
    x4 = pow(x, 4)
    X = (np.mat(x)).transpose((1, 0))
    X = np.insert(X, 0, 1, axis=1)
    X = np.insert(X, 2, x2, axis=1)
    X = np.insert(X, 3, x3, axis=1)
    X = np.insert(X, 4, x4, axis=1)
    Y = (np.mat(y)).transpose((1, 0))


    for i in range(0,sizeOfTest):


        #---g2
        X2 = X[:, 0]
        g2_avg,point_mean_g2 = bias_intermediate_calculation(X2[i],g2_thetas,i,g2_avg)
            #---var
        g2_var = var_intermediate_calculation(X2[i],g2_thetas,point_mean_g2,g2_var,i)

        #---g3
        X3 = X[:, [0, 1]]
        g3_avg,point_mean_g3 = bias_intermediate_calculation(X3[i], g3_thetas, i, g3_avg)
        # ---var
        g3_var = var_intermediate_calculation(X3[i], g3_thetas, point_mean_g3, g3_var, i)

        # ---g4
        X4 = X[:, [0, 1, 2]]
        g4_avg,point_mean_g4 = bias_intermediate_calculation(X4[i], g4_thetas, i, g4_avg)
        # ---var
        g4_var = var_intermediate_calculation(X4[i], g4_thetas, point_mean_g4, g4_var, i)
        # ---g5
        X5 = X[:, [0, 1, 2,3]]
        g5_avg,point_mean_g5 = bias_intermediate_calculation(X5[i], g5_thetas, i, g5_avg)
        # ---var
        g5_var = var_intermediate_calculation(X5[i], g5_thetas, point_mean_g5, g5_var, i)

        # ---g6
        X6 = X[:, [0, 1, 2,3,4]]
        g6_avg,point_mean_g6 = bias_intermediate_calculation(X6[i], g6_thetas, i, g6_avg)
        # ---var
        g6_var = var_intermediate_calculation(X6[i], g6_thetas, point_mean_g6, g6_var, i)

    print "Bias for samples = ",sample
    bias_g1 =  bias(np.transpose(np.ones((1,sizeOfTest))),Y,"g1")
    bias_g2 = bias(np.transpose(g2_avg),Y,"g2")
    bias_g3 = bias(np.transpose(g3_avg), Y,"g3")
    bias_g4 = bias(np.transpose(g4_avg), Y,"g4")
    bias_g5 = bias(np.transpose(g5_avg), Y,"g5")
    bias_g6 = bias(np.transpose(g6_avg), Y,"g6")

    #---------------------------------------------------------------------------
    #Variance-------------------------------------------------------------------
    print "Variance for sample = ",sample
    print "g1" , 0.0
    print "g2", np.mean(g2_var)
    print "g3", np.mean(g3_var)
    print "g4", np.mean(g4_var)
    print "g5", np.mean(g5_var)
    print "g6", np.mean(g6_var)

    # Training ridge----------------------------------------------------------------------------
    if(sample == 100):
        for l in range(0, len(lam)):
            print "\n"
            print "for lambda = ", lam[l]
            MSE_List_g4 = []
            theta_List_g4 = []
            for i in range(0, 100):
                x, y = random_sample(sample)
                x2 = pow(x, 2)
                x3 = pow(x, 3)
                x4 = pow(x, 4)
                X = (np.mat(x)).transpose((1, 0))
                Y = (np.mat(y)).transpose((1, 0))
                X = np.insert(X, 0, 1, axis=1)
                X = np.insert(X, 2, x2, axis=1)
                X = np.insert(X, 3, x3, axis=1)
                X = np.insert(X, 4, x4, axis=1)


                # For g4----------------------------------------------
                x4 = X[:, [0, 1, 2]]
                MSE_g4, theta_g4 = ridge_reg(x4, Y,lam[l])
                theta_g4 = theta_g4.reshape(3, 1)
                theta_List_g4.append(np.array(theta_g4.transpose(1, 0)))
                MSE_List_g4.append(MSE_g4)
            # ----------------------------------------------------

            g4_thetas = getThetaMatrix(theta_List_g4)


            g4_avg = np.zeros((1, 1))
            g4_var = np.zeros((1, 1))
            x, y = random_sample(sizeOfTest)
            x2 = pow(x, 2)
            x3 = pow(x, 3)
            x4 = pow(x, 4)
            X = (np.mat(x)).transpose((1, 0))
            X = np.insert(X, 0, 1, axis=1)
            X = np.insert(X, 2, x2, axis=1)
            X = np.insert(X, 3, x3, axis=1)
            X = np.insert(X, 4, x4, axis=1)
            Y = (np.mat(y)).transpose((1, 0))

            for i in range(0, sizeOfTest):

                # ---g4
                X4 = X[:, [0, 1, 2]]
                g4_avg, point_mean_g4 = bias_intermediate_calculation(X4[i], g4_thetas, i, g4_avg)
                # ---var
                g4_var = var_intermediate_calculation(X4[i], g4_thetas, point_mean_g4, g4_var, i)
            print "Bias for samples = ", sample

            bias_g4 = bias(np.transpose(g4_avg), Y, "g4")
            print "Variance for sample = ", sample
            print "g4", np.mean(g4_var)


    # ---------------------------------------------------------------------------------------

