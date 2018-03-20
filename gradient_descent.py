import sys  
#Training data set  
#each element in x represents (x1)  
x = [1,2,3,4,5,6]  
#y[i] is the output of y = theta0+ theta1 * x[1]  
y = [8,13,18,23,28,33]  
#设置允许误差值  
epsilon = 0.1  
#学习率  
alpha = 0.12  
diff = [0,0]  
max_itor = 20  
error1 = 0  
error0 =0  
cnt = 0  
m = len(x)  
#init the parameters to zero  
theta0 = 0  
theta1 = 0  
while 1:  
    cnt=cnt+1  
    diff = [0,0]  
    for i in range(m):  
        diff[0]+=theta0+ theta1 * x[i]-y[i]  
        diff[1]+=(theta0+theta1*x[i]-y[i])*x[i]  
    theta0=theta0-alpha/m*diff[0]  
    theta1=theta1-alpha/m*diff[1]  
    error1=0  
    for i in range(m):  
        error1+=(theta0+theta1*x[i]-y[i])**2  
    if abs(error1-error0) == 0: #< epsilon:  
        break  
    print('theta0 :%f,theta1 :%f,error:%f' % (theta0,theta1,error1))  
    if cnt>500:  
        print ('cnt>500')  
        break  
print('theta0 :%f,theta1 :%f,error:%f' % (theta0,theta1,error1))