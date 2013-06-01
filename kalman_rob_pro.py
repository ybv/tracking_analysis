import numpy as np
import sys
import decimal
import matplotlib.pyplot as plt
import pylab
import random
import time

def main():
	all_rob_pos_arr = {}
	all_est_arr={}
	rob_pos = np.zeros(shape=(2,1))
	est_arr = np.zeros(shape=(4,1))
	max_x =5.0
	max_y =5.0
	T=0
	filew = open('kalmanlog.txt', 'r+b')
	#P = np.matrix([[3,0,3/2,0],[0,3,0,3/2],[3/2,0,3,0],[0,3/2,0,3]])
	P = 0.9
	rob_x =[]
	rob_y =[]
	est_x =[]
	est_y=[]
	
	A =np.matrix([[1,0,T,0],[0,1,0,T],[0,0,1,0],[0,0,0,1]])
	C =np.matrix([[1,0,0,0],[0,1,0,0]])
	Ez =np.matrix([[0.2,0],[0,0.2]])
	init_rob = np.matrix([[random.uniform(0.0, 5.0)],[random.uniform(0.0, 5.0)]])
	all_rob_pos_arr[0]=init_rob
	init_est = np.matrix([[1],[2],[0],[0]])
	all_est_arr[0]=init_est
	ts1 = time.time()
	for T in range(0, 15):
		l= "step "+str(T)+"\n"
		filew.write(l)
		x = random.randint(0,3)
		x_disp_mat= np.matrix([[0.2],[0]])
		y_disp_mat= np.matrix([[0],[0.2]])
		x_disp_mat_est=np.matrix([[0.2],[0],[0],[0]])
		y_disp_mat_est=np.matrix([[0],[0.2],[0],[0]])
		if(x==0):#right
			new_rob = init_rob + x_disp_mat +np.matrix([[np.random.normal(0, 0.2)],[0]])
			if(new_rob[0]<max_x):
				all_rob_pos_arr[T]= new_rob
			else:
				all_rob_pos_arr[T]= init_rob
			new_est= init_est+ x_disp_mat_est
			if(new_est[0]<max_x):
				all_est_arr[T]=new_est
			else:
				all_est_arr[T]=init_est
			init_rob=new_rob
			init_est=new_est
			#print x

		if(x==1):#left
			new_rob = init_rob - x_disp_mat +np.matrix([[np.random.normal(0, 0.2)],[0]])
			if(new_rob[0]>0):
				all_rob_pos_arr[T]= new_rob
			else:
				all_rob_pos_arr[T]= init_rob
			new_est= init_est- x_disp_mat_est

			if(new_est[0]>0):
				all_est_arr[T]=new_est
			else:
				all_est_arr[T]=init_est
			all_est_arr[T]=new_est
			init_rob=new_rob
			init_est=new_est
			#print x
		if(x==2):#up
			new_rob = init_rob + y_disp_mat +np.matrix([[0],[np.random.normal(0, 0.2)]])
			if(new_rob[1]<max_y):
				all_rob_pos_arr[T]= new_rob
			else:
				all_rob_pos_arr[T]= init_rob
			new_est= init_est+ y_disp_mat_est
			if(new_est[1]<max_y):
				all_est_arr[T]=new_est
			else:
				all_est_arr[T]=init_est
			init_rob=new_rob
			init_est=new_est
			#print x
		if(x==3):#down
			new_rob = init_rob - y_disp_mat +np.matrix([[0],[np.random.normal(0, 0.2)]])
			if(new_rob[1]>0):
				all_rob_pos_arr[T]= new_rob
			else:
				all_rob_pos_arr[T]= init_rob
			new_est= init_est+ y_disp_mat_est
			if(new_est[1]>0):
				all_est_arr[T]=new_est
			else:
				all_est_arr[T]=init_est
			init_rob=new_rob
			init_est=new_est
		At = np.transpose(A)
		P = np.dot(P,np.dot(A,At))
   		#K = P*C'*inv(C*P*C'+Ez);
   		Ct= np.transpose(C)
   		inver= np.linalg.inv(np.dot(C,np.dot(P,Ct))+Ez)
   		kalg = np.dot(P,np.dot(Ct,inver))
   		#Q_estimate = Q_estimate + K * (Q_loc_meas - C * Q_estimate)
   		est_temp = all_rob_pos_arr[T]-np.dot(C,all_est_arr[T])
   		all_est_arr[T] = all_est_arr[T]+ np.dot(kalg,est_temp)
   		init_est=all_est_arr[T]

		sss =  "actual particle  at  (x, y) = ( "+str(new_rob[0])+" "+str(new_rob[1]) +" ), estimated value at (x, y) = ( "+str(new_est[0])+" "+str(new_est[1]) + " ) \n"	
		filew.write(sss)
   		
	ts2 = time.time()  	
   		
		
	

   	for i in range(0,15):
   		rob_x.append(float(all_rob_pos_arr[i][0]))
		rob_y.append(float(all_rob_pos_arr[i][1])) 
		est_x.append(float(all_est_arr[i][0]))
		est_y.append(float(all_est_arr[i][1]))



	timetaken = ts2-ts1
	print timetaken
	
	#figure(T)
	#pylab.xlim(0,5)
	#pylab.ylim(0,5)
	#plt.plot(all_rob_pos_arr[13][0],all_rob_pos_arr[13][0],'*' )
	#plt.plot(all_est_arr[13][0],all_est_arr[13][0],'gx')
	#plt.show()  

	print rob_x
	print rob_y
	print est_x
	print est_y

	

	#print all_rob_pos_arr
	#print all_est_arr	
	
	

	





if __name__ == '__main__':
    main()
