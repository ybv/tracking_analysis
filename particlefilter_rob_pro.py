import numpy as np
import sys
import decimal
import matplotlib.pyplot as plt
import pylab
import random
import time 

def main():

	def GaussianKernel(v1, v2, sigma):
		d=math.fabs(v1-v2)
		return ((1/math.sqrt(2*pi*(sigma))))*exp((-1)*(math.pow(d,2)/(sigma*2)))
	
   	upd_xy = np.zeros(shape=(500,2))
   	filew = open('partlog.txt', 'r+b')
	
	xy = np.zeros(shape=(500,2))
	max_x =5.0
	max_y =5.0

	a = decimal.Decimal('1')
	b = decimal.Decimal('200')
	c = a/b
	weight= np.float32(c)*np.ones(500,dtype=float)

	upd_weight= np.zeros(500,dtype=float)

	part_dist=np.ones(shape=(500,1),dtype=float)

	act_rob_pos=np.ones(shape=(25,2),dtype=float)

	rob_dist=np.ones(shape=(25,1),dtype=float)

	for i in range(0,200):

		for j in range(0,2):

			xy[i][0]=random.uniform(0.0, 5.0)
			xy[i][1]=random.uniform(0.0, 5.0)

	

		act_rob_pos[0][0]=random.uniform(0.0, 5.0)
		act_rob_pos[0][1]=random.uniform(0.0, 5.0)

	#for i in range(0,500):
	#	figure(0)
	#	pylab.xlim(0,5)
	#	pylab.ylim(0,5)
	
	#	plt.plot(xy[i][0],xy[i][1],'bx')
    #	plt.plot(act_rob_pos[0][0],act_rob_pos[0][1],'ro')
	#plt.show()	
	ts1 = time.time()
	for k in range (0,20):
		l= "step "+str(k)+"\n"
		filew.write(l)
		x = random.randint(0,3)
		print x
		if(x==0):#right
			robtemp= act_rob_pos[k][0]
			act_rob_pos[k][0] = act_rob_pos[k-1][0]+0.2+(random.random()*0.2)-(0.2/2)
			if(act_rob_pos[k][0]>max_x):
				act_rob_pos[k][0]=robtemp

			d1 = act_rob_pos[k][0]-max_x
	   		d2 = 0
	   		robdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   		robdist1=robdx1+(random.random()*0.25)-(0.25/2)
			for i in range(0,500):
				
				temp =xy[i][0]
				xy[i][0]=xy[i][0]+0.2
				if(xy[i][0]>max_x):
					xy[i][0]=temp
				
				d1 = xy[i][0]-max_x
	   			d2 = xy[i][1]-act_rob_pos[k][1]
				partdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   			
	   			partdist1= partdx1+(random.random()*0.25)-(0.25/2)

				kernel=GaussianKernel(robdist1,partdist1,0.3)
				updweight=np.float32(kernel)
				weight[i]=updweight

		if(x==1):#left
			robtemp=act_rob_pos[k][0]
			act_rob_pos[k][0] = act_rob_pos[k-1][0]-0.2+(random.random()*0.2)-(0.2/2)
			if(act_rob_pos[k][0]<0):
				act_rob_pos[k][0]=robtemp

			d1 = act_rob_pos[k][0]-0
	   		d2 = 0
	   		robdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   		robdist2=robdx1+(random.random()*0.25)-(0.25/2)
			for i in range(0,500):
				
				temp =xy[i][0]
				xy[i][0]=xy[i][0]-0.2
				if(xy[i][0]<0):
					xy[i][0]=temp
			
				d1 = xy[i][0]-0
	   			d2 = xy[i][1]-act_rob_pos[k][1]
				partdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   			
	   			partdist2= partdx1+(random.random()*0.25)-(0.25/2)

				kernel=GaussianKernel(robdist2,partdist2,0.3)
				updweight=np.float32(kernel)
				weight[i]=updweight
		if(x==2):#up
			robtemp=act_rob_pos[k][1]
			act_rob_pos[k][1] = act_rob_pos[k-1][1]+0.2+(random.random()*0.2)-(0.2/2)
			if(act_rob_pos[k][1]>max_y):
				act_rob_pos[k][0]=robtemp
			d1 = 0
	   		d2 = act_rob_pos[k][1]-max_y
	   		robdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   		robdist3=robdx1+(random.random()*0.25)-(0.25/2)
			for i in range(0,500):
				
				temp =xy[i][1]
				xy[i][1]=xy[i][1]+0.2
				if(xy[i][1]>max_y):
					xy[i][1]=temp
			
				d1 = xy[i][0]-act_rob_pos[k][0]
	   			d2 = xy[i][1]-max_y
				partdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   				
	   			partdist3= partdx1+(random.random()*0.25)-(0.25/2)

				kernel=GaussianKernel(robdist3,partdist3,0.3)
				updweight=np.float32(kernel)
				weight[i]=updweight
		if(x==3):
			robtemp=act_rob_pos[k][1]
			act_rob_pos[k][1] = act_rob_pos[k-1][1]-0.2+(random.random()*0.2)-(0.2/2)
			if(act_rob_pos[k][1]<0):
				act_rob_pos[k][0]=robtemp
			d1 = 0
	   		d2 = act_rob_pos[k][1]-0
	   		robdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   		robdist4=robdx1+(random.random()*0.25)-(0.25/2)
			for i in range(0,500):
				
				temp =xy[i][1]
				xy[i][1]=xy[i][1]-0.2
				if(xy[i][1]<0):
					xy[i][0]=temp	
			
				d1 = xy[i][0]-act_rob_pos[k][0]
	   			d2 = xy[i][1]-0
				partdx1 = math.sqrt(math.pow(d1,2)+math.pow(d2,2))
	   				
	   			partdist4= partdx1+(random.random()*0.25)-(0.25/2)

				kernel=GaussianKernel(robdist4,partdist4,0.3)
				updweight=np.float32(kernel)
				weight[i]=updweight


		kkk=0
		sumx= weight.sum()
		weight=float(1/sumx)*(weight) 
		cum=np.zeros(501,dtype=float)
		cum[0]=0
		
		#print weight
		for i in range(0,500):
			cum[i+1]=cum[i]+weight[i]

		#print cum

	
		noofpart =np.zeros(500,dtype=int)

		for i in range(0,500):
			f=random.uniform(0.0, 1.0)
			for m in range(0,500): 
				if(f>cum[m] and f<= cum[m+1]):
					noofpart[m]+=1

		#print noofpart
		for i in range(0,500):
			for m in range(0,noofpart[i]):
				#print "yes"
				upd_xy[kkk][0] =xy[i][0]
				upd_xy[kkk][1] =xy[i][1]
				upd_weight[kkk] =weight[i]
				kkk+=1
		#print upd_xy


		for i in range(0,500):

			xy[i][0]=upd_xy[i][0]
			xy[i][1]=upd_xy[i][1]
		#print xy
		weight=upd_weight
		ts2 = time.time()
		for i in range(0,500):
			s =  " particle  "+str(i)+"  (x, y) = ( "+str(xy[i][0])+" "+str(xy[i][1]) +" ), weight ="+ str(weight[i]) +"\n"	
			filew.write(s)
		
		#for i in range(0,500):
		#	figure(k+1)
		#	pylab.xlim(0,5)
		#	pylab.ylim(0,5)
		#	plt.plot(xy[i][0],xy[i][1],'bx')
    	#	plt.plot(act_rob_pos[k][0],act_rob_pos[k][1],'ro')
    	#plt.show()
    	







if __name__ == '__main__':
    main()