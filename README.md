# MDnet
MDnet visual tracking algorithm implementation version 3. Sample trained_nets which are trained on vot2016  are also uploaded as zip files


How to run:

		CUDA_VISIBLE_DEVICES=2 python srcv3.py online0
		
		the program asks you to input a video name and you need to download and prepare vot2016 datasets

Files:

		libv3.py: contains all classes and most functions
  
		options.py: contains all parameters we need to modify
  
		srcv3.py: offline_training, online_tracking


Dependencies:

		(1)python3.5
  
		(2)opencv,numpy
  
		(3)pytorch
  
		(4)scikit-learn
 

Hardware:

		Nvidia GTX TITAN X(recommended), but it can also run without gpu.
  

 
