# pycharmproject

To code is for my master project " Matrix Completion: Infer Missing Entries"
This code is for my master project " Matrix Completion: Infer Missing Entries". To run the code directly, you should first download required packages, change the directory to your local path and run "Main.py".

* I have attach my master dissertation, where the details of experiemnts are explained.
To run the code, you have to download required packages and change the directory to your local path.
All code and dataset used are stored in the "WTYprogram" folder. In detail, it contains one folder named "dataset", four python files named "Main.py", "dataloader.py", "imageloader.py", "ASD.py", "SVT.py", "SVP.py", "Schatten.py" and a report "Wang Tingyue_10439320_Master Project Report.pdf". Their usages are described below. 

* "dataset" folder contains "lena.png","picture.png" , which are used as the input matrix for image recovery task. There is also a folder named "ml-1m" containing the rating matrix used for rating prediction task.

* "Main.py" contains the main function for seven experiments described in the report. Setting the parameter "Experiment" to 1-7 can run experiment 1-7.

* "dataloader.py" is the data loading part code for random matrix completion task (experiemnt 1-4) and rating prediction task (experiment 7).

* "imageloader.py" is the data loading part code for image recovery task (experiment 5-6).

* "ASD.py", "SVT.py", "SVP.py" and "Schatten.py" are the core algorithm code for Alternating Steepest Descent Method, Singular Value Thresholding Method, Singular Value Projection Method and Schattern-p Norm Minimization Method respectively.

* "Wang Tingyue_10439320_Master Project Report.pdf" is my master project report, which describes the details about the experiment design and parameter setting.
