# Introduction to parallel computing - Homework 2 - Exercise 2

## Submitting the job to the PBS scheduler
First of all I raccomand to delete all files of prevuois runs in the `/results` folder or create that folder if dosn't exist. 
Then for submitting the job on the cluster we must use the following command:
```bash
qsub doStrongScaling.pbs
```
This command will submit the job on the cluster and will menage all the things are needed for doing the entire scaling test. The results will be saved in the `/results` folder.


