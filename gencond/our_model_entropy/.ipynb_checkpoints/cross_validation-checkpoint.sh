# /bin/bash

lambda="0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001 0.002 0.003 0.004 0.005 0.006"

for i in $lambda
do
	python train_qm9.py  --entropy_lambda $i
done
