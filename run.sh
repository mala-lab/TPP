for dataset in  CoraFull-CL Arxiv-CL
do
python train.py --dataset $dataset --method tpp --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False
done