#!/usr/bin/bash
if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

echo "VKS: SEQ NODE"
$1 ./examples/seq.py \
	--dataset VKS \
	--data_dir ./data/VKS.pkl \
	--tr_ind 75 \
	--val_ind 100 \
	--eval_ind 200 \
	--model NODE \
	--verbose

echo "VKS: SEQ HBNODE"
$1 ./examples/seq.py \
	--dataset VKS \
	--data_dir ./data/VKS.pkl \
	--tr_ind 75 \
	--val_ind 100 \
	--eval_ind 200 \
	--model HBNODE \
	--verbose

echo "VKS: SEQ GHBNODE"
$1 ./examples/seq.py \
	--dataset VKS \
	--data_dir ./data/VKS.pkl \
	--tr_ind 75 \
	--val_ind 100 \
	--eval_ind 200 \
	--model GHBNODE \
	--verbose

echo "KPP: SEQ NODE"
$1 ./examples/seq.py \
	--dataset KPP \
	--data_dir ./data/KPP.npz \
	--tr_ind 75 \
	--val_ind 100 \
	--epochs 50 \
	--plt_itvl 1 \
	--eval_ind 200 \
	--model NODE \
	--verbose

echo "KPP: SEQ HBNODE"
$1 ./examples/seq.py \
	--dataset KPP \
	--data_dir ./data/KPP.npz \
	--tr_ind 75 \
	--val_ind 100 \
	--epochs 50 \
	--plt_itvl 1 \
	--eval_ind 200 \
	--model HBNODE \
	--verbose

echo "KPP: SEQ GHBNODE"
$1 ./examples/seq.py \
	--dataset KPP \
	--data_dir ./data/KPP.npz \
	--tr_ind 75 \
	--val_ind 100 \
	--epochs 50 \
	--plt_itvl 1 \
	--eval_ind 200 \
	--model GHBNODE \
	--verbose

exit 1

cp ./out/seq_examples/*.gif ./doc/img/
