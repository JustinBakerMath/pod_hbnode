if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

echo "VKS DMD Example"
$1 ./examples/dmd.py \
	--dataset VKS \
	--data_dir ./data/VKS.pkl \
	--out_dir ./out/nonT_pred/ \
	--lifts sin cos quad cube \
	--modes 24 \
	--tstart 100 \
	--tstop 400 \
	--tpred 399
	--verbose

echo "KPP DMD Example"
$1 ./examples/dmd.py \
  --dataset KPP \
  --data_dir ./data/KPP.npz \
	--modes 24 \
	--tstart 0 \
	--tstop 1000 \
	--tpred 999 \
	--lifts sin cos quad cube \
  --verbose

cp ./out/dmd_examples/*.gif ./doc/img/
