if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

echo "EE: PARAM NODE"
$1 ./examples/param.py \
	--dataset EE \
	--data_dir ./data/EulerEqs.npz \
	--model NODE \
	--verbose

echo "EE: PARAM HBNODE"
$1 ./examples/param.py \
	--dataset EE \
	--data_dir ./data/EulerEqs.npz \
	--model HBNODE \
	--verbose

echo "EE: PARAM GHBNODE"
$1 ./examples/param.py \
	--dataset EE \
	--data_dir ./data/EulerEqs.npz \
	--model GHBNODE \
	--verbose

#echo "FIBER: PARAM NODE"
#$1 ./examples/param.py \
	#--dataset FIB \
	#--data_dir ./data/FIB.npz \
	#--model HBNODE \
	#--epochs 10 \
	#--verbose

#echo "FIBER: PARAM HBNODE"
#$1 ./examples/param.py \
	#--dataset FIB \
	#--data_dir ./data/FIB.npz \
	#--model HBNODE \
	#--verbose

cp ./out/param_examples/*.gif ./doc/img/
