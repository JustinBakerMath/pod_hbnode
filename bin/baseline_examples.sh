if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

echo "VKS Baseline"
$1 ./examples/baseline.py \
	--verbose

echo "KPP Baseline"
$1 ./examples/baseline.py \
  --dataset KPP \
  --data_dir ./data/KPP.npz \
  --verbose

echo "EE Baseline"
$1 ./examples/baseline.py \
  --dataset EE \
  --data_dir ./data/EulerEqs.npz \
  --verbose

echo "FIB Baseline"
$1 ./examples/baseline.py \
  --dataset FIB \
  --data_dir ./data/FIB.npz \
	--time 149 \
  --verbose


cp ./out/baseline_examples/*.gif ./doc/img/
