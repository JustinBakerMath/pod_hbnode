if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

echo "VKS: VAE NODE"
$1 ./examples/vae.py \
	--dataset VKS \
	--data_dir ./data/VKS.pkl \
	--tr_ind 75 \
	--val_ind 100 \
	--eval_ind 200 \
	--model NODE \
	--verbose

echo "VKS: VAE HBNODE"
$1 ./examples/vae.py \
	--dataset VKS \
	--data_dir ./data/VKS.pkl \
	--tr_ind 75 \
	--val_ind 100 \
	--eval_ind 200 \
	--model HBNODE \
	--verbose

cp ./out/vae_examples/*.gif ./doc/img/
