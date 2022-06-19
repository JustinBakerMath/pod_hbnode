if [ -z $1 ] ; then
	echo  "Must provide python executable: ['python','python3']"
	exit 1
fi

echo "Fetching VKS Data"
gdown 109kG16CUXIW5w9WLYAGnCLVi_bJPctiz -O ./data/VKS.pkl

echo "Generating KPP Data"
$1 ./data/kpp_generator.py

echo "Generating EE Data"
$1 ./data/ee_generator.py

echo "Generating FIB Data"
g++ ./data/fiber/fiber_automate.c -lm -o ./data/fiber/pde
./data/fiber/pde
$1 ./data/fib_generator.py
#rm ./data/fiber/*.dat
#rm ./data/fiber/*.npz
