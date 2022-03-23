# orig_ak_path="./autokeras"
# orig_kt_path="./kerastuner"
replace="./replace.txt"
site_path=$1
python_path=$2
tmp1='./Autokeras/engine/tuner.py'
tmp2='./utils/load_test_utils.py'
check_string='your_python_path'


# `site_path' should be sth like: "/xxx/anaconda3/envs/env_name/lib/python3.7/site-packages"
# `python_path' should be like: "/xxx/anaconda3/envs/env_name/bin/python"

if  [ ! -n "$site_path" ] ;then
    echo "You NEED to input a site-package path!"
    exit 1
fi
if  [ ! -n "$python_path" ] ;then
    echo "You NEED to input a python python!"
    exit 1
fi

echo 'Sitepackages_Path:'$site_path
echo 'Python_Path:'$python_path


sed -i "s#$check_string#$python_path#g" $tmp1
sed -i "s#$check_string#$python_path#g" $tmp2

echo 'FINISH: Replace the Python path in DREAM'


for line in `cat $replace`
do
    tmp_line=${line/autokeras/Autokeras}
    tmp_line=${tmp_line/kerastuner/Kerastuner}
    echo $tmp_line $site_path$line
    cp -r '.'$tmp_line $site_path$line
done
echo 'FINISH: Copy Files to AutoKeras & Kerastuner'