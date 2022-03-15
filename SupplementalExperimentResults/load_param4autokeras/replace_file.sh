tuner_path="./replace_file/tuner.py"
utils_path="./replace_file/utils.py"

site_path=$1

if  [ ! -n "$site_path" ] ;then
    echo "You NEED to input a site-package path!"
    exit 1
fi


cp -r $tuner_path $site_path'/autokeras/engine/tuner.py'
cp -r $utils_path $site_path'/autokeras/utils/utils.py'