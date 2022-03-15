bk_ak_path="./autokeras_backup"
bk_kt_path="./kerastuner_backup"

site_path=$1

if  [ ! -n "$site_path" ] ;then
    echo "You NEED to input a site-package path!"
    exit 1
fi

ak_path=$site_path'/autokeras/.'
kt_path=$site_path'/kerastuner/.'

if [ ! -d $bk_ak_path ] || [ ! -d $bk_kt_path ];then
    mkdir $bk_ak_path
    mkdir $bk_kt_path
    cp -r $ak_path $bk_ak_path'/.'
    cp -r $kt_path $bk_kt_path'/.'
else
    cp -r $bk_ak_path'/.' $ak_path
    cp -r $bk_kt_path'/.' $kt_path
fi
