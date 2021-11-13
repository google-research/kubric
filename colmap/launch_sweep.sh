SUBFOLDERS=`gsutil ls $1`
for INPUT in $SUBFOLDERS
do
  # OUTPUT=${folder/"input"/"output"}
  # --- remove last "/" from path (problem later)
  INPUT=${INPUT%?}

  # echo $INPUT
  ./launch.sh $INPUT
done