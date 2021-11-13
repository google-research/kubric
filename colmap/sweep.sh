SUBFOLDERS=`gsutil ls $1`
for INPUT in $SUBFOLDERS
do
  OUTPUT=${folder/"input"/"output"}
  ./launch.sh $INPUT $OUTPUT
done