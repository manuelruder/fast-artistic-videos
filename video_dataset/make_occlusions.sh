if [ ! -f ./consistencyChecker/consistencyChecker ]; then
  if [ ! -f ./consistencyChecker/Makefile ]; then
    echo "Consistency checker makefile not found."
    exit 1
  fi
  cd consistencyChecker/
  make
  cd ..
fi



if [ "$#" -le 0 ]; then
   echo "Usage: ./make_occlusions.sh <folder>"
   echo -e "\folder:\tFolder where dataset files including flow files (in a flow subdirectory) are stored."
   exit 1
fi

folder=$1

for D in `find $folder -type d`
do
  folderName=$D/flow
  echo $folderName
  for i in $(seq 1 30000)
  do
    j=$[ $i - 1 ]
    id=$(printf "%04d" $i)
    jd=$(printf "%04d" $j)
    if [ -a "${folderName}/${id}_${jd}.flo" ] && [ -a "${folderName}/${jd}_${id}.flo" ]; then
      ./consistencyChecker/consistencyChecker "${folderName}/${id}_${jd}.flo" "${folderName}/${jd}_${id}.flo" "${folderName}/reliable_${id}_${jd}.pgm"
      ./consistencyChecker/consistencyChecker "${folderName}/${jd}_${id}.flo" "${folderName}/${id}_${jd}.flo" "${folderName}/reliable_${jd}_${id}.pgm"
    fi
    if [ -a "${folderName}/s_${id}_${jd}.flo" ] && [ -a "${folderName}/s_${jd}_${id}.flo" ]; then
      ./consistencyChecker/consistencyChecker "${folderName}/s_${id}_${jd}.flo" "${folderName}/s_${jd}_${id}.flo" "${folderName}/reliable_${id}_${jd}.pgm"
      ./consistencyChecker/consistencyChecker "${folderName}/s_${jd}_${id}.flo" "${folderName}/s_${id}_${jd}.flo" "${folderName}/reliable_${jd}_${id}.pgm"
    fi
  done
done
