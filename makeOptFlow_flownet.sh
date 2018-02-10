# Specify the path to the flownet script here.
flowCommandLine="bash run-flownet-multiple.sh"

if [ ! -f ./consistencyChecker/consistencyChecker ]; then
  if [ ! -f ./consistencyChecker/Makefile ]; then
    echo "Consistency checker makefile not found."
    exit 1
  fi
  cd consistencyChecker/
  make
  cd ..
fi

filePattern=$1
folderName=$2
startFrame=${3:-1}

wait_for_file() {
   local filename=$1
   while [ ! -f "$filename" ]; do 
   sleep 1 
   done
}

if [ "$#" -le 1 ]; then
   echo "Usage: ./makeOptFlow <filePattern> <outputFolder> [<startNumber>]"
   echo -e "\tfilePattern:\tFilename pattern of the frames of the videos."
   echo -e "\toutputFolder:\tOutput folder."
   echo -e "\tstartNumber:\tThe index of the first frame. Default: 1"
   exit 1
fi

i=$[$startFrame]

mkdir -p "${folderName}"

# First create the flow list
if [ -f ${folderName}/flow_list.txt ] ; then
  rm ${folderName}/flow_list.txt
fi
while [ 1 ]; do
  j=$[ $i - 1 ]
  file1=$(printf "$filePattern" "$i")
  file2=$(printf "$filePattern" "$j")
  if [ -a $file2 ] && [ -a $file1 ]; then
    if [ ! -f ${folderName}/forward_${j}_${i}.flo ]; then
      echo "$file2" "$file1" "${folderName}/forward_${j}_${i}.flo" >> ${folderName}/flow_list.txt
    fi
    if [ ! -f ${folderName}/backward_${i}_${j}.flo ]; then
      echo "$file1" "$file2" "${folderName}/backward_${i}_${j}.flo" >> ${folderName}/flow_list.txt
    fi
  fi
  if [ ! -f $file1 ]; then
    break
  fi 
  i=$[$i + 1]
done

set -e

# Run flownet
if [ -f ${folderName}/flow_list.txt ] ; then
  eval $flowCommandLine ${folderName}/flow_list.txt
fi
