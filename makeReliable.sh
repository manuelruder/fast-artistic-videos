#if [ ! -f ./consistencyChecker/consistencyChecker ]; then
#  if [ ! -f ./consistencyChecker/Makefile ]; then
#    echo "Consistency checker makefile not found."
#    exit 1
#  fi
#  cd consistencyChecker/
#  make
#  cd ..
#fi

filePattern=frame_%04d.png
folderName=/home/rudera/fast-artistic-videos/TerraX/Zuendung
startFrame=270 #${3:-1}
stepSize=${4:-1}

i=$[$startFrame]
j=$[$startFrame + $stepSize]

mkdir -p "${folderName}_reliable"

while true; do
  frame1=$(printf "${folderName}_frames/$filePattern" "$i")
  frame2=$(printf "${folderName}_frames/$filePattern" "$j")
  flo1=$(printf "forward_%04d_%04d.flo" "$i" "$j")
  flo2=$(printf "backward_%04d_%04d.flo" "$j" "$i")
  rel1=$(printf "reliable_%04d_%04d.pgm" "$i" "$j")
  rel2=$(printf "reliable_%04d_%04d.pgm" "$j" "$i")
  if [ -a $frame2 ]; then
    if [ -a ${folderName}_flow/$flo1 ] && [ -a ${folderName}_flow/$flo2 ]; then
      ffmpeg -y -i $frame1 temp.ppm >/dev/null 2>&1
      /misc/student/rudera/consistencyChecker/consistencyChecker "${folderName}_flow/$flo1" "${folderName}_flow/$flo2" "${folderName}_reliable/$rel1" "/home/rudera/fast-artistic-videos/temp.ppm"
      ffmpeg -y -i $frame2 temp.ppm >/dev/null 2>&1
      /misc/student/rudera/consistencyChecker/consistencyChecker "${folderName}_flow/$flo2" "${folderName}_flow/$flo1" "${folderName}_reliable/$rel2" "/home/rudera/fast-artistic-videos/temp.ppm"
    fi
  else
    break
  fi
  i=$[$i +1]
  j=$[$j +1]
done
