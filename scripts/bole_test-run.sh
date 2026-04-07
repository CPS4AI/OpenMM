if [ ! $# -eq 2 ]
then
  echo -e "Please specify the flags"
  echo "Usage: bole_test.sh [0|1] [0|1]"
  echo "first flag represents two secret-share vector"
  echo "second flag represents do-Truncate after multiplication"
else
  ./build/bin/bole_test 1 $1 $2 & ./build/bin/bole_test 2 $1 $2
fi