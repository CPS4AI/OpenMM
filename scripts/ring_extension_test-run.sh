if [ ! $# -eq 1 ]
then
  echo -e "Please specify the bitwidth"
  echo "Usage: ring_extension_test-run.sh [1, 64]"
else
  ./build/bin/ring_extension_test 1 $1 & ./build/bin/ring_extension_test 2 $1
fi