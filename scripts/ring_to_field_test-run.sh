if [ ! $# -eq 1 ]
then
  echo -e "Please specify the bitwidth"
  echo "Usage: ring_to_field_test-run.sh [16|32|40|48|56|60|64]"
else
  ./build/bin/ring_to_field_test 1 $1 & ./build/bin/ring_to_field_test 2 $1
fi