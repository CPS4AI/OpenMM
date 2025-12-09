if [ ! $# -eq 1 ]
then
  echo -e "Please specify the bitwidth"
  echo "Usage: field_to_ring_test-run.sh [16|32|40|48|56|60|64]"
else
  ./build/bin/field_to_ring_test 1 $1 & ./build/bin/field_to_ring_test 2 $1
fi