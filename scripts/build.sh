. scripts/common.sh

for deps in eigen3 emp-ot emp-tool hexl SEAL-3.7
do
  if [ ! -d $BUILD_DIR/include/$deps ] 
  then
	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
	exit 1
  fi
done

for deps in zstd.h 
do
  if [ ! -f $BUILD_DIR/include/$deps ] 
  then
	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
	exit 1
  fi
done

cd $BUILD_DIR/
cmake .. -DCMAKE_BUILD_TYPE=Release -DSCI_BUILD_NETWORKS=OFF -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DCMAKE_PREFIX_PATH=$BUILD_DIR -DUSE_APPROX_RESHARE=ON
make ring_to_field_test -j8
make field_to_ring_test -j8
make ring_extension_test -j8
make bole_test -j8
make resnet50 -j8 & make sqnet -j8 & make densenet121 -j8
# make bole_test -j8
# make bole_then_truncate_test -j8