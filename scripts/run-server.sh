. scripts/common.sh

if [ ! $# -eq 1 ]
then
  echo -e "${RED}Please specify the network to run.${NC}"
  echo "Usage: run-server.sh [sqnet|resnet50|densenet121]"
else
  if ! contains "sqnet resnet50 densenet121" $1; then
    echo -e "Usage: run-server.sh ${RED}[sqnet|resnet50|densenet121]${NC}"
	exit 1
  fi
  # create a data/ to store the Ferret output
  mkdir -p data
  ls -lh pretrained/$1_model_scale12.inp
  echo -e "Runing ${GREEN}build/bin/$1${NC}, which might take a while...."
  cat pretrained/$1_model_scale12.inp | build/bin/$1 r=1 k=$FXP_SCALE ell=$SS_BITLEN nt=$NUM_THREADS p=$SERVER_PORT 1>$1_server.log
  echo -e "Computation done, check out the log file ${GREEN}$1_server.log${NC}"
fi
