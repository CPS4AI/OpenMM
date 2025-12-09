. scripts/common.sh

if [ ! $# -eq 1 ]
then
  echo -e "${RED}Please specify the network to run.${NC}"
  echo "Usage: run-client.sh [sqnet|resnet50|densenet121]"
else
  if ! contains "sqnet resnet50 densenet121" $1; then
    echo -e "Usage: run-client.sh ${RED}[sqnet|resnet50|densenet121]${NC}"
	exit 1
  fi
  # create a data/ to store the Ferret output
  mkdir -p data
  echo -e "Runing ${GREEN}build/bin/$1${NC}, which might take a while...."
  cat pretrained/$1_input_scale12_pred*.inp | build/bin/$1 r=2 k=$FXP_SCALE ell=$SS_BITLEN nt=$NUM_THREADS ip=$SERVER_IP p=$SERVER_PORT 1>$1_client.log
  echo -e "Computation done, check out the log file ${GREEN}$1-client.log${NC}"
fi
