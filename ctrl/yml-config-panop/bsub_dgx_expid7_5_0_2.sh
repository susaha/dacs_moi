DEBUG='False'
YML_FNAME='ctrl/yml-config-panop/exp_common.yml'
TARGET_DIR='/raid/susaha/datasets'
MS_EXP_ROOT='/raid/susaha/experiments/CVPR2022/cvpr2022'

STR1=$(date +%m-%Y)
STR11="exproot_"
EXPROOT="$STR11$STR1"
STR2=$(date +%d-%m-%Y)
STR22="phase_"
PHASENAME="$STR22$STR2"
subp=$(date +%T_%6N | sed "s/:/_/g")
STR3=${subp//[_]/-}
STR33="subphase_"
SUB_PHASENAME="$STR33$STR3"

BSUB_SCRIPT_FNAME='bsub_scripts/bsub_dgx_expid7_5_0_2.sh' # TODO

# TRAINING
#python -m torch.distributed.launch --nproc_per_node=4 main_panop.py \
CUDA_VISIBLE_DEVICES=0 python main_panop_dacs.py \
--debug $DEBUG \
--yml_fname $YML_FNAME \
BSUB_SCRIPT_FNAME $BSUB_SCRIPT_FNAME \
DATA_ROOT $TARGET_DIR \
MS_EXP_ROOT $MS_EXP_ROOT \
EXP_ROOT $EXPROOT \
PHASE_NAME $PHASENAME \
SUB_PHASE_NAME $SUB_PHASENAME