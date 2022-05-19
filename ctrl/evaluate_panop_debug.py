import pickle


def main():

    filename = '/media/suman/DATADISK2/apps/experiments/CVPR2022/cvpr2022/EULER-EXP/exproot_08-2021/' + \
               'phase_10-08-2021/subphase_15-44-22-914575/checkpoints/eval_logs_model_30000_0/debug_test/eval_panop_debug.pkl'





    infile = open(filename, 'rb')
    metrics = pickle.load(infile)
    semantic_metric = metrics['semanitc']
    instance_metric = metrics['instance']
    panoptic_metric = metrics['panoptic']
    infile.close()
    print()
    # semantic_results = semantic_metric.evaluate()
    instance_results = instance_metric.evaluate()
    # panoptic_results = panoptic_metric.evaluate()

if __name__ == '__main__':
    main()