# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/cityscapes_evaluation.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import logging
from collections import OrderedDict
import os
import glob

from fvcore.common.file_io import PathManager
from ctrl.utils.panoptic_deeplab import save_annotation


class CityscapesInstanceEvaluator:
    """
    Evaluate Cityscapes instance segmentation
    """
    def __init__(self, output_dir=None, train_id_to_eval_id=None,
                 gt_dir='./datasets/cityscapes/gtFine/val',
                 num_classes=16, DEBUG=None, num_samples=10,
                 dataset_name='Cityscapes', rgb2id=None, input_image_size=None, mapillary_dataloading_style='OURS'):
        """
        Args:
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
            gt_dir (str): path to ground truth annotations (gtFine).
        """
        self.debug = DEBUG
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        if output_dir is None:
            raise ValueError('Must provide a output directory.')
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self._mask_dir = os.path.join(self._output_dir, 'mask')
        if self._mask_dir:
            PathManager.mkdirs(self._mask_dir)
        self._train_id_to_eval_id = train_id_to_eval_id
        self.input_image_size = input_image_size
        self.mapillary_dataloading_style = mapillary_dataloading_style
        self.rgb2id = rgb2id

        self._logger = logging.getLogger(__name__)
        self._logger.info('ctrl/eval_panop/instance.py --> class CityscapesInstanceEvaluator: --> def __init__() --> self._logger : {}'.format(self._logger))

        self._gt_dir = gt_dir
        self._logger.info('ctrl/eval_panop/instance.py --> class CityscapesInstanceEvaluator: --> def __init__() --> self._gt_dir:{}'.format(self._gt_dir))
        self.num_classes = num_classes

    def update(self, instances, image_filename=None):
        pred_txt = os.path.join(self._output_dir, image_filename + "_pred.txt")
        num_instances = len(instances)

        with open(pred_txt, "w") as fout:
            for i in range(num_instances):
                pred_class = instances[i]['pred_class']
                if self._train_id_to_eval_id is not None:
                    pred_class = self._train_id_to_eval_id[pred_class]

                score = instances[i]['score']
                mask = instances[i]['pred_mask'].astype("uint8")
                png_filename = os.path.join(
                    self._mask_dir, image_filename + "_{}_{}.png".format(i, pred_class)
                )

                save_annotation(mask, self._mask_dir, image_filename + "_{}_{}".format(i, pred_class),
                                add_colormap=False, scale_values=True)
                fout.write("{} {} {}\n".format(os.path.join('mask', os.path.basename(png_filename)), pred_class, score))

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        if self._gt_dir is None:
            raise ValueError('Must provide cityscapes path for evaluation.')

        self._logger.info("Evaluating results under {} ...".format(self._output_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._output_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._output_dir, "gtInstances.json")

        # added by suman
        # to adapt the 16 cls setting for instance segmentation evaluation
        labels = None
        # if self.num_classes == 19:
        #     from cityscapesscripts.helpers.labels import labels
        # elif self.num_classes == 16:
        from synthiascripts.cityscapes_labels_16cls import labels
        cityscapes_eval.args.labels = labels


        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._gt_dir)

        self._logger.info('ctrl/eval_panop/instance.py --> class CityscapesInstanceEvaluator: --> def evaluate() --> gt_dir:{}'.format(gt_dir))

        if 'Cityscapes' in self.dataset_name:
            groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
            self._logger.info('len(groundTruthImgList): {}'.format(len(groundTruthImgList)))
        elif 'Mapillary' in self.dataset_name:
            groundTruthImgList = glob.glob(os.path.join(gt_dir, "*.png"))

        if self.debug:
            groundTruthImgList.sort()
            groundTruthImgList=groundTruthImgList[:self.num_samples]


        assert len(groundTruthImgList), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(cityscapes_eval.args.groundTruthSearch)
        predictionImgList = []

        if 'Cityscapes' in self.dataset_name:
            for gt in groundTruthImgList:
                predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        elif 'Mapillary' in self.dataset_name:
            from os import listdir
            from os.path import isfile, join
            predictionImgList = [join(self._output_dir, f) for f in listdir(self._output_dir) if isfile(join(self._output_dir, f))]

        if self.debug:
            predictionImgList.sort()

        results = cityscapes_eval.evaluateImgLists(predictionImgList, groundTruthImgList,
                                                   cityscapes_eval.args,
                                                   dataset_name=self.dataset_name,
                                                   rgb2id=self.rgb2id,
                                                   input_image_size=self.input_image_size,
                                                   mapillary_dataloading_style=self.mapillary_dataloading_style)["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        return ret
