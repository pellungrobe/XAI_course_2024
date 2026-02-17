from ..bbox import AbstractBBox
from ..dataset import Dataset
from ..encoder_decoder import EncDec
from .neighborhood_generator import NeighborhoodGenerator

from ..dataset.tabular_dataset import TabularDataset
import pandas as pd
import numpy as np

__all__ = ["NeighborhoodGenerator", "RandomGenerator"]


class RandomGenerator(NeighborhoodGenerator):
    """
    Random Generator creates neighbor instances by generating random values starting from the instance to explain
    """

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1, mutpb=0.1):
        """
        :param bbox: the Black Box model to explain
        :param dataset: the dataset with the descriptor of the original dataset
        :param encoder: an encoder to transfrom the data from/to the black box model
        :param ocr: acronym for One Class Ratio, it is the ratio of the number of instances of the minority class
        """
        super().__init__(bbox, dataset, encoder, ocr)
        self.generated_data = None
        self.mutpb = mutpb


    def generate(self, x, num_instances, descriptor, encoder):
        """
        random generation of new instances. The starting instance x is only used to detect the value type of each feature, in order
        to generate new values only for numeric features.

        :param x[dict]: the starting instance from the real dataset
        :param num_instances[int]: the number of instances to generate
        :param descriptor[dict]: data descriptor as generated from a Dataset object
        :param encoder[EncDec]: the encoder eventually to encode the instance
        The list (or range) associated to each key is used to randomly choice an element within the list.

        :return [instances]: an array of instances with the new data generated
        """

        generated_list = np.array([])
        columns = np.empty(len(x), dtype=object)

        for n in range(num_instances):
            instance = self.generate_synthetic_instance(x, self.mutpb)

            # append instance array to generated_list
            if generated_list.size == 0:
                generated_list = instance
            else:
                generated_list = np.vstack((generated_list, instance))

        balanced_list = super().balance_neigh(x, generated_list, num_instances)
        self.generated_data = balanced_list

        return balanced_list

