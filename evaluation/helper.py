from aif360.datasets import BinaryLabelDataset


class PreprocessingWrapper:
    """
    Wrapper Class for pre-processing methods that take a
    pandas DataFrame and return a pandas DataFrame instead
    of a BinaryLabelDataset.
    """
    def __init__(self, preprocessing):
        self.preprocessing = preprocessing
        self.__class__.__name__ = type(preprocessing).__name__

    def fit_transform(self, dataset_train: BinaryLabelDataset) -> BinaryLabelDataset:
        # z_train already included in x_train
        transformed_df_dataset = self.preprocessing.fit_transform(dataset_train.convert_to_dataframe()[0])

        transformed_dataset = BinaryLabelDataset(df=transformed_df_dataset,
                                                 protected_attribute_names=dataset_train.protected_attribute_names,
                                                 label_names=dataset_train.label_names)

        return transformed_dataset
