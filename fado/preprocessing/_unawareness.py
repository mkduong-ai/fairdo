import numpy as np

from fado.preprocessing._base import Preprocessing
from fado.metrics import statistical_parity_absolute_difference


class Unawareness(Preprocessing):
    """
    Does not work with aif360 (!)

    Fairness through unawareness
    Removes column of protected attribute
    """

    def __init__(self,
                 protected_attribute=None, label=None,
                 drop_protected_attribute=False,
                 drop_label=False,
                 drop_features=False,
                 dim_reduction=False, n_components=2, random_state=None):
        super().__init__(protected_attribute=protected_attribute, label=label,
                         drop_protected_attribute=drop_protected_attribute,
                         drop_label=drop_label,
                         drop_features=drop_features,
                         dim_reduction=dim_reduction, n_components=n_components)
        self.random_state = random_state
        np.random.seed(random_state)

    def transform(self):
        """

        Parameters
        ----------
        Returns
        -------

        """
        if self.dataset is None:
            raise Exception('Model not fitted.')
        if self.protected_attribute is None:
            raise Exception('Protected attribute not given.')

        return self.transformed_data.drop(columns=self.protected_attribute)
