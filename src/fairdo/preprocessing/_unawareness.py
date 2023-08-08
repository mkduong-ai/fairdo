from fairdo.preprocessing.base import Preprocessing

# third party
import numpy as np


class Unawareness(Preprocessing):
    """
    Does not work with aif360 (!)

    Fairness through unawareness
    Removes column of protected attribute
    """

    def __init__(self, protected_attribute=None, label=None, random_state=None):
        super().__init__(protected_attribute=protected_attribute, label=label)
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

        self.transformed_data = self.dataset.drop(columns=self.protected_attribute)
        return self.transformed_data
