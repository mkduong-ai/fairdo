# Needed to specify settings for the used heuristic
from functools import partial
# Dataset
from fairdo.utils.dataset import load_data
# Metric to optimize on
from fairdo.metrics import statistical_parity_abs_diff
# Load GA and the wrapper for pre-processing data
from fairdo.optimize.geneticalgorithm import genetic_algorithm
from fairdo.preprocessing import HeuristicWrapper


df, label, protected_attributes = load_data('compas')
# Declare certain settings for the genetic algorithm
# It is also possible to use different genetic operators.
heuristic = partial(genetic_algorithm,
                    pop_size=100,
                    num_generations=500)
disc_measure = statistical_parity_abs_diff
# Initialize HeuristicWrapper
preprocessor = HeuristicWrapper(heuristic=heuristic,
                                disc_measure=disc_measure,
                                protected_attribute=protected_attributes,
                                label=label)
                                
# Create pre-processing instance
preprocessor.fit(df)
# Remove samples to yield a fair dataset
df_fair = preprocessor.transform()

print(df_fair)
