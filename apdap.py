"""
Library with functions to add a point defect as a pseudo-particle in a LAMMPS dump file
"""


from functools import partial
import ovito
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats


class Structures:

    """
    Enumerator class with structure integer labels
    """

    OTHER = 0
    FCC = 1
    HCP = 2
    BCC = 3
    ICO = 4
    SC = 5
    CUBIC_DIAMOND = 6
    HEX_DIAMOND = 7
    GRAPHENE = 8

STRUCTURES = Structures()


def add_defect_modifier(frame: int, data: ovito.data.DataCollection, threshold: float) -> None:

    """
    Modifier that adds pseudo-particle at point defect
    Does polyhedral template matching, deletes structured atoms, deletes small clusters
    Checks if largest cluster goes over a boundary by checking if position standard deviations are above provided
    threshold
    If they are, split cluster into a small and a large cluster using KMeans clustering and place defect in the center
    of the larger sub-cluster
    :param frame: frame to add modifier
    :param data: data collection to modify
    :param threshold: standard deviation threshold
    """

    # get positions of largest cluster, assume small clusters have already been deleted
    cluster_positions = data.particles['Position'][...]

    # grab box lengths from cell
    cell = data.cell[...]
    box_lengths = cell[:, 0:3].diagonal()

    # get mean position and determine which dimensions split the cluster
    mean_position = np.mean(cluster_positions, axis=0)
    position_std = np.std(cluster_positions, axis=0)
    split_dimensions = [position_std > threshold][0]

    # loop through dimensions, correct mean position if cluster splits that dimension
    for index, split in enumerate(split_dimensions):
        if not split:
            continue
        # fit KMeans to provided positions, labeling each sub-cluster with an integer
        position_distribution = cluster_positions[:, index].copy()
        kmeans = KMeans(n_clusters=2, n_init='auto').fit(position_distribution.reshape(-1, 1))
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        # get largest cluster label
        largest_cluster_index = stats.mode(labels, keepdims=False).mode

        # either add or subtract a box length based on which cluster is larger
        displacement = box_lengths[index]
        if centers[largest_cluster_index] < centers[largest_cluster_index - 1]:
            displacement *= -1
        for j, position in enumerate(position_distribution):
            prediction = kmeans.predict(position.reshape(-1, 1))
            if prediction != largest_cluster_index:
                position_distribution[index] += displacement

        # correct position
        mean_position[index] = np.mean(position_distribution)

    # add particle of type 0
    data.particles_.add_particle(mean_position)


def create_defect_pipeline(
        file_name: str,
        rmsd_cutoff: float,
        threshold: float
) -> ovito.pipeline.Pipeline:

    """
    Function for creating pipeline with only defect
    :param file_name: original dump file
    :param rmsd_cutoff: RMSD cutoff for polyhedral template matching
    :param threshold: standard deviation threshold for defect modifier
    :return: pipeline with just defect
    """

    # initialize pipeline and add the polyhedral template matching modifier
    pipeline = ovito.io.import_file(file_name)
    pipeline.modifiers.append(ovito.modifiers.PolyhedralTemplateMatchingModifier(rmsd_cutoff=rmsd_cutoff))
    initial_frame = pipeline.compute(0)

    # create structure dictionary so we can find the most prominent structure
    structure_dict = {
        key: val for key, val in initial_frame.attributes.items() if 'PolyhedralTemplateMatching.counts' in key
    }

    # find the most prominent structure based on the largest counts
    max_counts = -1
    most_prominent_structure = None
    for structure, counts in structure_dict.items():
        if counts > max_counts:
            most_prominent_structure = structure
            max_counts = counts

    # get corresponding integer label for that structure
    _, __, structure = most_prominent_structure.split('.')
    structure_integer_label = STRUCTURES.__getattribute__(structure)

    # initialize defect modifier with provided threshold
    defect_modifier = partial(add_defect_modifier, threshold=threshold)

    # select most prominent structure
    # delete atoms with most prominent structure
    # sort atoms by clusters
    # delete small clusters
    # add defect pseudo-particle
    # delete all but the defect pseudo-particle
    modifiers = [
        ovito.modifiers.ExpressionSelectionModifier(expression=f'StructureType=={structure_integer_label}'),
        ovito.modifiers.DeleteSelectedModifier(),
        ovito.modifiers.ClusterAnalysisModifier(sort_by_size=True),
        ovito.modifiers.ExpressionSelectionModifier(expression='Cluster!=1'),
        ovito.modifiers.DeleteSelectedModifier(),
        defect_modifier,
        ovito.modifiers.ExpressionSelectionModifier(expression='ParticleType!=0'),
        ovito.modifiers.DeleteSelectedModifier()
    ]
    for modifier in modifiers:
        pipeline.modifiers.append(modifier)

    return pipeline


def create_combined_pipeline(file_name, rmsd_cutoff=0.12, threshold=5.0):

    """
    Function for creating a pipeline with original atoms + defect
    :param file_name: LAMMPS dump file
    :param rmsd_cutoff: rmsd cutoff for Polyhedral Template Matching
    :param threshold: standard deviation threshold for defect modifier
    :return:
    """

    # get the pipeline without the defect
    unmodified_pipeline = ovito.io.import_file(file_name)

    # create pipeline with defect
    pipeline = create_defect_pipeline(file_name, rmsd_cutoff, threshold)

    # combine the two datasets
    unmodified_pipeline.modifiers.append(ovito.modifiers.CombineDatasetsModifier(source=pipeline.data_provider))

    return unmodified_pipeline
