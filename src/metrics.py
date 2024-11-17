from scipy.spatial import KDTree
import pandas as pd
from params import PARTICLE_RADIUS, PARTICLE_TYPES, BETA, WEIGHTS

def compute_metrics(reference_points, reference_radius, candidate_points):
    """
    Compute the number of true positives, false positives and false negatives between two sets of points.
    """
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = ref_tree.query_ball_tree(candidate_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn

def result_frame_to_dict(df):
    """
    transform a dataframe to a dictionary of the form:
    {
        experiment1: {
            particle_type1: np.array([[x1, y1, z1], [x2, y2, z2], ...]),
            particle_type2: np.array([[x1, y1, z1], [x2, y2, z2], ...]),
            ...
        },
        experiment2: {
            particle_type1: np.array([[x1, y1, z1], [x2, y2, z2], ...]),
            particle_type2: np.array([[x1, y1, z1], [x2, y2, z2], ...]),
            ...
        },
        ...
    }
    """
    submission = {}
    for experiment in df["experiment"].unique():
        submission[experiment] = {}
        for particle_type in df["particle_type"].unique():
            submission[experiment][particle_type] = df.query(
                f"experiment == '{experiment}' and particle_type == '{particle_type}'")[["x", "y", "z"]].values
    return submission

def compute_score(submission_path, reference_path):
    """
    Compute the score of a submission compared to a reference.
    """
    submission = pd.read_csv(submission_path)
    reference = pd.read_csv(reference_path)
    submission = result_frame_to_dict(submission)
    reference = result_frame_to_dict(reference)

    results = {}
    for particle_type in PARTICLE_TYPES:
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in submission:
        for particle_type in submission[experiment]:
            reference_radius = PARTICLE_RADIUS[particle_type]
            reference_points = reference[experiment][particle_type]
            candidate_points = submission[experiment][particle_type]
            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)
            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + BETA**2) * (precision * recall) / (BETA**2 * precision + recall) if (precision + recall) > 0 else 0.0
        aggregate_fbeta += fbeta * WEIGHTS.get(particle_type, 1.0)

    aggregate_fbeta = aggregate_fbeta / sum(WEIGHTS.values())
    return aggregate_fbeta
