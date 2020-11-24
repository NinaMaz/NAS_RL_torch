""" Trajectory transformations used by neural architecture search. """
import numpy as np


class RewardsAsValueTargets:
    """ Sets value targets that are equal to rewards. """

    def __call__(self, trajectory):
        if not np.all(trajectory["resets"]):
            raise ValueError(f"all resets must be True, got {trajectory['resets']}")
        trajectory["value_targets"] = trajectory["rewards"].astype(np.float32)


class PadActions:
    """ Pads actions to maximal length. """

    def __call__(self, trajectory):

        if isinstance(trajectory["actions"], list):
            actions = trajectory["actions"]
        else:
            actions = trajectory["actions"].tolist()

        for i, a in enumerate(actions):
            if not isinstance(a, np.ndarray):
                actions[i] = np.array(a)

        assert isinstance(actions, list), type(actions)
        assert all(a.ndim == 1 for a in actions), [a.ndim for a in actions]
        maxlen = max(a.shape[0] for a in actions)
        trajectory["actions"] = [
            np.pad(a, [(0, maxlen - a.shape[0])], mode="constant")
            for a in actions
        ]


class AsArray:
    """ Converts trajectory values to np.ndarray. """

    def __call__(self, trajectory):
        for key, val in filter(lambda kv: kv[0] != "state", trajectory.items()):
            trajectory[key] = np.asarray(val)


class TileValueTargets:
    """ Value targets are tiled to have the same shape as actions. """

    def __call__(self, trajectory):
        actions_shape = trajectory["actions"].shape
        # pylint: disable=invalid-name
        na, nv = len(actions_shape), trajectory["value_targets"].ndim
        value_targets = trajectory["value_targets"][(...,) + (None,) * (na - nv)]
        trajectory["value_targets"] = np.tile(value_targets,
                                              actions_shape[-(na - nv):])


def get_nas_transforms():
    """ Returns trajectory transformations for NAS. """
    return [
        PadActions(),
        AsArray(),
        RewardsAsValueTargets(),
        TileValueTargets()
    ]
