from .base import IndoorSim, Frame  # torch-free

__all__ = ["IndoorSim", "Frame", "SyntheticRoom", "DronePhysics"]


def __getattr__(name):
    # Lazy imports so torch-free consumers (e.g. the habitat-env physics drone)
    # don't drag in SyntheticRoom's torch dependency.
    if name == "SyntheticRoom":
        from .synthetic_room import SyntheticRoom
        return SyntheticRoom
    if name == "DronePhysics":
        from .drone_body import DronePhysics
        return DronePhysics
    raise AttributeError(name)
