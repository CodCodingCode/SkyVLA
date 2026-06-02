"""Gymnasium tasks over IndoorSim + the GS coverage reward."""

__all__ = ["GSCoverageEnv", "PhysicsCoverageEnv"]


def __getattr__(name):
    # Lazy: importing PhysicsCoverageEnv must not pull in the gsplat-dependent
    # GSCoverageEnv (the physics env uses geometric coverage for now).
    if name == "PhysicsCoverageEnv":
        from .physics_coverage_env import PhysicsCoverageEnv
        return PhysicsCoverageEnv
    if name == "GSCoverageEnv":
        from .coverage_env import GSCoverageEnv
        return GSCoverageEnv
    raise AttributeError(name)
