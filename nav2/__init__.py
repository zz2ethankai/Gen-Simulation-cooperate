"""Unified Nav2 package for the split Isaac/Nav2 deployment.

Current layout:

- ``nav2.runtime``: workflow-side navigation session manager
- ``nav2.bridge``: ROS topic bridge and clock helpers
- ``nav2.mapgen``: static map export and bootstrap artifact generation
- ``nav2.container``: Dockerfile and entrypoint for the nav2 service
"""
