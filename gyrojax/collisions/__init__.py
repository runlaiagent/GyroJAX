"""
Collision operators for GyroJAX δf PIC.

All operators act on the GCState and return an updated GCState
(modified weights and/or velocities).

Collision model selection via cfg.collision_model:
  'none'       — no collisions
  'krook'      — BGK/Krook: dw/dt = -nu*w
  'lorentz'    — pitch-angle scattering (stochastic v∥ kicks)
  'dougherty'  — model Fokker-Planck (GX-compatible, conservative)
"""

from gyrojax.collisions.operators import apply_collisions
