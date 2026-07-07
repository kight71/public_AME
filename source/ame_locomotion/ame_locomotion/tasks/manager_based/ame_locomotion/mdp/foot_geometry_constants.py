"""Shared G1 foot geometry constants for planner and foothold rewards."""

# Stock G1 USD ankle-roll collision footprint in the ankle_roll_link frame.
# The collider union spans x=[-0.055, 0.145], y=[-0.035, 0.035], so the
# support polygon is shifted forward from the link origin.
G1_FOOT_LENGTH = 0.20
G1_FOOT_WIDTH = 0.07
G1_FOOT_OFFSET_X = 0.045
G1_FOOT_OFFSET_Y = 0.0
G1_FOOT_N_LONG = 4
G1_FOOT_N_LAT = 3
G1_SOLE_Z_OFFSET = -0.035409145057201385
