"""
This file takes raw hand data from MediaPipe
and converts it into numbers for the AI model.

1 hand = 21 points × 3 (x,y,z) = 63
+ 2 position numbers = 65

2 hands = 130 numbers per frame
Dynamic sign = 30 frames × 130 numbers
"""

import math
import numpy as np

NUM_POINTS = 21                 # 21 points in one hand
PER_HAND = NUM_POINTS * 3 + 2  
PER_FRAME = PER_HAND * 2       
SEQ_LEN = 30                   # 30 frames for dynamic signs


# One hand → 65 features
def hand_features(hand):
    pts = np.asarray(hand,dtype=np.float64)  # Convert hand into NumPy numbers
    wrist = pts[0]                            # Point 0 = wrist
    rel = pts - wrist                         # Make wrist the reference point

    # Find the farthest point to normalize hand size
    max_dist = np.sqrt((rel ** 2).sum(axis=1)).max()

    if max_dist < 1e-6:                       # Avoid dividing by zero
        max_dist = 1.0

    shape = (rel / max_dist).reshape(-1)      # Normalize → 63 shape numbers
    position = [wrist[0] - 0.5, wrist[1] - 0.5]  # Wrist position → 2 numbers

    return np.concatenate([shape, position])  # 63 + 2 = 65


# One frame → 130 features
def frame_features(frame):
    """One RAW frame (list of 0, 1 or 2 hands) -> 130 numbers."""

    # Sort hands left-to-right and keep maximum 2
    hands = sorted(frame or [], key=lambda h: h[0][0])[:2]

    zeros = np.zeros(PER_HAND)                # 65 zeros for a missing hand

    slot0 = hand_features(hands[0]) if len(hands) > 0 else zeros
    slot1 = hand_features(hands[1]) if len(hands) > 1 else zeros

    return np.concatenate([slot0, slot1])     # 65 + 65 = 130


# Select 30 frames evenly from the video
def pick_frames(frames, count=SEQ_LEN):
    """Pick 'count' frames evenly: index i = floor(i*(n-1)/29 + 0.5)."""

    n = len(frames)

    if n == 0:
        return []

    return [frames[math.floor(i * (n - 1) / (count - 1) + 0.5)] for i in range(count)]


# Dynamic sign: many frames → 30 × 130 features
def dynamic_features(frames):
    """Dynamic sign: list of RAW frames -> array of shape (30, 130)."""

    return np.stack([frame_features(f) for f in pick_frames(frames)])


# Static sign: one frame → 130 features
def static_features(frame):
    """Static sign: one RAW frame -> array of shape (130,)."""
    return frame_features(frame)


# Check if a frame has at least one hand
def has_hands(frame):
    return isinstance(frame, list) and len(frame) > 0