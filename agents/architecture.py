"""
Observations
- Minimap features
  - height_map: terrain level
  - visibility: parts of map that are hidden/visible/been seen
  - creep: parts with zerg creep
  - camera: parts visible in screen layers
  - player_id: who owns the units
  - player_relative: units that are friendly vs hostile
  - selected: selected units

- Screen
  - height_map
  - visibility
  - creep
  - power: parts with protoss power, only shows yours
  - player_id
  - player_relative
  - unit_type: unit type id
  - selected
  - hit_points: how many hit points the units has
  - energy: how much energy the unit has
  - shields: how much shields the unit has (only protoss)
  - unit_density: how many units are in this pixel
  - unit_density_aa: anti_aliased version

- Structured (non-spatial)
  - player_id
  - minerals
  - vespene
  - food used (supply)
  - food cap
  - food used by army
  - food used by workers
  - idle worker count
  - army count
  - warp gate count (protoss)
  - larva count (zerg)
"""

from keras.models import Model
from keras import layers, Input

# Feature layers
minimap_input = Input(shape=(64, 64), dtype='float32')
screen_input = Input(shape=(64, 64), dtype='float32')
(preprocessed_minimap, preprocessed_screen) = input_pre_processing(minimap_input, screen_input)
non_spatial_input = Input() # TODO: decide on non-spatial features

convolved_minimap = layers.Conv2D(filters=16, kernel_size=(5, 5))(preprocessed_minimap)
convolved_minimap = layers.Conv2D(filters=32, kernel_size=(3, 3))(convolved_minimap)
convolved_screen = layers.Conv2D(filters=16, kernel_size=(5, 5))(preprocessed_screen)
convolved_screen = layers.Conv2D(filters=32, kernel_size=(3, 3))(convolved_screen)

# State reperesentation
state_representation = layers.concatenate([convolved_minimap,  convolved_screen, non_spatial_input])

# Output
value = layers.Dense(units=256, activation="relu")(state_representation)
non_spatial_action_policy = layers.Dense(units=1)(value)
spatial_action_policy = layers.Conv2D(filters=1, kernel_size=(1, 1))(state_representation)

def input_pre_processing(minimap, screen):
  """
  Embed feature layers containing categorical values into a continuous space (OHE followed by
  1x1 convolution), and log transform layers containing numerical values.
  """

  # TODO: Transpose?

