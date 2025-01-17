# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2020-2021 antillia.com Toshiyuki Arai

# 2020/07/31 Based on visualize/vis_utils.py
# 2020/08/15 Updated to suppoert objects_stats
# vis_utils2.py

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import matplotlib
matplotlib.use('Agg')  # Set headless-friendly backend.
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
from visualize.vis_utils import * 
from visualize import shape_utils
from visualize import standard_fields as fields


#2020/07/22 Added filters and detected_objects parameters
def draw_bounding_box_on_image_array_with_filters(filters,          #list of classes
                                     image,
                                     detected_objects, #list of detected_object attribute (id, label, score)
                                     objects_stats,    #
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
  #print("--- draw_bounding_box_on_image_array_with_filters")
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image_with_filters(filters, 
                             image_pil, 
                             detected_objects, # in_out
                             objects_stats,    # in_out 2020/08/15
                             ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


#2020/07/22 Added filters, id and detected_objects parameters
def draw_bounding_box_on_image_with_filters(filters,
                               image,
                               detected_objects, # in_out
                               objects_stats,    # in_out 2020/08/15
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
  #print("--- draw_bounding_box_on_image_with_filters")
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  
  #if thickness > 0:
  #  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
  #             (left, top)],
  #            width=thickness,
  #            fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
    
  # 2024/01/31
  # DeprecationWarning: getsize is deprecated and will be removed in Pillow 10 (2023-07-01). 
  # Use getbbox or getlength instead.
  # (x, y, w, h) = font.getbbox('test')
  try:
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  except:
    # The latest Pillow version
    display_str_heights= [font.getbbox(ds)[3] for ds in display_str_list]

  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height

  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    # 2024/01/31
    try:
      text_width, text_height = font.getsize(display_str)
    except:
      _, _, text_width, text_height = font.getbbox(display_str)

    margin = np.ceil(0.05 * text_height)
 
    #2020/07/22 atlan
    sarray = display_str.split(':')
    classname = ""
    score     = ""
    if len(sarray)>1:
      classname = sarray[0]
      score     = sarray[1]
    
    # check filters is None or not.
    if filters is None or filters == "" or filters == "None":
      #print("----------filters is None")
      id = len(detected_objects) +1
      # id, class, score, x, y, w, h
      #print("{},  {}, {}, {}, {}, {}, {}".format(id, classname, score, 
      #      int(left), int(top), int(right-left), int(bottom-top) ))
      detected_objects.append((id, classname, score, int(left), int(top), int(right-left), int(bottom-top) ))
      # 2020/08/15
      if classname not in objects_stats:
         objects_stats[classname] = 1
      else:
        count = int(objects_stats[classname]) 
        objects_stats.update({classname: count+1})
      #print(objects_stats)
      #  
      
      if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

      draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
      draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill='black',
              font=font)
      text_bottom -= text_height - 2 * margin

    else:
     if classname in filters:
       id = len(detected_objects) +1
       # id, class, score, x, y, w, h
       #print("{},  {}, {}, {}, {}, {}, {}".format(id, classname, score, 
       #     int(left), int(top), int(right-left), int(bottom-top) ))
       detected_objects.append((id, classname, score, int(left), int(top), int(right-left), int(bottom-top) ))

       # 2020/08/15
       if classname not in objects_stats:
          objects_stats[classname] = 1
       else:
         count = int(objects_stats[classname]) 
         objects_stats.update({classname: count+1})
       #print(objects_stats)
       #  
       
       if thickness > 0:
         draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
 
       draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
       draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill='black',
              font=font)
       text_bottom -= text_height - 2 * margin
       
####

#2020/07/22 
#2020/08/15 Returns objects_stats
# (image, detected_objects, objects_stats)
def visualize_boxes_and_labels_on_image_array_with_filters(
    filters,
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then this
      function assumes that the boxes to be plotted are groundtruth boxes and
      plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as normalized
      coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all
      boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  #print("--- vis_utils2  visualize_boxes_and_labels_on_image_array_with_filters")
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100 * scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[(prime_multipler *
                                                   track_ids[i]) %
                                                  len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                  len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  #2020/08/15 
  detected_objects = []
  objects_stats    = {}  
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image, box_to_instance_masks_map[box], color=color)
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image, box_to_instance_boundaries_map[box], color='red', alpha=1.0)
    #2020/07/22
    #print("---  draw_bounding_box_on_image_array_with_filters")
    #print(" xmin {} ymin {} xmax {} ymax {} color {}".format(xmin, ymin, xmax, ymax, color))
    draw_bounding_box_on_image_array_with_filters(
        filters,          # in
        image,
        detected_objects, # in_out
        objects_stats,    # in_out 2020/08/15
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=0 if skip_boxes else line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates,
          keypoint_edges=keypoint_edges,
          keypoint_edge_color=color,
          keypoint_edge_width=line_thickness // 2)

  #return image
  #2020/08/15
  return (image, detected_objects, objects_stats)

