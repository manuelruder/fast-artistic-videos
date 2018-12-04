#!/usr/bin/env python3
import os, json, argparse
from threading import Thread
from queue import Queue

import numpy as np
from scipy.misc import imread, imresize
import h5py
from random import shuffle
import sys

"""
Create an HDF5 file of video frames, optical flow and certainty masks for training a feedforward video style transfer model.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='')
parser.add_argument('--output_file', default='video-364.h5')
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=384)
parser.add_argument('--max_images', type=int, default=-1)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--include_val', type=int, default=1)
parser.add_argument('--max_resize', default=16, type=int)
parser.add_argument('--sequence_length', default=2, type=int)
args = parser.parse_args()


def read_flow(filename):
  if len(filename) > 1:
    with open(filename, 'rb') as f:
      magic = np.fromfile(f, np.float32, count=1)
      if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
      else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        # print 'Reading %d x %d flo file' % (w, h)
        data = np.fromfile(f, np.float32, count=2*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (h, w, 2))
        return data2D


def add_data(h5_file, image_dir, prefix, args):
  # Make a list of all images in the source directory
  image_list = []
  image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'},
  
  for item in os.listdir(image_dir):
    full_path = os.path.join(image_dir, item)
    sub_folder = os.path.join(full_path, "flow")
    if os.path.exists(sub_folder):
      for filename in os.listdir(sub_folder):
        if filename.endswith(".flo"):
          frames = (os.path.splitext(filename)[0]).split('_')
          # Test if start of sequence
          if frames[0] == "s":
            if int(frames[1]) < int(frames[2]):
              image_list.append( ( full_path, int(frames[1]) ) )

  num_images = len(image_list)
  
  print("Found %d images" % num_images)

  shuffle(image_list)

  # Resize all images and copy them into the hdf5 file
  # We'll bravely try multithreading
  dset_imgs1_name = os.path.join(prefix, 'frames1')
  dset_imgs2_name = os.path.join(prefix, 'frames2')
  dset_flow_name = os.path.join(prefix, 'flow')
  dset_cert_name = os.path.join(prefix, 'cert')
  
  dset_size3 = (num_images, args.sequence_length,   3, args.height, args.width)
  dset_size2 = (num_images, args.sequence_length-1, 2, args.height, args.width)
  dset_size1 = (num_images, args.sequence_length-1,    args.height, args.width)
  imgs_dset = h5_file.create_dataset(dset_imgs1_name, dset_size3, np.uint8)
  flow_dset = h5_file.create_dataset(dset_flow_name, dset_size2, np.float32)
  cert_dset = h5_file.create_dataset(dset_cert_name, dset_size1, np.uint8)

  # input_queue stores (idx, filename) tuples,
  # output_queue stores (idx, resized_img) tuples
  input_queue = Queue()
  output_queue = Queue()
  
  # Read workers pull images off disk and resize them
  def read_worker():
    while True:
      imgs = []
      flows = []
      certs = []
      idx, frame_paths, flow_paths, cert_paths = input_queue.get()
      for frame_path in frame_paths:
        imgs.append(imread(frame_path))
      for flow_path in flow_paths:
        flows.append(read_flow(flow_path)) 
      for cert_path in cert_paths:
        certs.append(imread(cert_path))
      input_queue.task_done()
      output_queue.put((idx, imgs, flows, certs))
  
  # Write workers write resized images to the hdf5 file
  def write_worker():
    num_written = 0
    while True:
      idx, imgs, flows, certs = output_queue.get()
      # RGB image, transpose from H x W x C to C x H x W
      if imgs[0].ndim == 3:
        for i, img in enumerate(imgs):
          imgs_dset[idx,i] = img.transpose(2, 0, 1)
      elif imgs[0].ndim == 2:
        # Grayscale image; it is H x W so broadcasting to C x H x W will just copy
        # grayscale values into all channels.
        for i, img in enumerate(imgs):
          imgs_dset[idx,i] = img
      for i, flow in enumerate(flows):
        flow_dset[idx,i] = flow.transpose(2, 0, 1)
      for i, cert in enumerate(certs):
        cert_dset[idx,i] = cert.transpose(0, 1)
      output_queue.task_done()
      num_written = num_written + 1
      if num_written % 100 == 0:
        print('Copied %d / %d image sequences' % (num_written, num_images))

  # Start the read workers.
  for i in range(args.num_workers):
    t = Thread(target=read_worker)
    t.daemon = True
    t.start()
    
  # h5py locks internally, so we can only use a single write worker =(
  t = Thread(target=write_worker)
  t.daemon = True
  t.start()
    
  for idx, tuple in enumerate(image_list):
    filesTuple = []
    certsTuple = []
    flowsTuple = []
    for i in range(0, args.sequence_length):
      filesTuple.append(os.path.join(tuple[0], "frame_{:04d}.png".format(tuple[1]+i)))
    flowsTuple.append(os.path.join(tuple[0], "flow/s_{:04d}_{:04d}.flo".format(tuple[1]+1, tuple[1])))
    for i in range(1, args.sequence_length-1):
      flowsTuple.append(os.path.join(tuple[0], "flow/{:04d}_{:04d}.flo".format(tuple[1]+i+1, tuple[1]+i)))
    for i in range(0, args.sequence_length-1):
      certsTuple.append(os.path.join(tuple[0], "flow/reliable_{:04d}_{:04d}.pgm".format(tuple[1]+i+1, tuple[1]+i)))
    if args.max_images > 0 and idx >= args.max_images: break
    input_queue.put((idx, filesTuple, flowsTuple, certsTuple))

  input_queue.join()
  output_queue.join()

if __name__ == '__main__':
  
  with h5py.File(args.output_file, 'w') as f:
    add_data(f, args.input_dir, 'train', args)

    if args.include_val != 0:
      add_data(f, args.input_dir, 'val', args)

