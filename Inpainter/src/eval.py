import os
import cv2
import numpy as np
from tqdm import tqdm 
from natsort import natsorted 

def calculate_psnr(img1, img2):
  """
  Calculates the PSNR between two images.

  Args:
      img1: The first image (grayscale) as a NumPy array.
      img2: The second image (grayscale) as a NumPy array.

  Returns:
      The PSNR value in dB.
  """
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  mse = np.mean((img1 - img2) ** 2)  # Mean Squared Error
  if mse == 0:  # Avoid zero denominator
    return float('inf')
  max_pixel = 255.0
  psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
  return psnr

def calculate_ssim(img1, img2):
  """
  Calculates the SSIM between two images.

  Args:
      img1: The first image (grayscale) as a NumPy array.
      img2: The second image (grayscale) as a NumPy array.

  Returns:
      The SSIM value between 0 and 1.
  """
  C1 = (0.01 * 255) ** 2
  C2 = (0.03 * 255) ** 2

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)

  mu1 = cv2.filter2D(img1, -1, cv2.getGaussianKernel(11, 1.5))
  mu2 = cv2.filter2D(img2, -1, cv2.getGaussianKernel(11, 1.5))

  mu1_sq = mu1 ** 2
  mu2_sq = mu2 ** 2
  mu1_mu2 = mu1 * mu2

  sigma1_sq = cv2.filter2D(img1 ** 2, -1, cv2.getGaussianKernel(11, 1.5)) - mu1_sq
  sigma2_sq = cv2.filter2D(img2 ** 2, -1, cv2.getGaussianKernel(11, 1.5)) - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, cv2.getGaussianKernel(11, 1.5)) - mu1_mu2

  ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
  return np.mean(ssim)  # Average SSIM across channels

def process_images(folder1, folder2):
  """
  Calculates PSNR and SSIM for all image pairs in two folders.

  Args:
      folder1: Path to the first folder containing images.
      folder2: Path to the second folder containing images.

  Returns:
      A dictionary containing average PSNR and SSIM values.
  """
  psnr_list = []
  ssim_list = []
  for filename in tqdm(natsorted(os.listdir(folder1))):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
    #   print(filename.split("_")[0])
      img1_path = os.path.join(folder1, filename)
    #   print(str(filename.split("_")[0]) + '.png')
      #print(img1_path)
      img2_path = os.path.join(folder2, filename)  # Assuming same filenames
    #   print(str(filename.split("_")[0]) + '.png')
    #   print(img2_path)
      if os.path.exists(img2_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
          continue 
        psnr = calculate_psnr(img1, img2)
        ssim = calculate_ssim(img1, img2)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        if (len(ssim_list)==0 ) or len(psnr_list) == 0 :
          continue
        print("SSIM : " , sum(ssim_list)/len(ssim_list))
        print("PSNR : " , sum(psnr_list)/len(psnr_list))
      else:
        print(f"Error: File {filename} not found" )



process_images(r"/home/santhi/MIPI_Promptir/MIPI/gan_without_output" , r"/home/santhi/MIPI_Promptir/MIPI/Datasets/BracketFlare/test/gt")

