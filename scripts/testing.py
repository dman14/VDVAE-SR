from skimage.metrics import structural_similarity as ssim
from scripts.img_helper import *
from scripts.dataloader import *

class Tester:
  def __init__(self):
    a = 1
  def setup_dataloader(self, test_path, scale = 4, 
                       reupscale = None,batch_size = 1, single = None, size = 64,
                       shuffle = False, num_workers = 0):
  
    self.dataloader_main = SRDataLoader(test_path , scale,
                                        reupscale, single,
                                        size, batch_size,
                                        shuffle, num_workers)
    self.test_dataloader = self.dataloader_main.get_dataloader()
    
  def load_model(self, model):
    self.model = model
  
  def run_test(self, model, device, test_loader, test_step, loss_func):
    self.psnr_list = []
    self.ssim_list = []
    model.eval()
    with torch.no_grad():
      for lr_batch, ref_batch in test_loader:
        output = test_step(model, device, lr_batch)
        for i in range (0,lr_batch.shape[0]):
          psnr, ssimScore = quality_measure_YCbCr(ref_batch[i], output[i])
          self.psnr_list.append(psnr)
          self.ssim_list.append(ssimScore)
    
 
  def mean_metrics(self, psnr_list, ssim_list):
    self.mean_psnr = np.mean(psnr_list)
    self.ssim_list = np.mean(ssim_list)
 
  def example_image():
    a = 1
  def main(self, test_step, device, loss_func, model, test_path,
           scale = 4, reupscale = None, single = None,
           size = 64, shuffle = False, num_workers = 0, batchsize = 1):
    
    self.setup_dataloader(test_path=test_path, scale= scale, reupscale= reupscale,
                          batch_size=batchsize, single=single, size=size,
                          shuffle=shuffle, num_workers=num_workers)

    self.load_model(model)
    self.run_test(self.model, device, self.test_dataloader,test_step, loss_func)
    self.mean_metrics(self.psnr_list, self.ssim_list)
    return self.mean_psnr, self.ssim_list