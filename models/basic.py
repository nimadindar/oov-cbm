"""Parent class for all experiments."""
import os
import torch
from torch import nn
from utils.utils import sample_code , sample_noise

class  Basic(nn.Module):
  """A basic experiment class that will be inherited by all other experiments."""

  def __init__(self, config: dict):

    super().__init__()
    self.config = config
    self.noise_dim =  config["model"]["latent_noise_dim"]
    self.num_channels =  config["dataset"]["num_channels"]
    self.has_concepts= config["model"]["has_concepts"]
    self.model_type = config["model"]["type"]
    self.num_ws = config["model"].get("num_ws", None)
    self.scale_sup_wts = config["model"].get("scale_sup_wts", False)
    self.hf_token = config["model"].get("hf_token", None)
    if(self.has_concepts):
      self.concepts_output = config["model"]["concepts"]["concept_output"]
      self.concept_type= config["model"]["concepts"]["types"]
      self.concept_name =config["model"]["concepts"]["concept_names"]
      self.concept_bins = config["model"]["concepts"]["concept_bins"]

      # self.concept_latent = config["model"]["concepts"]["concept_latent"]
      self.input_latent_dim  = config["model"]["input_latent_dim"]

      self.n_concepts =len(self.concept_name)
      # reduce one concept if unknown neurons are being used
      # but make sure unknown is the last one in the config file
      if 'unknown' in self.concept_name:
        self.n_concepts -= 1
     
      ind=0   
      self.index_per_concept=[]
      for c in range(self.n_concepts):
          if(self.concept_type[c]=="cat"):
              cat_indx = []
              for _ in range(self.concepts_output[c]):
                  cat_indx.append(ind)
                  ind+=1
              self.index_per_concept.append(cat_indx)
          elif(self.concept_type[c]=="cont"):
              self.index_per_concept.append([ind])
              ind+=1
          elif(self.concept_type[c]=="bin"):
              self.index_per_concept.append([ind])
              ind+=1

      noise_idx=[]
      for _ in range(self.noise_dim):
          noise_idx.append(ind)
          ind+=1   
      self.index_per_concept.append(noise_idx)
      self.emb_size =  config["model"]["concepts"]["emb_size"]

      self._build_model()

  def _build_model(self):
    raise NotImplementedError

  def generate_given_code(self,index,range_of_concept):
    raise NotImplementedError


  def sample_noise(self, num: int):
      return sample_noise(num, self.noise_dim, self.device)

  def sample_code(self, num: int):
      return sample_code(num, model)

  @property
  def device(self):
      return next(self.parameters()).device